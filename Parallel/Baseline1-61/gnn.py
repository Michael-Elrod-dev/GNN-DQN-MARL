import copy
import torch
import torch.nn as nn
import torch_geometric.loader as loader

from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, TransformerConv
from torch_geometric.typing import OptPairTensor
from torch_geometric.utils import to_dense_batch


def init(module: nn.Module, weight_init, bias_init, gain: float = 1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def get_clones(module: nn.Module, N: int):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class EmbedConv(MessagePassing):
    def __init__(self,input_dim,num_embeddings,embedding_size,hidden_size,layer_N,use_orthogonal,use_ReLU,use_layerNorm,add_self_loop,edge_dim=0):
        super(EmbedConv, self).__init__(aggr="add")
        self._layer_N = layer_N
        self._add_self_loops = add_self_loop
        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        layer_norm = [nn.Identity(), nn.LayerNorm(hidden_size)][use_layerNorm]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(["tanh", "relu"][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        self.entity_embed = nn.Embedding(num_embeddings, embedding_size)
        self.lin1 = nn.Sequential(init_(nn.Linear(input_dim + embedding_size + edge_dim + 3, hidden_size)),active_func,layer_norm)
        self.lin_h = nn.Sequential(init_(nn.Linear(hidden_size, hidden_size)), active_func, layer_norm)
        self.lin2 = get_clones(self.lin_h, self._layer_N)

    def forward(self, x, edge_index, edge_attr=None):
        x: OptPairTensor = (x, x)
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        node_feat_j = x_j[..., :-1]
        entity_type_j = x_j[..., -1].long()
        entity_embed_j = self.entity_embed(entity_type_j)
        node_feat = torch.cat([node_feat_j, entity_embed_j, edge_attr], dim=-1)
        x = self.lin1(node_feat)

        for i in range(self._layer_N):
            x = self.lin2[i](x)
        return x


class TransformerConvNet(nn.Module):
    def __init__(self,input_dim: int,num_embeddings: int,embedding_size: int,hidden_size: int,num_heads: int,concat_heads: bool,layer_N: int,use_ReLU: bool,graph_aggr: str,global_aggr_type: str,embed_hidden_size: int,embed_layer_N: int,embed_use_orthogonal: bool,embed_use_ReLU: bool,embed_use_layerNorm: bool,embed_add_self_loop: bool,max_edge_dist: float,edge_dim: int = 1):
        super(TransformerConvNet, self).__init__()
        self.active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        self.edge_dim = edge_dim
        self.max_edge_dist = max_edge_dist
        self.graph_aggr = graph_aggr
        self.global_aggr_type = global_aggr_type
        self.embed_layer = EmbedConv(input_dim=input_dim,num_embeddings=num_embeddings,embedding_size=embedding_size,hidden_size=embed_hidden_size,layer_N=embed_layer_N,use_orthogonal=embed_use_orthogonal,use_ReLU=embed_use_ReLU,use_layerNorm=embed_use_layerNorm,add_self_loop=embed_add_self_loop,edge_dim=edge_dim)
        self.gnn1 = TransformerConv(in_channels=embed_hidden_size,out_channels=hidden_size,heads=num_heads,concat=concat_heads,beta=False,dropout=0.0,edge_dim=edge_dim,bias=True,root_weight=True)
        self.gnn2 = nn.ModuleList()
        
        for _ in range(layer_N):
            self.gnn2.append(self.addTCLayer(self.getInChannels(hidden_size), hidden_size))

    def forward(self, node_obs: Tensor, adj: Tensor, agent_id: Tensor):
        # convert adj to edge_index, edge_attr and then collate them into a batch
        batch_size = node_obs.shape[0]
        datalist = []
        for i in range(batch_size):
            edge_index, edge_attr = self.processAdj(adj[i])
            # if edge_attr is only one dimensional
            if len(edge_attr.shape) == 1:
                edge_attr = edge_attr.unsqueeze(1)
            datalist.append(Data(x=node_obs[i], edge_index=edge_index, edge_attr=edge_attr))
        loader_data = loader.DataLoader(datalist, shuffle=False, batch_size=batch_size)
        data = next(iter(loader_data))
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch

        if self.edge_dim is None:
            edge_attr = None

        # forward pass through embedConv
        x = self.embed_layer(x, edge_index, edge_attr)

        # forward pass through first transfomerConv
        x = self.active_func(self.gnn1(x, edge_index, edge_attr))

        # forward pass conv layers
        for i in range(len(self.gnn2)):
            x = self.active_func(self.gnn2[i](x, edge_index, edge_attr))

        # x is of shape [batch_size*num_nodes, out_channels]
        # convert to [batch_size, num_nodes, out_channels]
        x, _ = to_dense_batch(x, batch)

        # only pull the node-specific features from output
        x = self.gatherNodeFeats(x, agent_id)  # shape [batch_size, out_channels]

        return x

    def addTCLayer(self, in_channels, out_channels):
        return TransformerConv(in_channels=in_channels,out_channels=out_channels,heads=self.num_heads,concat=self.concat_heads,beta=False,dropout=0.0,edge_dim=self.edge_dim,root_weight=True)

    def getInChannels(self, out_channels):
        return out_channels + (self.num_heads - 1) * self.concat_heads * (out_channels)

    def processAdj(self, adj):
        assert adj.dim() >= 2 and adj.dim() <= 3
        assert adj.size(-1) == adj.size(-2)
        # filter far away nodes and connection to itself
        connect_mask = ((adj < self.max_edge_dist) * (adj > 0)).float()
        adj = adj * connect_mask
        index = adj.nonzero(as_tuple=True)
        edge_attr = adj[index]
        if len(index) == 3:
            batch = index[0] * adj.size(-1)
            index = (batch + index[1], batch + index[2])
        return torch.stack(index, dim=0), edge_attr

    def gatherNodeFeats(self, x, idx):
        if x.shape[0] == 1:
            return x[0, idx, :]
        else:
            batch_size = x.shape[0]
            feature_size = x.shape[-1]
            idx_expanded = idx.view(batch_size, 1, 1).expand(batch_size, 1, feature_size)
            gathered_feats = torch.gather(x, dim=1, index=idx_expanded).squeeze(1)
            return gathered_feats


class GNNBase(nn.Module):
    def __init__(self, args):
        super(GNNBase, self).__init__()
        self.args = args
        self.input_dim = args.node_obs_shape
        self.num_actions = args.action_size
        self.hidden_dim = args.gnn_hidden_size
        self.heads = args.gnn_num_heads
        self.concat = args.gnn_concat_heads

        self.gnn = TransformerConvNet(
            input_dim=args.node_obs_shape,
            edge_dim=args.edge_dim,
            num_embeddings=args.num_embeddings,
            embedding_size=args.embedding_size,
            hidden_size=args.gnn_hidden_size,
            num_heads=args.gnn_num_heads,
            concat_heads=args.gnn_concat_heads,
            layer_N=args.gnn_layer_N,
            use_ReLU=args.gnn_use_ReLU,
            graph_aggr=args.graph_aggr,
            global_aggr_type=args.global_aggr_type,
            embed_hidden_size=args.embed_hidden_size,
            embed_layer_N=args.embed_layer_N,
            embed_use_orthogonal=args.use_orthogonal,
            embed_use_ReLU=args.embed_use_ReLU,
            embed_use_layerNorm=args.use_feat_norm,
            embed_add_self_loop=args.embed_add_self_loop,
            max_edge_dist=args.max_edge_dist,
        )

        self.fc1 = nn.Linear(self.out_dim, self.hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_dim, self.num_actions)

    def forward(self, obs, node_obs, adj, agent_id):
        x = self.gnn(node_obs, adj, agent_id)
        if obs.shape[0] > 1:
            x = torch.cat((obs, x), dim=-1)
        else:
            x = torch.cat((obs, x.unsqueeze(0)), dim=-1)
        x = self.fc1(x)
        x = self.relu(x)
        q_values = self.fc2(x)
        return q_values

    @property
    def out_dim(self):
        return self.args.gnn_hidden_size + self.args.node_obs_shape
    
    def count_layers_and_params(self):
        def count_layers(module, module_name=''):
            if isinstance(module, nn.Linear):
                return {'linear': 1}, module_name
            elif isinstance(module, nn.Embedding):
                return {'embedding': 1}, module_name
            elif isinstance(module, TransformerConv):
                return {'transformer': 1}, module_name
            elif isinstance(module, EmbedConv):
                embed_layers = {'embedding': 1, 'linear': 0}
                for child in module.children():
                    if isinstance(child, nn.Linear):
                        embed_layers['linear'] += 1
                    elif isinstance(child, nn.Sequential):
                        for subchild in child:
                            if isinstance(subchild, nn.Linear):
                                embed_layers['linear'] += 1
                return embed_layers, module_name
            else:
                layer_count = {}
                layer_names = []
                for name, child in module.named_children():
                    child_count, child_names = count_layers(child, f"{module_name}.{name}" if module_name else name)
                    for key, value in child_count.items():
                        layer_count[key] = layer_count.get(key, 0) + value
                    layer_names.extend(child_names if isinstance(child_names, list) else [child_names])
                return layer_count, layer_names

        def count_params(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        total_layers, layer_names = count_layers(self)
        total_params = count_params(self)

        print("Detailed Layer and Parameter Count:")
        print("===================================")

        # Count layers and params for GNN
        gnn_layers, gnn_layer_names = count_layers(self.gnn, "gnn")
        gnn_params = count_params(self.gnn)
        print("GNN Layers:")
        for layer_type, count in gnn_layers.items():
            print(f"  - {layer_type.capitalize()}: {count}")
        print(f"GNN Parameters: {gnn_params}")
        print("-----------------------------------")

        # Count layers and params for fully connected layers
        fc_layers = 2  # fc1, fc2
        fc_params = count_params(self.fc1) + count_params(self.fc2)
        print(f"Fully Connected Layers: {fc_layers}")
        print("  - fc1")
        print("  - fc2")
        print(f"Fully Connected Parameters: {fc_params}")
        print("-----------------------------------")

        # Total counts
        print("Total Layers:")
        for layer_type, count in total_layers.items():
            print(f"  - {layer_type.capitalize()}: {count}")
        print(f"Total Parameters: {total_params}")

        return total_layers, total_params

    def get_gnn_structure(self):
        def get_structure(module, indent=0):
            if isinstance(module, (nn.Linear, nn.Conv2d, TransformerConv, EmbedConv, nn.Embedding)):
                return f"{'  ' * indent}{module.__class__.__name__}: {module}\n"
            else:
                structure = f"{'  ' * indent}{module.__class__.__name__}:\n"
                for name, child in module.named_children():
                    structure += get_structure(child, indent + 1)
                return structure

        gnn_structure = get_structure(self.gnn)
        
        full_structure = "GNNBase Structure:\n"
        full_structure += "==================\n"
        full_structure += "GNN:\n"
        full_structure += gnn_structure
        full_structure += "Fully Connected Layers:\n"
        full_structure += f"  fc1: {self.fc1}\n"
        full_structure += f"  fc2: {self.fc2}\n"
        
        return full_structure