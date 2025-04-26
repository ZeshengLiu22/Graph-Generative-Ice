import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv



class EncoderGNN(nn.Module):
    '''
    A Graph Neural Network (GNN) encoder using SAGEConv layers.
    This encoder processes the input node features and edge indices
    to produce a normalized output representation.
    Temporarily, we use the same encoder for all 20 layers.
    Input:
        - x: Node features of shape [num_nodes, num_node_features]
        - edge_index: Graph connectivity in COO format of shape [2, num_edges]
    Output:
        - h: Encoded node features of shape [num_nodes, hidden_dim]
    '''
    def __init__(self, in_dim, hidden_dim=128, depth=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(depth - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index):
        h = x
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
        return self.norm(h)
    

    
class TemporalGRU(nn.Module):
    '''
    A GRU-based temporal encoder that processes the output of the GNN encoder.
    This module treats each node's features across time as a sequence.
    Input:
        - x: Encoded node features of shape [num_layers, num_nodes, hidden_dim]
    Output:
        - out: Encoded node features of shape [num_layers, num_nodes, hidden_dim]

    '''
    def __init__(self, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        out, _ = self.gru(x)
        return out.permute(1, 0, 2)
    
# After the GNN encoder and temporal GRU, we have a tensor of shape [num_layers, num_nodes, hidden_dim].
# We need to decode this tensor into two different outputs: thickness and physical features.

class ThicknessDecoder(nn.Module):
    '''
    A decoder that processes the output of the temporal GRU to produce thickness predictions.
    Input:
        - x: Encoded node features of shape [num_layers, num_nodes, hidden_dim]
    Output:
        - out: Thickness predictions of shape [num_layers, num_nodes, 1]
    '''
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.fc(x)

class PhysicalFeatureDecoder(nn.Module):
    '''
    A decoder that processes the output of the temporal GRU to produce physical feature predictions.
    Input:
        - x: Encoded node features of shape [num_layers, num_nodes, hidden_dim]
    Output:
        - out: Physical feature predictions of shape [num_layers, num_nodes, num_physical_features]
    '''
    def __init__(self, hidden_dim, num_physical_features=7):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_physical_features)
        )

    def forward(self, x):
        return self.fc(x)
    

class MaskedAutoEncoder(nn.Module):
    '''
    A Masked Autoencoder for graph data.
    This model consists of a GNN encoder, a temporal GRU, and two decoders for thickness and physical features.
    Input:
        - graph_list: List of graphs, each represented by node features and edge indices
    Output:
        - thickness_out: Thickness predictions of shape [num_layers, num_nodes, 1]
        - physical_out: Physical feature predictions of shape [num_layers, num_nodes, num_physical_features]
    '''
    def __init__(self, in_dim, hidden_dim=128, depth=2):
        super().__init__()
        self.encoder = EncoderGNN(in_dim, hidden_dim, depth)
        self.temporal_gru = TemporalGRU(hidden_dim)
        self.thickness_decoder = ThicknessDecoder(hidden_dim)
        self.physical_decoder = PhysicalFeatureDecoder(hidden_dim)

    def forward(self, graph_list):
        '''
        graph_list contains 20 Data objects(20 masked graphs), each with the following attributes:
        - x: Node features of shape [num_nodes, num_node_features]
        - edge_index: Graph connectivity in COO format of shape [2, num_edges]
        '''

        T = len(graph_list) # Number of layers, should be 20
        N = graph_list[0].x.shape[0] # Number of nodes, should be 256

        all_encoded = []
        # Extract edge_index from first graph
        edge_index = graph_list[0].edge_index
        for graph in graph_list:
            x = graph.x # Masked node features of shape [num_nodes, num_node_features], should be (N, 10)

            h = self.encoder(x, edge_index) # Encoded node features of shape [num_nodes, hidden_dim]
            all_encoded.append(h)

        encoded = torch.stack(all_encoded, dim=0) # Shape: [T, N, hidden_dim]
        h_temporal = self.temporal_gru(encoded) # Shape: [T, N, hidden_dim]

        pred_thickness = self.thickness_decoder(h_temporal) # Shape: [T, N, 1]
        pred_physical = self.physical_decoder(h_temporal) # Shape: [T, N, num_physical_features]

        return pred_thickness, pred_physical