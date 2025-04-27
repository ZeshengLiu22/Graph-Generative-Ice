import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv
from torch_geometric.data import Data

class AdaptiveConditionedGraphTransformer(nn.Module):
    def __init__(self, node_feature_dim=10, hidden_dim=256, num_heads=4, num_graphs=20, num_spatial_layers=2):
        '''
        Adaptive Conditioned Graph Transformer (CGT) Model with Lat/Lon sharing.

        Args:
            node_feature_dim (int): Number of node features (10 for your data).
            hidden_dim (int): Hidden dimension size.
            num_heads (int): Number of attention heads.
            num_graphs (int): Total number of layers to generate (e.g., 20).
            num_spatial_layers (int): Depth of spatial transformer layers.
        '''
        super().__init__()
        self.num_graphs = num_graphs
        self.hidden_dim = hidden_dim
        self.node_feature_dim = node_feature_dim

        # Spatial Graph Transformer Layers (stacked)
        self.spatial_transformers = nn.ModuleList([
            TransformerConv(node_feature_dim if i == 0 else hidden_dim, hidden_dim, heads=num_heads, concat=False)
            for i in range(num_spatial_layers)
        ])

        # Positional Encoding for temporal ordering
        self.positional_encoding = nn.Embedding(num_graphs, hidden_dim)

        # Temporal Attention across known/generated layers
        self.temporal_attn = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)

        # Decoder: maps temporal context to node features (excluding lat/lon)
        self.decoder = nn.Linear(hidden_dim, node_feature_dim - 2)  # Only predict thickness + physical (8 dims)

    def forward(self, known_graphs, edge_index):
        '''
        Autoregressively generates missing layers until reaching num_graphs.

        Args:
            known_graphs (list of Data): Known layers (each with 256 nodes).
            edge_index (Tensor): Graph connectivity (shared across layers).

        Returns:
            generated_layers (list): List of tuples (layer_index, generated_node_features).
        '''
        generated_layers = []
        spatial_hidden = []

        # Step 1: Extract shared lat/lon from last known layer
        shared_latlon = known_graphs[-1].x[:, 0:2]  # (256, 2), assumed shared for all generated layers

        # Step 2: Encode known layers spatially + positional
        for idx, graph in enumerate(known_graphs):
            x = graph.x  # (256, node_feature_dim)
            for conv in self.spatial_transformers:
                x = conv(x, edge_index)  # (256, hidden_dim)

            pe = self.positional_encoding(torch.tensor(idx, device=x.device))  # (hidden_dim,)
            x = x + pe.unsqueeze(0)  # (256, hidden_dim)
            spatial_hidden.append(x.unsqueeze(1))  # (256, 1, hidden_dim)

        # Start autoregressive generation
        current_length = len(known_graphs)
        current_graphs = known_graphs.copy()

        while current_length < self.num_graphs:
            # Stack known/generated hidden representations
            h = torch.cat(spatial_hidden, dim=1)  # (256, T_current, hidden_dim)

            # Temporal attention across current layers
            h_temporal = self.temporal_attn(h)  # (256, T_current, hidden_dim)

            # Use last temporal context for next prediction
            h_context = h_temporal[:, -1, :]  # (256, hidden_dim)

            # Decode to next layer's non-geo features (thickness + physical)
            x_next_non_geo = self.decoder(h_context)  # (256, 8)

            # Concatenate shared lat/lon
            x_next = torch.cat([shared_latlon, x_next_non_geo], dim=1)  # (256, 10)

            # Create Data object for new layer
            last_graph = current_graphs[-1]
            new_data = Data(
                x=x_next,
                edge_index=last_graph.edge_index,
                edge_attr=last_graph.edge_attr if 'edge_attr' in last_graph else None,
                pos=last_graph.pos if 'pos' in last_graph else None
            )
            new_data.num_nodes = 256
            current_graphs.append(new_data)
            generated_layers.append((current_length, x_next))

            # Encode the generated layer for temporal memory
            x_gen_spatial = x_next
            for conv in self.spatial_transformers:
                x_gen_spatial = conv(x_gen_spatial, edge_index)

            pe_gen = self.positional_encoding(torch.tensor(current_length, device=x_next.device))
            x_gen_spatial = x_gen_spatial + pe_gen.unsqueeze(0)

            spatial_hidden.append(x_gen_spatial.unsqueeze(1))  # Append to temporal context
            current_length += 1  # Move to next step

        return generated_layers
