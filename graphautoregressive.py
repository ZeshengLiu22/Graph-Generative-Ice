import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data

class AutoRegressiveGraphGenerator(nn.Module):
    def __init__(self, node_feature_dim=10, hidden_dim=256, num_graphs=20):
        '''
        Autoregressive model with Positional Encoding and LSTM.
        Args:
            node_feature_dim (int): Number of node features (e.g., 10 for your data).
            hidden_dim (int): Dimension of the hidden representations.
            num_graphs (int): Total number of graphs (layers) to generate (e.g., 20).
        '''
        super().__init__()
        self.num_graphs = num_graphs
        self.hidden_dim = hidden_dim
        self.node_feature_dim = node_feature_dim

        # GNN Encoder: Encodes node features of a graph into hidden representation.
        self.encoder_gnn = SAGEConv(node_feature_dim, hidden_dim)

        # Positional Encoding: learned per layer index [0, 1, ..., 19]
        self.positional_encoding = nn.Embedding(num_graphs, hidden_dim)

        # Temporal LSTM: Models sequential dependencies across layers.
        self.temporal_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # GNN Decoder: Decodes hidden temporal context into node feature space.
        self.decoder_gnn = SAGEConv(hidden_dim, node_feature_dim - 2)  # Only predict thickness + 7 physical

    def forward(self, known_graphs, edge_index):
        '''
        Autoregressively generates missing graph layers until reaching 20 layers.

        Args:
            known_graphs (list of Data): Known layers (Data objects with 256 nodes each).
            edge_index (Tensor): Graph connectivity (shared across layers).

        Returns:
            generated_layers (list): List of tuples (layer_index, generated_node_features).
        '''
        generated_layers = []  # Store generated layers as (index, node_features)
        hiddens = []  # Store encoded hidden representations of known layers

        # Step 1: Encode known layers with GNN + Positional Encoding
        for idx, graph in enumerate(known_graphs):
            assert graph.x.size(0) == 256, f"Graph {idx} has {graph.x.size(0)} nodes, expected 256"
            h = self.encoder_gnn(graph.x, edge_index)  # (256, hidden_dim)
            pe = self.positional_encoding(torch.tensor(idx, device=h.device))  # (hidden_dim,)
            h = h + pe.unsqueeze(0)  # (256, hidden_dim)
            hiddens.append(h.unsqueeze(0))  # (1, 256, hidden_dim)

        # Stack known layer embeddings: (num_known_layers, 256, hidden_dim)
        hiddens = torch.cat(hiddens, dim=0)

        # Permute to (256, num_known_layers, hidden_dim) for LSTM
        hiddens_for_lstm = hiddens.permute(1, 0, 2)

        # Pass through LSTM
        lstm_out, (h_t, c_t) = self.temporal_lstm(hiddens_for_lstm)  # h_t: (1, 256, hidden_dim)
        h_t = h_t.squeeze(0)  # (256, hidden_dim)

        # Start autoregressive generation
        current_graphs = known_graphs.copy()
        current_length = len(current_graphs)

        while len(current_graphs) < self.num_graphs:
            # Positional Encoding for current generation step
            pe_gen = self.positional_encoding(torch.tensor(current_length, device=h_t.device))  # (hidden_dim,)
            h_t_pe = h_t + pe_gen.unsqueeze(0)  # (256, hidden_dim)

            # Decode next layer's node features (only predict features 2:10)
            x_next_predicted = self.decoder_gnn(h_t_pe, edge_index)  # (256, 8)

            # Create full feature tensor: copy lat/lon, predict rest
            x_next = torch.zeros(256, self.node_feature_dim, device=h_t.device)
            x_next[:, 0:2] = current_graphs[-1].x[:, 0:2].detach()  # Copy lat/lon
            x_next[:, 2:] = x_next_predicted  # Predicted thickness + physical

            # Create new Data object for generated layer
            last_graph = current_graphs[-1]
            new_data = Data(
                x=x_next,
                edge_index=last_graph.edge_index,
                edge_attr=last_graph.edge_attr if 'edge_attr' in last_graph else None,
                pos=last_graph.pos if 'pos' in last_graph else None
            )
            new_data.num_nodes = 256  # Ensure node count consistency
            current_graphs.append(new_data)

            # Store generated result
            generated_layers.append((len(current_graphs) - 1, x_next))

            # Re-encode generated layer for LSTM update
            h_next = self.encoder_gnn(x_next, edge_index)  # (256, hidden_dim)
            pe_next = self.positional_encoding(torch.tensor(len(current_graphs) - 1, device=h_t.device))  # (hidden_dim,)
            h_next = h_next + pe_next.unsqueeze(0)  # (256, hidden_dim)
            h_next_seq = h_next.unsqueeze(1)  # (256, 1, hidden_dim) for LSTM

            # Update LSTM hidden state
            _, (h_t_update, c_t_update) = self.temporal_lstm(h_next_seq, (h_t.unsqueeze(0), c_t))
            h_t = h_t_update.squeeze(0)  # (256, hidden_dim)
            c_t = c_t_update

            current_length += 1  # Increment layer index

        return generated_layers  # List of generated (index, node features)
