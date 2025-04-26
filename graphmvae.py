import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
import random
import numpy as np

class GraphMVAE(nn.Module):
    def __init__(self, node_feature_dim=10, hidden_dim=64, latent_dim=32, num_graphs=20):
        '''
        Graph Masked Variational Autoencoder (Graph-MVAE) model.
        Take a graph list as input, where each graph is complete without any missing node features.
        Total number of graph in the list is less than 20.
        The model will generate synthetic to make it 20 graphs in the end.

        Args:
            node_feat_dim (int): Number of features per node (e.g., 10: lat, lon, thickness, physical features).
            hidden_dim (int): Hidden dimension for GNN and GRU layers.
            latent_dim (int): Size of the latent vector for sampling.
            num_graphs (int): Total number of graph in a full sequence (e.g., 20 graphs for 20 ice layers).
        '''
        super(GraphMVAE, self).__init__()
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_graphs = num_graphs

        # SAGEConv layer for encoding individual graph layers(node-wise to hidden_dim)
        self.encoder = SAGEConv(node_feature_dim, hidden_dim)

        # GRU to process temporal sequence of graph-level embeddings
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # Fully connected layers to project GRU output into latent mean and log-variance.
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder fully connected layers to combine latent vector + context into hidden features.
        # Context includes all graph-level embeddings, even zeros for masked layers.
        self.decoder_fc = nn.Linear(latent_dim + num_graphs * hidden_dim, hidden_dim)

        # SAGEConv layer to reconstruct node features from masked layers.
        self.decoder_gnn = SAGEConv(hidden_dim, node_feature_dim)

    def encode(self, graph_list, mask):
        '''
                Encode observed graph layers into latent distribution parameters.

        Args:
            graph_list (list): List of PyG Data objects (each representing a layer).
            mask (list): Boolean list indicating which layers are observed (True) or masked (False).

        Returns:
            mu (Tensor): Latent mean vector.
            logvar (Tensor): Latent log-variance vector.
            node_reprs (list): List of graph-level embeddings (hidden_dim) for all layers.
        '''

        node_reprs = []
        edge_index= graph_list[0].edge_index
        for i, data in enumerate(graph_list):
            if mask[i]: # Only process observed layers
                # Encode node features using SAGEConv, apply ReLU
                x = F.relu(self.encoder(data.x, edge_index)) # (256, hidden_dim)
                # Aggregate node features to a single graph-level embedding
                pooled = x.mean(dim=0)  # (1, hidden_dim)
                node_reprs.append(pooled)
            else:
                # For masked layers, use zero vector as placeholder
                node_reprs.append(torch.zeros(self.hidden_dim, device=data.x.device))

        # Stack graph-level embeddings to a sequence: shape (1, num_graphs, hidden_dim)
        node_seq = torch.stack(node_reprs).unsqueeze(0)

        # Process sequence using GRU to capture temporal dependencies
        _, h = self.gru(node_seq)  # h shape: (1, batch_size=1, hidden_dim)
        h = h.squeeze(0) # Remove batch dimension: shape (hidden_dim,)

        # Compute latent mean and log-variance
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar, node_reprs
    
    def reparameterize(self, mu, logvar):
        '''
                Reparameterization trick: sample from N(mu, sigma^2) using mu and logvar.

        Args:
            mu (Tensor): Latent mean vector.
            logvar (Tensor): Latent log-variance vector.

        Returns:
            z (Tensor): Sampled latent vector.
        '''
        std = torch.exp(0.5 * logvar) # Convert log-variance to standard deviation
        eps = torch.randn_like(std) # Sample epsilon ~ N(0, 1)
        return mu + eps * std # Return sampled latent vector
    
    def decode(self, z, node_reprs, graph_list, mask):
        '''
        Decode latent vector to reconstruct node features for masked layers.

        Args:
            z (Tensor): Latent vector sampled from the latent distribution.
            node_reprs (list): Graph-level embeddings for all layers.
            graph_list (list): Original list of Data objects.
            mask (list): Boolean list indicating observed (True) or masked (False) layers.

        Returns:
            reconstructed_layers (list): List of tuples (layer index, reconstructed node features).
        '''
        # Pad node_reprs dynamically if graph_list < num_graphs
        pad_len = self.num_graphs - len(node_reprs)
        if pad_len > 0:
            padding = [torch.zeros(self.hidden_dim, device=node_reprs[0].device) for _ in range(pad_len)]
            node_reprs += padding

        # Concatenate latent vector with all graph-level embeddings (flattened)
        context = torch.cat(node_reprs, dim=0) # shape: (num_graphs, hidden_dim)
        z_context = torch.cat([z.view(-1), context.view(-1)], dim=0) # shape: (latent_dim + num_graphs * hidden_dim,)

        # Pass through decoder fully connected layer to get hidden representation
        dec_input = F.relu(self.decoder_fc(z_context)) # shape: (hidden_dim,)

        reconstructed_layers = []

        edge_index = graph_list[0].edge_index
        for i, data in enumerate(graph_list):
            if not mask[i]: # Only reconstruct masked layers
                # Repeat dec_input for each node in the graph (shape: num_nodes x hidden_dim)
                dec_node_input = dec_input.unsqueeze(0).repeat(data.x.size(0), 1)

                # Reconstruct node features using SAGEConv
                reconstructed_x = self.decoder_gnn(dec_node_input, edge_index)

                # Store the reconstructed node features with their layer index
                reconstructed_layers.append((i, reconstructed_x))

        # Generate additional synthetic layers if needed
        for i in range(len(graph_list), self.num_graphs):
            template_data = graph_list[-1]  # Use last graph's structure
            dec_node_input = dec_input.unsqueeze(0).repeat(template_data.x.size(0), 1)
            reconstructed_x = self.decoder_gnn(dec_node_input, edge_index)
            reconstructed_layers.append((i, reconstructed_x))

        return reconstructed_layers

    
    def forward(self, graph_list, mask):
        '''
        Full forward pass through the Graph-MVAE.

        Args:
            graph_list (list): List of Data objects representing a sequence.
            mask (list): Boolean list indicating observed/masked layers.

        Returns:
            reconstructed_layers (list): Reconstructed node features for masked layers.
            mu (Tensor): Latent mean vector.
            logvar (Tensor): Latent log-variance vector.
        '''
        # Encode observed layers to get latent distribution parameters
        mu, logvar, node_reprs = self.encode(graph_list, mask)

        # Sample from the latent distribution
        z = self.reparameterize(mu, logvar)

        # Decode the latent vector to reconstruct masked layers
        reconstructed_layers = self.decode(z, node_reprs, graph_list, mask)

        return reconstructed_layers, mu, logvar

