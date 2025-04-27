import dill
import torch
import numpy as np
from graphautoregressive import AutoRegressiveGraphGenerator
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

def load_dill(path):
    with open(path, "rb") as dill_file:
        return dill.load(dill_file)

if __name__ == "__main__":
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    dataset = load_dill('data-pretrain/dataset')
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"Dataset loaded with {len(dataset)} graph sequences.")

    # Normalization parameters
    mean_features = torch.tensor([ 
            7.4666e+01, -4.3115e+01,  5.6084e+01,  1.8426e-01,  2.4458e+02,
            3.7355e-01,  9.3855e-04,  2.5639e+01,  3.0965e+02,  3.3909e+03],
            dtype=torch.float32, device=device)

    std_features = torch.tensor([
            1.5720e+00, 4.6361e+00, 1.6982e+01, 5.1707e-02, 1.6345e+00, 
            1.6973e+00, 4.4164e-03, 1.7188e-01, 4.1240e+00, 2.5931e+02], 
            dtype=torch.float32, device=device)

    # Load model
    model = AutoRegressiveGraphGenerator(node_feature_dim=10, hidden_dim=256, num_graphs=20).to(device)
    model.load_state_dict(torch.load('model_weights/autoreg_graph.pth'))
    model.eval()
    print("Model loaded.")

    # Evaluation loop
    total_geo_loss = 0
    total_thickness_loss = 0
    total_physical_loss = 0
    total_total_loss = 0  # All 10 features

    total_geo_loss_denorm = 0
    total_thickness_loss_denorm = 0
    total_physical_loss_denorm = 0
    total_total_loss_denorm = 0

    with torch.no_grad():
        for graph_list in loader:
            graph_list = [g.to(device) for g in graph_list]

            for graph in graph_list:
                graph.x = (graph.x - mean_features) / std_features
                graph.x = graph.x.float()

            # Example known layers: 13
            known_layers = graph_list[:13]
            target_layers = graph_list[13:]
            edge_index = graph_list[0].edge_index

            # Generate missing layers
            generated_layers = model(known_layers, edge_index)

            for (gen_idx, x_recon), target_graph in zip(generated_layers, target_layers):
                target_norm = target_graph.x

                # Geo Loss (Lat/Lon)
                loss_geo = F.l1_loss(x_recon[:, 0:2], target_norm[:, 0:2])
                total_geo_loss += loss_geo.item()

                # Thickness Loss (feature 2)
                loss_thickness = F.l1_loss(x_recon[:, 2], target_norm[:, 2])
                total_thickness_loss += loss_thickness.item()

                # Physical Features Loss (features 3:10)
                loss_physical = F.l1_loss(x_recon[:, 3:], target_norm[:, 3:])
                total_physical_loss += loss_physical.item()

                # Total Loss (All 10 features)
                loss_total = F.l1_loss(x_recon, target_norm)
                total_total_loss += loss_total.item()

                # Denormalized
                target_denorm = target_graph.x * std_features + mean_features
                x_recon_denorm = x_recon * std_features + mean_features

                # Geo Loss Denorm
                loss_geo_denorm = F.l1_loss(x_recon_denorm[:, 0:2], target_denorm[:, 0:2])
                total_geo_loss_denorm += loss_geo_denorm.item()

                # Thickness Loss Denorm
                loss_thickness_denorm = F.l1_loss(x_recon_denorm[:, 2], target_denorm[:, 2])
                total_thickness_loss_denorm += loss_thickness_denorm.item()

                # Physical Features Loss Denorm
                loss_physical_denorm = F.l1_loss(x_recon_denorm[:, 3:], target_denorm[:, 3:])
                total_physical_loss_denorm += loss_physical_denorm.item()

                # Total Loss Denorm
                loss_total_denorm = F.l1_loss(x_recon_denorm, target_denorm)
                total_total_loss_denorm += loss_total_denorm.item()

    num_sequences = len(loader) * len(target_layers)
    print("\n--- Evaluation Results ---")
    print(f"Avg Geo Loss (Lat/Lon, Norm): {total_geo_loss / num_sequences:.4f}")
    print(f"Avg Thickness Loss (Norm): {total_thickness_loss / num_sequences:.4f}")
    print(f"Avg Physical Features Loss (Norm): {total_physical_loss / num_sequences:.4f}")
    print(f"Avg Total Loss (All 10, Norm): {total_total_loss / num_sequences:.4f}")
    print(f"Avg Geo Loss (Lat/Lon, Denorm): {total_geo_loss_denorm / num_sequences:.4f}")
    print(f"Avg Thickness Loss (Denorm): {total_thickness_loss_denorm / num_sequences:.4f}")
    print(f"Avg Physical Features Loss (Denorm): {total_physical_loss_denorm / num_sequences:.4f}")
    print(f"Avg Total Loss (All 10, Denorm): {total_total_loss_denorm / num_sequences:.4f}")
