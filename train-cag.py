import dill
import torch
from tqdm import tqdm
import numpy as np
from torch_geometric.seed import seed_everything
import random
import argparse
from cag import AdaptiveConditionedGraphTransformer  # Updated import
from torch_geometric.loader import DataLoader
import os
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

# Load .dill file
def load_dill(path):
    with open(path, "rb") as dill_file:
        return dill.load(dill_file)

if __name__ == "__main__":
    # Only use GPU 7
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    parser = argparse.ArgumentParser(description="Train Adaptive CGT model")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    seed = 1337
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    seed_everything(seed)

    dataset = load_dill('data/perfect/dataset')
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    print(f"Dataset loaded with {len(dataset)} graph sequences.")

    mean_features = torch.tensor([ 
            7.4666e+01, -4.3115e+01,  5.6084e+01,  1.8426e-01,  2.4458e+02,
            3.7355e-01,  9.3855e-04,  2.5639e+01,  3.0965e+02,  3.3909e+03],
            dtype=torch.float64, device=device)

    std_features = torch.tensor([
            1.5720e+00, 4.6361e+00, 1.6982e+01, 5.1707e-02, 1.6345e+00, 
            1.6973e+00, 4.4164e-03, 1.7188e-01, 4.1240e+00, 2.5931e+02], 
            dtype=torch.float64, device=device)

    model = AdaptiveConditionedGraphTransformer(node_feature_dim=10, hidden_dim=256, num_heads=4, num_graphs=20, num_spatial_layers=2).to(device)
    print("Adaptive CGT Model initialized.")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6)
    scaler = GradScaler()

    model.train()
    for epoch in tqdm(range(args.epochs)):
        total_loss = total_recon_denorm = 0
        total_geo_loss = total_thickness_loss = total_physical_loss = 0
        total_geo_loss_denorm = total_thickness_loss_denorm = total_physical_loss_denorm = 0

        for graph_list in loader:
            graph_list = [g.to(device) for g in graph_list]
            for graph in graph_list:
                graph.x = (graph.x - mean_features) / std_features
                graph.x = graph.x.float()

            a = random.randint(10, 19)
            known_layers = graph_list[:a]
            target_layers = graph_list[a:20]
            edge_index = graph_list[0].edge_index

            with autocast():
                generated_layers = model(known_layers, edge_index)

                recon_loss = geo_loss = thickness_loss = physical_loss = 0
                recon_loss_denorm = geo_loss_denorm = thickness_loss_denorm = physical_loss_denorm = 0

                for (gen_idx, x_recon), target_graph in zip(generated_layers, target_layers):
                    target_norm = target_graph.x
                    recon_loss += F.l1_loss(x_recon, target_norm)

                    geo_loss += F.l1_loss(x_recon[:, 0:2], target_norm[:, 0:2])
                    thickness_loss += F.l1_loss(x_recon[:, 2], target_norm[:, 2])
                    physical_loss += F.l1_loss(x_recon[:, 3:], target_norm[:, 3:])

                    target_denorm = target_graph.x * std_features + mean_features
                    x_recon_denorm = x_recon * std_features + mean_features
                    recon_loss_denorm += F.l1_loss(x_recon_denorm, target_denorm)
                    geo_loss_denorm += F.l1_loss(x_recon_denorm[:, 0:2], target_denorm[:, 0:2])
                    thickness_loss_denorm += F.l1_loss(x_recon_denorm[:, 2], target_denorm[:, 2])
                    physical_loss_denorm += F.l1_loss(x_recon_denorm[:, 3:], target_denorm[:, 3:])

                loss = thickness_loss + physical_loss 

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_recon_denorm += recon_loss_denorm.item()
            total_geo_loss += geo_loss.item()
            total_thickness_loss += thickness_loss.item()
            total_physical_loss += physical_loss.item()
            total_geo_loss_denorm += geo_loss_denorm.item()
            total_thickness_loss_denorm += thickness_loss_denorm.item()
            total_physical_loss_denorm += physical_loss_denorm.item()

        avg_loss = total_loss / len(loader)
        avg_recon_denorm = total_recon_denorm / len(loader)
        avg_geo_loss = total_geo_loss / len(loader)
        avg_thickness_loss = total_thickness_loss / len(loader)
        avg_physical_loss = total_physical_loss / len(loader)
        avg_geo_loss_denorm = total_geo_loss_denorm / len(loader)
        avg_thickness_loss_denorm = total_thickness_loss_denorm / len(loader)
        avg_physical_loss_denorm = total_physical_loss_denorm / len(loader)

        print(f"Epoch {epoch+1}/{args.epochs}, Known Layers: {a}, Loss: {avg_loss:.4f}, Recon Loss (Denorm): {avg_recon_denorm:.4f}")
        print(f"Geo Loss: {avg_geo_loss:.4f}, Thickness Loss: {avg_thickness_loss:.4f}, Physical Loss: {avg_physical_loss:.4f}")
        print(f"Geo Loss (Denorm): {avg_geo_loss_denorm:.4f}, Thickness Loss (Denorm): {avg_thickness_loss_denorm:.4f}, Physical Loss (Denorm): {avg_physical_loss_denorm:.4f}")
        lr_scheduler.step(avg_loss)

    os.makedirs('model_weights', exist_ok=True)
    torch.save(model.state_dict(), 'model_weights/cgt_model.pth')
    print("Adaptive CGT Model saved to model_weights/cgt_model.pth")
