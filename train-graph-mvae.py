import dill
import torch
from tqdm import tqdm
import numpy as np
from torch_geometric.seed import seed_everything
import random
import argparse
from graphmvae import GraphMVAE
from torch_geometric.loader import DataLoader
import os
import torch.nn.functional as F
import copy
from torch.cuda.amp import autocast, GradScaler



def load_dill(path):
    with open(path, "rb") as dill_file:
        return dill.load(dill_file)


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Train Graph-MVAE model")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    # parser.add_argument('--pretrained_weights', type=str, default=None)
    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Seed
    seed = 1337
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    seed_everything(seed)

    # Load dataset
    dataset = load_dill('data-pretrain-no-physics/dataset') # Pretrain data with 20 layers
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    print(f"Dataset loaded with {len(dataset)} graph sequences.")

    mean_features = torch.tensor([7.4666e+01, -4.3115e+01,  5.6084e+01], dtype=torch.float64, device=device)
    
    std_features = torch.tensor([1.5720e+00, 4.6361e+00, 1.6982e+01], dtype=torch.float64, device=device)

    # Initialize model
    model = GraphMVAE(node_feature_dim=3, hidden_dim=256, latent_dim=128, num_graphs=20).to(device)
    print("Model initialized.")

    # Optimizer and LR scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6
    )
    # Gradient scaler for mixed precision training
    scaler = GradScaler()

    # Train loop
    model.train()
    for epoch in tqdm(range(args.epochs)):
        total_loss = 0
        total_recon = 0
        total_recon_denorm = 0
        total_kl = 0

        for graph_list in loader:
            graph_list = [g.to(device) for g in graph_list]

            for graph in graph_list:
                graph.x = (graph.x - mean_features) / std_features
                graph.x = graph.x.float()            

            # Create mask for observed layers
            # True indicates observed, False indicates masked
            # Randomly decide how many bottom layers to mask (e.g., 1 to 8)
            # a = random.randint(1, 8) 
            a = 3
            mask = [True] * (model.num_graphs - a) + [False] * a

            with autocast():
                # Forward pass
                recon_layers, mu, logvar = model(graph_list, mask)

                # Reconstruction loss
                recon_loss = 0
                recon_loss_denorm = 0 # For logging the denormalized loss
                for idx, x_recon in recon_layers:
                    target_norm = graph_list[idx].x
                    recon_loss += F.l1_loss(x_recon[:, 2], target_norm[:, 2])        # Thickness (feature 2)

                    target_denorm = graph_list[idx].x * std_features + mean_features
                    x_recon_denorm = x_recon * std_features + mean_features
                    recon_loss_denorm += F.l1_loss(x_recon_denorm[:, 2], target_denorm[:, 2])

                # KL divergence
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                # Total loss
                loss = recon_loss + 0.01 * kl_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            total_recon_denorm += recon_loss_denorm.item()

        # Print epoch statistics
        avg_loss = total_loss / len(loader)
        avg_recon = total_recon / len(loader)
        avg_kl = total_kl / len(loader)
        avg_recon_denorm = total_recon_denorm / len(loader)

        print(f"Epoch {epoch+1}/{args.epochs}, Number of Masked Layers: {a}, Loss: {avg_loss:.4f}, Recon Loss: {avg_recon:.4f}, KL Loss: {avg_kl:.4f}, Recon Loss (Denorm): {avg_recon_denorm:.4f}")
        lr_scheduler.step(avg_recon)

    # Save the model
    os.makedirs('model_weights', exist_ok=True)
    torch.save(model.state_dict(), 'model_weights/graph_mvae.pth')
    print("Model saved to model_weights/graph_mvae.pth")


