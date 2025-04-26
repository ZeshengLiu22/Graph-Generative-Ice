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
    dataset = load_dill('data-pretrain/dataset') # Pretrain data with 20 layers
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

    # Initialize model
    model = GraphMVAE(node_feature_dim=10, hidden_dim=64, latent_dim=32, num_graphs=20).to(device)
    print("Model initialized.")

    # Optimizer and LR scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6
    )
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # Gradient scaler for mixed precision training
    scaler = GradScaler()

    # Train loop
    model.train()
    for epoch in tqdm(range(args.epochs)):
        total_loss = 0
        total_recon = 0
        total_kl = 0

        for graph_list in loader:
            graph_list = [g.to(device) for g in graph_list]

            for graph in graph_list:
                graph.x = (graph.x - mean_features) / std_features
                graph.x = graph.x.float()            

            # Create mask for observed layers
            # True indicates observed, False indicates masked
            # Randomly decide how many bottom layers to mask (e.g., 1 to 8)
            a = random.randint(1, 8) 
            mask = [True] * (model.num_graphs - a) + [False] * a

            with autocast():
                # Forward pass
                recon_layers, mu, logvar = model(graph_list, mask)

                # Reconstruction loss
                recon_loss = 0
                for idx, x_recon in recon_layers:
                    target = graph_list[idx].x * std_features + mean_features
                    x_recon_denorm = x_recon * std_features + mean_features
                    
                    recon_loss += F.mse_loss(x_recon_denorm, target)

                # KL divergence
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                # Total loss
                loss = recon_loss + 0.001 * kl_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

        # Print epoch statistics
        avg_loss = total_loss / len(loader)
        avg_recon = total_recon / len(loader)
        avg_kl = total_kl / len(loader)
        print(f"Epoch {epoch+1}/{args.epochs}, Number of Masked Layers: {a}, Loss: {avg_loss:.4f}, Recon Loss: {avg_recon:.4f}, KL Loss: {avg_kl:.4f}")
        lr_scheduler.step(avg_recon)

    # Save the model
    os.makedirs('model_weights', exist_ok=True)
    torch.save(model.state_dict(), 'model_weights/graph_mvae.pth')
    print("Model saved to model_weights/graph_mvae.pth")


