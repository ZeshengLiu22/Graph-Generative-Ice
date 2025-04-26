import dill
import torch
from tqdm import tqdm
import numpy as np
from torch_geometric.seed import seed_everything
import random
import argparse
from MaskedAutoEncoder import MaskedAutoEncoder
from torch_geometric.loader import DataLoader
import os
import torch.nn.functional as F
import copy

# def save_dill(obj, path):
#     with open(path, "wb") as dill_file:
#         dill.dump(obj, dill_file)

def load_dill(path):
    with open(path, "rb") as dill_file:
        return dill.load(dill_file)
    

# Function to create a mask for a graph_list
def apply_mask(graph_list, mask_prob=0.15, strategy='random'):
    '''
    Apply a mask to the graph_list based on the specified strategy.
    Args:
        graph_list (list): List of graphs to be masked.
        mask_prob (float): Probability of masking a node.
        strategy (str): Masking strategy ('random', 'nan').
    '''

    T = len(graph_list) # Number of graphs, should be 20
    N, F = graph_list[0].x.shape # Number of nodes and features, should be 256, 10

    mask_matrix = torch.zeros((T, N), dtype=torch.bool) # Mask matrix to store the mask for each graph

    masked_graph_list = []
    for t, graph in enumerate(graph_list):
        x = graph.x.clone()

        if strategy == 'random':
            mask = torch.rand(N) < mask_prob
        elif strategy == 'nan':
            mask = torch.isnan(x[:, 2])
        else:
            raise ValueError("Invalid masking strategy. Choose 'random' or 'nan'.")
        
        x[mask, 2:10] = 0.0  # Set masked values to 0
        graph.x = x
        mask_matrix[t, mask] = True
        masked_graph_list.append(graph)

    return masked_graph_list, mask_matrix

def baseline_neighbor_avg_fill(graph_list, mask_matrix):
    '''
    Baseline fill: for masked nodes, use the average of neighboring nodes (index-based).
    Args:
        graph_list: original unmasked graphs (ground truth).
        mask_matrix: boolean mask (T, N), True = masked node.
    Returns:
        mse_baseline: mean squared error of baseline fill.
    '''

    T, N = mask_matrix.shape # (20, 256)
    gt = torch.stack([g.x for g in graph_list], dim=0) # (20, 256, 10)
    filled_thickness = gt[:, :, 2].clone() # (20, 256, 7)
    filled_physical = gt[:, :, 3:10].clone() # (20, 256, 1)


    mse_thick_total = 0.0
    mse_phys_total = 0.0
    count = 0

    for t in range(T):
        for n in range(N):
            if mask_matrix[t, n]:
                neighbors_thick = []
                neighbors_phys = []

                if n > 0:
                    neighbors_thick.append(filled_thickness[t, n-1].item())
                    neighbors_phys.append(filled_physical[t, n-1].cpu().numpy())
                if n < N - 1:
                    neighbors_thick.append(filled_thickness[t, n+1].item())
                    neighbors_phys.append(filled_physical[t, n+1].cpu().numpy())

                if neighbors_thick and neighbors_phys:
                    avg_thick = sum(neighbors_thick) / len(neighbors_thick)

                    neighbors_phys_array = np.stack(neighbors_phys, axis=0)
                    avg_phys = np.mean(neighbors_phys_array, axis=0)


                    mse_thick_total += (avg_thick - gt[t, n, 2].item()) ** 2
                    mse_phys_total += np.sum((avg_phys - gt[t, n, 3:10].cpu().numpy()) ** 2)

                    count += 1

    mse_thickness = mse_thick_total / count if count > 0 else 0.0
    mse_physical = mse_phys_total / (count * 7) if count > 0 else 0.0

    return mse_thickness, mse_physical

    
if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description='Train MaskAutoEncoder')
    parser.add_argument('--mode', type=str, default='mae_pretrain', choices=['mae_pretrain', 'fine_tune', 'nan_only'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--pretrained_weights', type=str, default=None)
    args = parser.parse_args()

    # Set the device for PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set the random seed for reproducibility
    seed = 1337
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    seed_everything(seed)

    # Training parameters
    epochs = args.epochs
    lr = args.lr
    mode = args.mode
    print(f"Training mode: {mode}")

    # Load the dataset, and set the mean and std for normalization
    if mode == 'mae_pretrain':
        dataset = load_dill('data-l2-pretrain/dataset')
        
        # mean_features = torch.tensor([ 
        #     7.4666e+01, -4.3115e+01,  5.6084e+01,  1.8426e-01,  2.4458e+02,
        #     3.7355e-01,  9.3855e-04,  2.5639e+01,  3.0965e+02,  3.3909e+03],
        #     dtype=torch.float64)

        mean_features = torch.tensor([ 
        7.6795e+01, -4.9625e+01,  5.8094e+01,  2.1486e-01,  2.4549e+02,
        1.3217e+00,  3.4205e-03,  2.5664e+01,  3.1252e+02,  3.0169e+03],
        dtype=torch.float64)

        # std_features = torch.tensor([
        #     1.5720e+00, 4.6361e+00, 1.6982e+01, 5.1707e-02, 1.6345e+00, 
        #     1.6973e+00, 4.4164e-03, 1.7188e-01, 4.1240e+00, 2.5931e+02], 
        #     dtype=torch.float64)

        std_features = torch.tensor([
        8.7640e-01, 5.2234e+00, 1.9439e+01, 7.6635e-02, 1.9833e+00, 3.4144e+00,
        8.8475e-03, 1.9078e-01, 5.5896e+00, 2.8551e+02], 
        dtype=torch.float64)
    else:
        dataset = load_dill('data-l2-nan/dataset')
        mean_features = torch.tensor([ 
            7.6361e+01, -4.5710e+01, float('nan'), 1.7543e-01, 2.4448e+02,
            4.1272e-01,  1.0003e-03, 2.5617e+01,  3.1043e+02,  3.1965e+03],
            dtype=torch.float64)
        std_features = torch.tensor([
            1.3037e+00, 3.8302e+00, float('nan'), 6.5598e-02, 1.6997e+00, 
            2.2117e+00, 5.3461e-03, 1.7652e-01, 4.4067e+00, 2.1875e+02], 
            dtype=torch.float64)
        
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    print(f"Dataset loaded with {len(dataset)} graphs.")

    # Initialize the model
    model = MaskedAutoEncoder(in_dim=10, hidden_dim=128, depth=2).to(device)

    # Load the pretrained weights if provided
    if args.pretrained_weights:
        if os.path.exists(args.pretrained_weights):
            print(f"Loading pretrained weights from {args.pretrained_weights}")
            model.load_state_dict(torch.load(args.pretrained_weights, map_location=device))
        else:
            raise FileNotFoundError(f"Pretrained weights file not found: {args.pretrained_weights}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Training loop
    model.train()
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        baseline_thickness_total = 0
        baseline_physical_total = 0
        model_thickness_total = 0
        model_physical_total = 0

        for graph_list in loader:
            assert len(graph_list) == 20, "Graph list should contain 20 graphs"
            for graph in graph_list:
                graph.x = (graph.x - mean_features) / std_features
                graph.x = graph.x.float()

            original_graph_list = copy.deepcopy(graph_list)

            if mode == 'mae_pretrain':
                masked_graphs, mask_matrix = apply_mask(graph_list, mask_prob=0.4, strategy='random')
            else:
                masked_graphs, mask_matrix = apply_mask(graph_list, strategy='nan')

            # Baseline fill
            baseline_thickness, baseline_physical = baseline_neighbor_avg_fill(original_graph_list, mask_matrix)
            baseline_thickness_total += baseline_thickness
            baseline_physical_total += baseline_physical

            masked_graphs = [graph.to(device) for graph in masked_graphs]
            mask_matrix = mask_matrix.to(device)
            gt = torch.stack([g.x.float() for g in original_graph_list], dim=0).to(device)

            pred_thickness, pred_phys = model(masked_graphs)

            # Denormalize the predictions
            pred_thickness_denorm = pred_thickness * std_features[2].to(device) + mean_features[2].to(device)
            pred_phys_denorm = pred_phys * std_features[3:10].to(device) + mean_features[3:10].to(device)
            gt_thickness_denorm = gt[:, :, 2:3] * std_features[2].to(device) + mean_features[2].to(device)
            gt_phys_denorm = gt[:, :, 3:10] * std_features[3:10].to(device) + mean_features[3:10].to(device)

            if mode == 'mae_pretrain':
                # Loss is calculated both on thickness and physical features
                loss_thickness = F.mse_loss(pred_thickness_denorm[mask_matrix], gt_thickness_denorm[mask_matrix])
                loss_phys = F.mse_loss(pred_phys_denorm[mask_matrix], gt_phys_denorm[mask_matrix])
                loss = loss_thickness + loss_phys

                model_thickness_total += loss_thickness.item()
                model_physical_total += loss_phys.item()
            else:
                # Fine-tuning on data-nan or directly training on data-nan
                # Now loss is only calculated on physical features
                loss = F.mse_loss(pred_phys_denorm[mask_matrix], gt_phys_denorm[mask_matrix])
                model_physical_total += loss.item()
                model_thickness_total += 0.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        avg_baseline_thickness = baseline_thickness_total / len(loader)
        avg_baseline_physical = baseline_physical_total / len(loader)
        avg_model_thickness = model_thickness_total / len(loader)
        avg_model_physical = model_physical_total / len(loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, "
              f"Baseline Thickness: {avg_baseline_thickness:.4f}, "
              f"Baseline Physical: {avg_baseline_physical:.4f}, "
              f"Model Thickness: {avg_model_thickness:.4f}, "
              f"Model Physical: {avg_model_physical:.4f}")
        lr_scheduler.step()

    
    # Final save of the model
    print("Finishing training, saving model...")
    torch.save(model.state_dict(), f"model_weights/{mode}_final_model.pth")
    print("Model saved successfully.")
    

