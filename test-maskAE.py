import dill
import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from MaskedAutoEncoder import MaskedAutoEncoder
import os
'''
This script is bug free and works as intended.
But we need a better way to generate graph with NaN values.
Currently, the graph generation part is broken.
'''

# Load dill data
def load_dill(path):
    with open(path, "rb") as dill_file:
        return dill.load(dill_file)

# Save dill data
def save_dill(obj, path):
    with open(path, "wb") as dill_file:
        dill.dump(obj, dill_file)

# Apply masking
def apply_mask(graph_list, strategy='nan'):
    T = len(graph_list)
    N, F = graph_list[0].x.shape
    mask_matrix = torch.zeros((T, N), dtype=torch.bool)
    masked_graph_list = []
    for t, graph in enumerate(graph_list):
        x = graph.x.clone()
        if strategy == 'nan':
            mask = torch.isnan(x[:, 2])
        else:
            raise ValueError("Only 'nan' strategy is allowed.")
        x[mask, 2:10] = 0.0
        graph.x = x
        mask_matrix[t, mask] = True
        masked_graph_list.append(graph)
    return masked_graph_list, mask_matrix

# Baseline fill
def baseline_neighbor_avg_fill(graph_list, mask_matrix, mean_features, std_features):
    device = mean_features.device
    T, N = mask_matrix.shape
    gt = torch.stack([g.x for g in graph_list], dim=0).to(device)
    filled_thickness = gt[:, :, 2].clone()
    filled_physical = gt[:, :, 3:10].clone()

    std_thickness = std_features[2]
    mean_thickness = mean_features[2]
    std_phys = std_features[3:10]
    mean_phys = mean_features[3:10]

    filled_graph_list = [g.clone() for g in graph_list]

    for t in range(T):
        n = 0
        while n < N:
            if mask_matrix[t, n]:
                start_idx = n
                while n < N and mask_matrix[t, n]:
                    n += 1
                end_idx = n - 1

                valid_thick = []
                valid_phys = []

                left_idx = start_idx - 1 if start_idx > 0 else None
                right_idx = n if n < N else None

                if left_idx is not None and not mask_matrix[t, left_idx]:
                    valid_thick.append(filled_thickness[t, left_idx])
                    valid_phys.append(filled_physical[t, left_idx])

                if right_idx is not None and not mask_matrix[t, right_idx]:
                    valid_thick.append(filled_thickness[t, right_idx])
                    valid_phys.append(filled_physical[t, right_idx])

                if valid_thick and valid_phys:
                    valid_thick_tensor = torch.stack(valid_thick, dim=0).to(device)
                    valid_phys_tensor = torch.stack(valid_phys, dim=0).to(device)

                    avg_thick_norm = valid_thick_tensor.mean()
                    avg_phys_norm = valid_phys_tensor.mean(dim=0)

                    avg_thickness_denorm = avg_thick_norm * std_thickness + mean_thickness
                    avg_phys_denorm = avg_phys_norm * std_phys + mean_phys

                    for i in range(start_idx, end_idx + 1):
                        filled_graph_list[t].x[i, 2] = avg_thickness_denorm.cpu()
                        filled_graph_list[t].x[i, 3:10] = avg_phys_denorm.cpu()
            else:
                n += 1

    return filled_graph_list

# === Set Paths ===
WEIGHTS_PATH = "model_weights/mixture/mae_graph.pth"
OUTPUT_PATH_BASE = "outputs/mixture/nan/filled_dataset"

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = load_dill('data-nan/dataset')
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"Dataset loaded with {len(dataset)} sequences.")

    mean_features = torch.tensor([ 
        7.4666e+01, -4.3115e+01,  5.6084e+01,  1.8426e-01,  2.4458e+02,
        3.7355e-01,  9.3855e-04,  2.5639e+01,  3.0965e+02,  3.3909e+03],
        dtype=torch.float64, device=device)
    std_features = torch.tensor([
        1.5720e+00, 4.6361e+00, 1.6982e+01, 5.1707e-02, 1.6345e+00, 
        1.6973e+00, 4.4164e-03, 1.7188e-01, 4.1240e+00, 2.5931e+02], 
        dtype=torch.float64, device=device)

    model = MaskedAutoEncoder(in_dim=10, hidden_dim=128, depth=2).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()

    baseline_filled_all = []
    model_filled_all = []

    with torch.no_grad():
        for graph_list in tqdm(loader):
            assert len(graph_list) == 20
            for graph in graph_list:
                graph.x = (graph.x - mean_features.cpu()) / std_features.cpu()
                graph.x = graph.x.float()

            original_graph_list = [g.clone() for g in graph_list]
            masked_graphs, mask_matrix = apply_mask(graph_list, strategy='nan')
            mask_matrix = mask_matrix.to(device)
            masked_graphs = [g.to(device) for g in masked_graphs]

            # Baseline fill
            filled_baseline_graphs = baseline_neighbor_avg_fill(
                original_graph_list, mask_matrix, mean_features.cpu(), std_features.cpu()
            )
            baseline_filled_all.append(filled_baseline_graphs)

            pred_thickness, pred_phys = model(masked_graphs)
            pred_thickness_denorm = pred_thickness * std_features[2] + mean_features[2]
            pred_phys_denorm = pred_phys * std_features[3:10] + mean_features[3:10]

            filled_model_graphs = []
            for t in range(len(graph_list)):
                g_filled = graph_list[t].clone().to(device)
                for n in range(g_filled.num_nodes):
                    if mask_matrix[t, n]:
                        g_filled.x[n, 2] = pred_thickness_denorm[t, n]
                        g_filled.x[n, 3:10] = pred_phys_denorm[t, n]
                    else:
                        g_filled.x[n, 2] = g_filled.x[n, 2] * std_features[2] + mean_features[2]
                        g_filled.x[n, 3:10] = g_filled.x[n, 3:10] * std_features[3:10] + mean_features[3:10]
                filled_model_graphs.append(g_filled.cpu())  # Move to CPU only for saving

            model_filled_all.append(filled_model_graphs)

    baseline_output_path = OUTPUT_PATH_BASE + "_baseline.dill"
    model_output_path = OUTPUT_PATH_BASE + "_model.dill"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(baseline_output_path), exist_ok=True)


    save_dill(baseline_filled_all, baseline_output_path)
    save_dill(model_filled_all, model_output_path)

    print(f"Baseline-filled dataset saved to {baseline_output_path}")
    print(f"Model-filled dataset saved to {model_output_path}")
