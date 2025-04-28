import dill
import torch
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from graphautoregressive import AutoRegressiveGraphGenerator
import os

# Load dill data
def load_dill(path):
    with open(path, "rb") as dill_file:
        return dill.load(dill_file)

# Save dill data
def save_dill(obj, path):
    with open(path, "wb") as dill_file:
        dill.dump(obj, dill_file)

# Apply NaN mask
def apply_nan_mask(graph_list):
    T = len(graph_list)
    N = graph_list[0].x.size(0)
    mask_matrix = torch.zeros((T, N), dtype=torch.bool)
    for t, graph in enumerate(graph_list):
        mask = torch.isnan(graph.x[:, 2])
        mask_matrix[t, mask] = True
    return mask_matrix

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

                # Correct neighbor indices
                left_idx = start_idx - 1 if start_idx > 0 else None
                right_idx = end_idx + 1 if end_idx + 1 < N else None

                print(f"\n[DEBUG] Time {t}, Masked Range: {start_idx} to {end_idx}")
                if left_idx is not None:
                    print(f"Left Neighbor Index: {left_idx}, Masked: {mask_matrix[t, left_idx]}, Value: {filled_thickness[t, left_idx].item()}")
                else:
                    print("No Left Neighbor")

                if right_idx is not None:
                    print(f"Right Neighbor Index: {right_idx}, Masked: {mask_matrix[t, right_idx]}, Value: {filled_thickness[t, right_idx].item()}")
                else:
                    print("No Right Neighbor")

                # Collect valid neighbors
                if left_idx is not None and not mask_matrix[t, left_idx]:
                    valid_thick.append(filled_thickness[t, left_idx])
                    valid_phys.append(filled_physical[t, left_idx])
                    print(f"Valid Left Neighbor Found at {left_idx}")
                if right_idx is not None and not mask_matrix[t, right_idx]:
                    valid_thick.append(filled_thickness[t, right_idx])
                    valid_phys.append(filled_physical[t, right_idx])
                    print(f"Valid Right Neighbor Found at {right_idx}")

                if valid_thick and valid_phys:
                    valid_thick_tensor = torch.stack(valid_thick, dim=0).to(device)
                    valid_phys_tensor = torch.stack(valid_phys, dim=0).to(device)

                    avg_thick_norm = valid_thick_tensor.mean()
                    avg_phys_norm = valid_phys_tensor.mean(dim=0)

                    avg_thickness_denorm = avg_thick_norm * std_thickness + mean_thickness
                    avg_phys_denorm = avg_phys_norm * std_phys + mean_phys

                    print(f"Filling with Averaged Values: Thickness={avg_thickness_denorm.item()}, Physical[0]={avg_phys_denorm[0].item()}")
                else:
                    # Fallback if no valid neighbors
                    print(f"[WARNING] No valid neighbors for filling at time {t}, indices {start_idx} to {end_idx}. Using mean values.")
                    avg_thickness_denorm = mean_thickness
                    avg_phys_denorm = mean_phys

                for i in range(start_idx, end_idx + 1):
                    filled_graph_list[t].x[i, 2] = avg_thickness_denorm
                    filled_graph_list[t].x[i, 3:10] = avg_phys_denorm
            else:
                n += 1

    return filled_graph_list



# === Set Paths ===
WEIGHTS_PATH = "model_weights/mixture/autoreg_graph.pth"
OUTPUT_PATH = "outputs/mixture/missinglayer/autoreg_filled_dataset"

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = load_dill('data/lesslayer/truncated_dataset_between_10_20.dill')
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

    # Load AR Model
    model = AutoRegressiveGraphGenerator(node_feature_dim=10, hidden_dim=256, num_graphs=20).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()

    filled_sequences = []

    with torch.no_grad():
        for graph_list in tqdm(loader):
            graph_list = [g.to(device) for g in graph_list]

            # Step 1: Apply mask for NaNs
            mask_matrix = apply_nan_mask(graph_list)

            # Step 2: Baseline fill for NaNs
            graph_list = baseline_neighbor_avg_fill(graph_list, mask_matrix, mean_features, std_features)

            # Debug: Ensure no NaNs remain in known layers
            for i, graph in enumerate(graph_list):
                if torch.isnan(graph.x).any():
                    print(f"NaN still detected in graph_list[{i}] AFTER baseline fill!")
                    raise ValueError("NaNs remain after baseline fill.")

            # Step 3: Normalize known layers
            for i, graph in enumerate(graph_list):
                graph.x = (graph.x - mean_features) / std_features
                graph.x = graph.x.float()

            T_known = len(graph_list)
            edge_index = graph_list[0].edge_index

            # Step 4: Generate missing layers with AR model
            generated_layers = model(graph_list, edge_index)

            # Debug: Check for NaNs in generated layers
            for gen_idx, x_recon in generated_layers:
                if torch.isnan(x_recon).any():
                    print(f"NaN detected in generated layer {gen_idx}! Applying fallback clamp.")
                    x_recon = torch.nan_to_num(x_recon, nan=0.0, posinf=1e3, neginf=-1e3)

            # Step 5: Combine known + generated layers
            combined_filled = []

            # De-normalize known layers
            for g in graph_list:
                g_filled = g.clone()
                g_filled.x = g_filled.x * std_features + mean_features
                combined_filled.append(g_filled.cpu())

            # De-normalize generated layers
            for (gen_idx, x_recon) in generated_layers:
                g_generated = graph_list[0].clone()
                g_generated.x = x_recon * std_features + mean_features
                combined_filled.append(g_generated.cpu())

            filled_sequences.append(combined_filled)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    save_dill(filled_sequences, OUTPUT_PATH)
    print(f"Autoregressive filled dataset saved to {OUTPUT_PATH}")
