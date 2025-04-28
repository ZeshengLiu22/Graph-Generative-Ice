import dill
import mat73
import scipy.io
import numpy as np
import torch
from torch_geometric.utils.convert import from_networkx
import networkx as nx
from tqdm import tqdm
import math
import os

'''
This script prepares the dataset for the Graph Generative Ice model.
It loads the data from .mat files, processes it into a format suitable for graph neural networks,
and saves the processed data into .dill files.

Known Issues:
It seems that the logic for loading the .mat files is not working as expected.
For the perfect 20 layers, it should be fine
but for datasets with perfect less than 20 layers, it seems to be loading some wrong files.

Besides, we need to think a better way to handle the NaN values.
The raw NaN values are in term of the node coordinates.
But we want to have the NaN values in terms of the node thickness.
'''

DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(DEVICE)
if DEVICE == 'cuda':
    torch.cuda.empty_cache()

torch.manual_seed(1337)

# Experiment settings
EXPERIMENT_NAME = "data-mixture-0428"

# Test L files FIRST in the list
FULL_DATA_FOLDERS = [
    r"/data/datasets/SR_Dataset_v1/test_L_files/L1/",
    r"/data/datasets/SR_Dataset_v1/test_L_files/L2/",
    r"/data/datasets/SR_Dataset_v1/test_L_files/L3/",
    r"/data/datasets/SR_Dataset_v1/train_data",
    r"/data/datasets/SR_Dataset_v1/val_for_training_data/",
]

LAYER_PREDICT_COUNT = 0
LAYER_FEATURE_COUNT = 20
PREDICT_HISTORIC = True
NODE_COUNT = 256
FEATURE_COUNT = 10
REMOVE_ALL_PHYSICAL_PARAMS = False
TOTAL_LAYERS = LAYER_FEATURE_COUNT + 1  # 21 layers (0-20)

if not os.path.exists(EXPERIMENT_NAME + "/"):
    os.mkdir(EXPERIMENT_NAME + "/")

def save_dill(obj, path):
    with open(path, "wb") as dill_file:
        dill.dump(obj, dill_file)

def smart_loadmat(filepath):
    try:
        return mat73.loadmat(filepath)
    except TypeError as e:
        if "not a MATLAB 7.3 file" in str(e):
            return scipy.io.loadmat(filepath)
        else:
            raise e

def get_distance(lat0, long0, lat1, long1):
    lat0 = math.radians(lat0)
    long0 = math.radians(long0)
    lat1 = math.radians(lat1)
    long1 = math.radians(long1)
    delta_lat = lat1 - lat0
    delta_long = long1 - long0
    h = (math.sin(delta_lat / 2) ** 2) + math.cos(lat0) * math.cos(lat1) * (math.sin(delta_long / 2) ** 2)
    d = 2 * math.asin(math.sqrt(h))
    return d

def load_dataset(mat_limit=0):
    dataset_between_10_20 = []
    dataset_layers_20_with_nan = []
    dataset_perfect_20 = []

    less_than_10_layers = 0
    between_10_and_20_layers = 0
    layers_20_with_nan = 0
    perfect_20_layers = 0

    mat_file_list = []
    for df in FULL_DATA_FOLDERS:
        mat_file_list.extend(f for f in os.listdir(df) if f.endswith(".mat"))

    for mat_file_name in tqdm(mat_file_list):
        if mat_limit > 0 and (len(dataset_between_10_20) + len(dataset_layers_20_with_nan) + len(dataset_perfect_20)) >= mat_limit:
            break

        # Correct File Path Logic
        full_path = None
        for i in range(len(FULL_DATA_FOLDERS)):
            if i >= 2:  # train_data or val_for_training_data
                full_path = os.path.join(FULL_DATA_FOLDERS[i], mat_file_name)
            else:  # test L files
                full_path = os.path.join("/data/datasets/SR_Dataset_v1/final_test_data/", mat_file_name)

            if os.path.exists(full_path):
                break
            else:
                full_path = None

        if full_path is None:
            print(f"File not found in expected locations: {mat_file_name}")
            continue
        # full_path = '/data/datasets/SR_Dataset_v1/final_test_data/20120330_04_0939_2km.mat'
        mat_data = smart_loadmat(full_path)

        valid_layers = -1
        nan_in_valid_layers = False

        for y in range(TOTAL_LAYERS):
            # print("Layer:", y)
            # print("layer vec at first 3:", mat_data["layers_vector"][y, :3])
            nan_count = np.isnan(mat_data["layers_vector"][y, :]).sum()
            # print("nan count:", nan_count)
            # print("current valid layers:", valid_layers)
            # print("-----")

            if nan_count < NODE_COUNT: # Not all nodes are NaN, we call this a valid layer
                valid_layers += 1 # Count valid layers
                if nan_count > 0: # Track if there are any NaN values in the valid layers
                    nan_in_valid_layers = True
            else:   # All nodes are NaN, we call this an invalid layer
                break # We stop counting valid layers
                # if y < 10:  # If the invalid layer is in the first 10 layers, we stop counting valid layers
                #     fully_nan_in_first_10 = True
                #     break
                # else: # If the invalid layer is after the first 10 layers, we stop counting valid layers
                #     break

        if valid_layers < 10:
            less_than_10_layers += 1
            continue

        max_valid_layers = min(valid_layers, 20)

        if max_valid_layers >= 10 and max_valid_layers < 20:
            if nan_in_valid_layers:
                continue
            between_10_and_20_layers += 1
            current_case = "between_10_and_20_and_no_nan"
        elif max_valid_layers == 20 and nan_in_valid_layers:
            layers_20_with_nan += 1
            current_case = "layers_20_with_nan"
        elif max_valid_layers == 20 and not nan_in_valid_layers:
            perfect_20_layers += 1
            current_case = "perfect_20"
        else:
            continue

        timesteps = []
        for layer in range(max_valid_layers):
            G = nx.Graph()
            layer_heights = np.zeros(LAYER_FEATURE_COUNT + LAYER_PREDICT_COUNT)
            for x in range(NODE_COUNT):
                height_sum = 0
                for i in range(LAYER_FEATURE_COUNT + LAYER_PREDICT_COUNT):
                    layer_heights[i] = mat_data["layers_vector"][i + 1, x] - height_sum
                    height_sum += layer_heights[i]

                feat_layer = layer if PREDICT_HISTORIC else (layer + LAYER_PREDICT_COUNT)

                elevation = mat_data["Elevation"][x]
                lat = mat_data["Latitude"][x]
                long = mat_data["Longitude"][x]
                height = layer_heights[feat_layer]
                smb = mat_data["curr_smb"][feat_layer + 1, x]
                temp = mat_data["curr_temp"][feat_layer + 1, x]
                density = mat_data["curr_dens"][feat_layer + 1, x]
                refreezing = mat_data["curr_RZ_vals"][feat_layer + 1, x]
                height_change = mat_data["curr_SHC_vals"][feat_layer + 1, x]
                snow_pack = mat_data["curr_SNHS_vals"][feat_layer + 1, x]

                F = [lat, long, height, smb, temp, refreezing, height_change, snow_pack, density, elevation]
                if REMOVE_ALL_PHYSICAL_PARAMS:
                    F = F[:3]

                G.add_node(x, x=F)

                if layer == 0:
                    for adj in range(x + 1, NODE_COUNT):
                        G.add_edge(x, adj, edge_weight=get_distance(lat, long, mat_data["Latitude"][adj], mat_data["Longitude"][adj]))

            min_weight = 0
            max_weight = 0
            for u, v, d in G.edges(data=True):
                if d["edge_weight"] > max_weight:
                    max_weight = d["edge_weight"]
                if d["edge_weight"] < min_weight:
                    min_weight = d["edge_weight"]

            min_weight *= 0.95
            max_weight *= 1.05

            for u, v, d in G.edges(data=True):
                d["edge_weight"] = (d["edge_weight"] - min_weight) / (max_weight - min_weight)

            data = from_networkx(G)
            if layer == 0:
                data.mat_file = full_path
            else:
                del data.edge_index
                del data.edge_weight

            timesteps.append(data)

        if current_case == "between_10_and_20":
            dataset_between_10_20.append(timesteps)
        elif current_case == "layers_20_with_nan":
            dataset_layers_20_with_nan.append(timesteps)
        elif current_case == "perfect_20":
            dataset_perfect_20.append(timesteps)


    # Save datasets separately
    save_dill(dataset_between_10_20, f"{EXPERIMENT_NAME}/dataset_between_10_20.dill")
    save_dill(dataset_layers_20_with_nan, f"{EXPERIMENT_NAME}/dataset_layers_20_with_nan.dill")
    save_dill(dataset_perfect_20, f"{EXPERIMENT_NAME}/dataset_perfect_20.dill")

    print("Less than 10 layers:", less_than_10_layers)
    print("Between 10 and 20 layers:", between_10_and_20_layers)
    print("Layers 20 with NaN:", layers_20_with_nan)
    print("Perfect 20 layers:", perfect_20_layers)

    return dataset_between_10_20, dataset_layers_20_with_nan, dataset_perfect_20

# Run the dataset loader
dataset_10_20, dataset_20_nan, dataset_20_perfect = load_dataset()
