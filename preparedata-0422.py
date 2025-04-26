import dill

def save_dill(obj, path):
    with open(path, "wb") as dill_file:
        dill.dump(obj, dill_file)

def load_dill(path):
    with open(path, "rb") as dill_file:
        return dill.load(dill_file)
    

import mat73
import scipy.io
import imageio
import numpy as np

import torch
from torch_geometric.data import HeteroData
from torch_geometric.utils.convert import from_networkx

import networkx as nx

from tqdm import tqdm
import math
import os

DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

if(DEVICE == 'cuda'):
    torch.cuda.empty_cache()

# Ignore these variables.
torch.manual_seed(1337) #123456
FEATURE_ABLATION = "1111111"
REMOVE_ALL_PHYSICAL_PARAMS = True
NODE_COUNT = 256
FEATURE_COUNT = 10

############ START HERE !

# The name of the folder to save results to.
EXPERIMENT_NAME = "data-l2-pretrain"

# The folders from which to gather .mat files.
FULL_DATA_FOLDERS = [
    # r"/data/datasets/SR_Dataset_v1/train_data",
    # r"/data/datasets/SR_Dataset_v1/val_for_training_data/",
    # r"/data/datasets/SR_Dataset_v1/test_L_files/L1/",
     r"/data/datasets/SR_Dataset_v1/test_L_files/L2/"]#,
    # r"/data/datasets/SR_Dataset_v1/test_L_files/L3/"]

# How many layer thicknesses to predict.
LAYER_PREDICT_COUNT = 0
# How many "feature layers" to use.
LAYER_FEATURE_COUNT = 20
# Whether to predict deep-to-shallow (False) or shallow-to-deep (True)
PREDICT_HISTORIC = True 

# NOTE: These 3 options should be mutually exclusive!!!
# Whether or not the experiment should run the (GCN/AGCN/GAT)-LSTM
RUN_TEMPORAL_GEOMETRIC = True
# Whether or not the experiment should run the pure LSTM.
RUN_NONGEOMETRIC_LSTM = True
# Whether or not the experiment should run the pure GCN.
RUN_NONTEMPORAL_GCN = True

# Whether or not to add an adaptive layer to the GCN(or GAT)-LSTM.
ADAPTIVE = True
# Whether to use GCN-LSTM (false) or GAT-LSTM (true)
USE_GAT = False
# Whether to use GCN-LSTM (false) or Transformer-LSTM (true)
#USE_Transformer = False
USE_TGCN = False
USE_A3TGCN = False
USE_DCRNN = False
USE_SAGE = False

# The number of channels within each of the three linear layers in the model.
DIMENSIONALITIES = [ 256, 128, 64 ]

# The initial learning rate for the model.
INITIAL_LEARNING_RATE = 0.01
# Whether or not to use a dynamic learning rate.
DYNAMIC_LEARNING_RATE = True
# How often (in epochs) to reduce the learning rate.
DYNAMIC_LEARNING_RATE_INTERVAL = 75
# By what ratio to multiply the learning rate every DYNAMIC_LEARNING_RATE_INTERVAL epochs.
DYNAMIC_LEARNING_RATE_RATIO = 0.5

# Whether or not to run a physical parameter ablation study.
# In the output experiment name, each feature is shown as either a 1 (included) or 0 (excluded).
# The order of the features is as follows:
# 1: Snow mass balance
# 2: Average yearly surface temp
# 3: Height change due to refreezing
# 4: Height change due to melt
# 5: Amount of snow pack
# 6: Snow density
# 7: Elevation
# As an example, an experiment named GAT_TEST_100001 means that snow mass balance and elevation were used but no others.
RUN_ABLATION_STUDY = True
# The name of the experiment that generated the desired dataset.
DATASET_EXPERIMENT = EXPERIMENT_NAME

# Batch Size:
BATCH_SIZE = 1#32

############ END HERE !

REMOVE_ALL_PHYSICAL_PARAMS = not RUN_ABLATION_STUDY
if REMOVE_ALL_PHYSICAL_PARAMS:
    print("No Physics Parameters!")
    FEATURE_COUNT = 3

if not os.path.exists(EXPERIMENT_NAME + "/"):
    os.mkdir(EXPERIMENT_NAME + "/")

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

    global FEATURE_COUNT

    valid_data_names = []
    dataset = []
    total_valid_data = 0
    data_with_nan = 0
    data_less_layer = 0

    #saved_dataset_path = r"/work/09182/bjz222lu/ls6/SnowRadar/Models/LS6_GCNLSTM_DeepToShallow/dataset"
    #saved_cache_path = r"/work/09182/bjz222lu/ls6/SnowRadar/Models/LS6_GCNLSTM_DeepToShallow/matcache"

    saved_dataset_path = DATASET_EXPERIMENT + "/dataset"
    saved_cache_path = DATASET_EXPERIMENT + "/matcache"
    saved_name_path = DATASET_EXPERIMENT + "/name"

    if os.path.isfile(saved_dataset_path):
        dataset = load_dill(saved_dataset_path)
        for gc in dataset:
            for i in range(len(gc)):
                gc[i] = gc[i]#.to(DEVICE)
        return dataset

    use_cache = os.path.isfile(saved_cache_path)

    mat_file_list = None
    mat_file_count = 0

    if use_cache:
        mat_file_list = load_dill(saved_cache_path)
    else:
        mat_file_list = []
        for df in FULL_DATA_FOLDERS:
            mat_file_list.extend(f for f in os.listdir(df) if f.endswith(".mat"))

    cache_list = []

    for mat_file_name in tqdm(mat_file_list):

        if mat_limit > 0 and len(dataset) >= mat_limit:
            break

        #print(f"Loading .mat file {mat_file_name} ({mat_file_count} / {len(mat_file_list)}) [{len(dataset)}] ...")
        mat_file_count += 1

        full_path = None
        for i in range(len(FULL_DATA_FOLDERS)):
            # if i == 0 or i == 1:
            #     full_path = os.path.join(FULL_DATA_FOLDERS[i], mat_file_name) 
            # else:
            full_path = os.path.join("/data/datasets/SR_Dataset_v1/final_test_data/", mat_file_name) #Changed 0312 for L files 

            if os.path.exists(full_path):
                break
            else:
                full_path = None
        if full_path is None:
            print("Couldn't find in full data folders; continuing...")
            continue

        mat_data = mat73.loadmat(full_path)

        valid_layers = 0
        has_nan = False
        for y in range(LAYER_PREDICT_COUNT + LAYER_FEATURE_COUNT + 1):
            nan_count = 0
            for x in range(NODE_COUNT):
                if np.isnan(mat_data["layers_vector"][y, x]):
                    nan_count += 1
            if nan_count < NODE_COUNT:
                valid_layers += 1
            if nan_count > 0:
                has_nan = True
        
        if valid_layers < LAYER_PREDICT_COUNT + LAYER_FEATURE_COUNT:
            data_less_layer += 1
            continue # Not enough layers to predict
        elif has_nan:
            data_with_nan += 1
            continue # NaN in data
        else:
            total_valid_data += 1
            # continue
            # Full valid data

        if not use_cache:
            cache_list.append(mat_file_name)

        timesteps = []
        for layer in range(LAYER_FEATURE_COUNT):

            G = nx.Graph()
            layer_heights = np.zeros(LAYER_FEATURE_COUNT+LAYER_PREDICT_COUNT)
            pred_voffsets = np.zeros(NODE_COUNT)
            for x in range(NODE_COUNT):

                height_sum = 0
                for i in range(LAYER_FEATURE_COUNT+LAYER_PREDICT_COUNT):
                    layer_heights[i] = mat_data["layers_vector"][i + 1, x] - height_sum
                    height_sum += layer_heights[i]

                # Y = np.copy(layer_heights[LAYER_FEATURE_COUNT:]) if PREDICT_HISTORIC else np.copy(layer_heights[:LAYER_PREDICT_COUNT])
                # pred_voffsets[x] = mat_data["layers_vector"][LAYER_FEATURE_COUNT, x] if PREDICT_HISTORIC else mat_data["layers_vector"][0, x]

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

                F = [lat,
                     long,
                     height,
                     smb,
                     temp,
                     refreezing,
                     height_change,
                     snow_pack,
                     density,
                     elevation]
                
                if REMOVE_ALL_PHYSICAL_PARAMS:
                    F = F[:3]

                if layer == 0:
                    G.add_node(x, x=F)# , y=Y)
                else:
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

            # Prevent 0- and 1-weight neighbors
            min_weight *= 0.95
            max_weight *= 1.05

            for u, v, d in G.edges(data=True):
                d["edge_weight"] = (d["edge_weight"] - min_weight) / (max_weight - min_weight)

            data = from_networkx(G)

            if layer == 0:
                data.mat_file = full_path
                # data.pred_voffsets = pred_voffsets
            else:
                del data.edge_index
                del data.edge_weight

            data = data#.to(DEVICE)

            timesteps.append(data)

        dataset.append(timesteps)

    save_dill(dataset, saved_dataset_path)
    if not use_cache:
        save_dill(cache_list, saved_cache_path)

    save_dill(valid_data_names, saved_name_path)

    print("Full data folder:", FULL_DATA_FOLDERS) 
    print("Valide data", total_valid_data)
    print("Data with NaN", data_with_nan)
    print("Data less layer", data_less_layer)

    return dataset


DATASET = load_dataset()


def normalize_dataset_zscore(dataset):

    all_features = None

    for graph_collection in tqdm(dataset):
        for graph in graph_collection:
            if all_features is None:
                all_features = graph.x
            else:
                all_features = torch.cat((all_features, graph.x), 0)

    features_mean = torch.mean(all_features, dim=0)
    features_std = torch.std(all_features, dim=0)

    print("Normalized means and stds:")
    print(features_mean)
    print(features_std)

    for graph_collection in dataset:
        for graph in graph_collection:
            graph.x -= features_mean
            graph.x /= features_std

normalize_dataset_zscore(DATASET)

if not REMOVE_ALL_PHYSICAL_PARAMS:
    for gc in DATASET:
        for i in range(len(gc)):
            for j in range(len(FEATURE_ABLATION)):
                if FEATURE_ABLATION[j] == "0":
                    gc[i].x[:, 3+j] = 0



