import os
import torch
os.environ['TORCH'] = torch.__version__
print(torch.__version__)

import json, time

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import json, os, sys
import sklearn
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from torch_geometric.data import InMemoryDataset, Data 
from torch_geometric.loader import DataLoader
import wandb
import pprint

kB = 0.0019872 # Bolzman constant
C2T = 273.15 # conversion from celsius to kalvin

LOG_PATH = os.path.join(os.path.dirname(__file__), "debug-100c7d.log")


def _log(hypothesis_id: str, message: str, data: dict):
    payload = {
        "sessionId": "100c7d",
        "runId": "pre-fix",
        "hypothesisId": hypothesis_id,
        "location": "gnn_run.py",
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    print("CUDA not available; using CPU")

sys.path.append('..')
from nnn.gnn import *

# When running saved model, only `saved_model_path` is actually used
# Everything else is just for logging purpose
linear_hidden_channels=[128]
config = dict(
    mode='val',
    n_epoch=200,
    params=['dH', 'Tm'], # not used by the program, for logging only
    norm_method='normalize', # not used by the program, for logging only
    hidden_channels=125,
    pooling='Set2Set',
    processing_steps=10,
    n_graphconv_layer=4,
    n_linear_layer=len(linear_hidden_channels),
    linear_hidden_channels=linear_hidden_channels,
    graphconv_dropout=0.012732466797412492,  
    linear_dropout=.25,#0.22559831635994448,
    batch_size=1842,
    learning_rate=0.0023788383566734047,
    dataset="NNN_v2", # NNN_v1 or NNN_v2 (+duplex) or NNN_curve_v1 (17 dim prediction)
    use_train_set_ratio=1,
    architecture="GraphTransformer",
    concat=False,
    saved_model_path='./models/gnn_state_dict_bumbling-serenity-13.pt',
    )

# 3: Load saved model and run validation

# #region agent log
_log(
    "H_checkpoint",
    "Saved model path check",
    {
        "saved_model_path": config.get("saved_model_path"),
        "exists": os.path.exists(config.get("saved_model_path", "")),
        "cwd": os.getcwd(),
        "models_dir_exists": os.path.isdir("./models"),
        "models_dir_listing": (sorted(os.listdir("./models")) if os.path.isdir("./models") else None),
    },
)
# #endregion agent log

if not os.path.exists(config["saved_model_path"]):
    raise FileNotFoundError(
        f"Saved model not found at {config['saved_model_path']}. "
        f"Set config['saved_model_path'] to an existing .pt file under ./models/, "
        f"or train a model to create one."
    )

trained_model = run_saved_model(config, 
    test_result_fn='test_result_aggr_out.npz',
    log_wandb=False)

model = model_pipeline(config, save_model=True)

# Manual model saving with custom path and name
# model_path = r'E:\project ms\nnn_paper\anant_experiments\my_exp1.pt'
# torch.save(model.state_dict(), config['saved_model_path'])
# print(f"Model saved successfully at: {config['saved_model_path']}")

# trained_model = run_saved_model(config, 
#     test_result_fn='test_result_aggr_out.npz',
#     log_wandb=False)

## SAVING MODEL ##
# model_path = f'/mnt/d/data/nnn/models/gnn_state_dict_{wandb.run.name}.pt'
# torch.save(trained_model.state_dict(), model_path)