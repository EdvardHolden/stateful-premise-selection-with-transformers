import time
import os
from pathlib import Path

timestr = time.strftime("%Y_%m%d-%H%M%S")
ORIGINAL_DATASET = os.path.join(Path.home(), "gnn-entailment-caption/nndata/")

device = "cpu"
# device = "gpu"

EVALUATE = False

EPOCHS = 11
# EPOCHS = 1001
# EPOCHS = 500  # They load predictions from here


# The model paramters
model_params = {
    "model_num_layers": 3,
    "model_state_size": 512,
    "model_num_heads": 8,
    "model_hidden_size": 2048,
}

BATCH_SIZE = 64

axiom_map_path = "name_axiom_map.pkl"

# model_path = "./saved_models/transformer_500"
model_path = "./saved_models/transformer_10"

# Path to result of running predict
predictions_path = os.path.join("./predictions", Path(model_path).stem)


DATA_PATH = "/home/eholden/stateful-premise-selection-with-RNNs/"

# train_source_path = DATA_PATH + "data/train/standard.src"
# train_target_path = DATA_PATH + "data/train/standard.tgt"
train_source_path = DATA_PATH + "data/debug/standard.src"
train_target_path = DATA_PATH + "data/debug/standard.tgt"


valid_source_path = DATA_PATH + "data/debug/standard.src"
valid_target_path = DATA_PATH + "data/debug/standard.tgt"

# Path used for predictions
evaluate_source_path = DATA_PATH + "data/debug/standard.src"

model_save_path = "./saved_models/"

PAD_INDEX = 0
START_END_INDEX = 1
