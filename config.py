import time
import os
from pathlib import Path

timestr = time.strftime("%Y_%m%d-%H%M%S")
ORIGINAL_DATASET = os.path.join(Path.home(), "gnn-entailment-caption/nndata/")

device = "cpu"
# device = "gpu"

EVALUATE = False

EPOCHS = 501
#EPOCHS = 3
# EPOCHS = 1001
#EPOCHS = 501  # They load predictions from here


# The model paramters
model_params = {
    "model_num_layers": 3,
    #"model_state_size": 512,
    "model_state_size": 256,
    "model_num_heads": 8,
    #"model_hidden_size": 2048,
    "model_hidden_size": 512,
    #"conjecture_max_length": 1000,
    "conjecture_max_length": 500,
}

BATCH_SIZE = 128

axiom_map_path = "name_axiom_map.pkl"


model_save_path = "./saved_models_merged/"

#model_path = os.path.join(model_save_path, "transformer_500")
model_path = os.path.join(model_save_path, "transformer_15")

# Path to result of running predict
predictions_path = os.path.join("./predictions", Path(model_path).stem)


#DATA_PATH = "/home/eholden/stateful-premise-selection-with-RNNs/"
DATA_PATH = "/shareddata/home/holden/stateful-premise-selection-with-RNNs/"

train_source_path = DATA_PATH + "data/train/standard.src"
train_target_path = DATA_PATH + "data/train/standard.tgt"
#train_source_path = DATA_PATH + "data/debug/standard.src"
#train_target_path = DATA_PATH + "data/debug/standard.tgt"


valid_source_path = DATA_PATH + "data/debug/standard.src"
valid_target_path = DATA_PATH + "data/debug/standard.tgt"

# Path used for predictions
#evaluate_source_path = DATA_PATH + "data/debug/standard.src"
evaluate_source_path = DATA_PATH + "data/all/standard.src"




PAD_INDEX = 0
START_END_INDEX = 1
