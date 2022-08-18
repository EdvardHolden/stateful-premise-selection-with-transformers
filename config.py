import time
import os
from pathlib import Path

timestr = time.strftime("%Y_%m%d-%H%M%S")

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

# model_path = "./saved_models/transformer_500"
model_path = "./saved_models/transformer_10"

# Path to result of running predict
predictions_path = os.path.join("./predictions", Path(model_path).stem)

# train_source_path = "./data/train/prefix.src"
# train_target_path = "./data/train/prefix.tgt"
train_source_path = "./data/debug/prefix.src"
train_target_path = "./data/debug/prefix.tgt"


valid_source_path = "./data/debug/prefix.src"
valid_target_path = "./data/debug/prefix.tgt"

# Path used for predictions
evaluate_source_path = "./data/debug/prefix.src"

model_save_path = "./saved_models/"

PAD_INDEX = 0
START_END_INDEX = 1
