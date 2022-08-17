device = "cpu"
# device = "gpu"

EVALUATE = False

EPOCHS = 11
# EPOCHS = 1001


BATCH_SIZE = 64

# train_source_path = "./data/train/prefix.src"
# train_target_path = "./data/train/prefix.tgt"
train_source_path = "./data/debug/prefix.src"
train_target_path = "./data/debug/prefix.tgt"


valid_source_path = "./data/debug/prefix.src"
valid_target_path = "./data/debug/prefix.tgt"

model_save_path = "./models/"

PAD_INDEX = 0
START_END_INDEX = 1
