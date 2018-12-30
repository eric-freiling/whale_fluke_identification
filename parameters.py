from os.path import join
from pathlib import Path

freeze_flag = False
input_shape = (224, 224, 3)
bw_flag = False
filter_sizes = [64, 128, 128, 256]
conv_shapes = [(2, 2), (2, 2), (2, 2), (2, 2)]
conv_acts = ["relu", "relu", "relu", "relu"]

# Set final connected layers
dense_shapes = [4096, 4096]
dense_acts = ["sigmoid", "relu"]

# Set parameters
validation_flag = True
validation_percent = 0.20
batch_size = 20
epochs = 10000
learning_rate = 0.0001
print_iter = 10
save_iter = 10

save_dir = 'saved_models'
if bw_flag:
    model_name = str(input_shape[0]) + "_" + str(input_shape[1]) + '_bw'
else:
    model_name = str(input_shape[0]) + "_" + str(input_shape[1]) + '_color'
model_path = join(save_dir, model_name)

data_path = Path("../data/whale_fluke_data")
train_path = Path(data_path / "train")
test_path = Path(data_path / "test")
