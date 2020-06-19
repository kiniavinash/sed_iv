import torch
import os

from datetime import datetime

# hyperparameters
SAMPLE_LENGTH = 1.5  # stupid work around to avoid errors due to corrupted samples
DATA_DIR = "/media/micarray/Seagate_8TB/results/BCC_combined/labeled_data/engineOnData-" + str(SAMPLE_LENGTH) + "-0-0/data"
RUN_START = datetime.now().strftime("%b%d_%H-%M-%S")
MODEL_PATH = "saved_models/" + RUN_START + "/"

print("Starting run now: {}".format(RUN_START))

SAMPLE_RATE = 47998
MEL_BANKS = 128
MIC_USED = 5  # mic number used
N_FFT = 2048
OVERLAP = 0.275
WINDOW = 'hamming'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 150
BATCH_SIZE = 8
CLASS_TYPE = "coarse_class"
RANDOM_SEED = 42
ENABLE_CLASS_COUNTING = False
LR = 0.001
OPTIM_FUNC = "Adam"

# model parameters
CNN_CHANNELS = 32
RNN_IN_SIZE = 32
RNN_HH_SIZE = 32
DROPOUT = 0.5
MODEL_NAME = CLASS_TYPE + "_epochs_" + str(EPOCHS) + \
             "_fmaps_" + str(CNN_CHANNELS) + "_dropout_"\
             + str(DROPOUT) + ".pth"

test_cnn_channels = CNN_CHANNELS
test_rnn_in_size = RNN_IN_SIZE
test_rnn_hh_size = RNN_HH_SIZE
test_dropout = DROPOUT
test_model_name = CLASS_TYPE + "_epochs_" + str(EPOCHS) + \
                  "_fmaps_" + str(test_cnn_channels) + "_dropout_" + \
                  str(test_dropout) + ".pth"

if CLASS_TYPE == "fine_class":
    REF_LABELS = ["front", "left", "negative", "right"]
    OUT_CLASSES = 4
    CLASS_MODE = "multi_class"
elif CLASS_TYPE == "coarse_class":
    REF_LABELS = ["positive", "negative"]
    OUT_CLASSES = 1
    CLASS_MODE = "single_class"
else:
    raise ValueError("Error in class type selection!")

# create folder to store the models
os.mkdir(MODEL_PATH)