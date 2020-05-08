import torch

DATA_DIR = "../../data"
MODEL_PATH = "saved_models/"
SAMPLE_RATE = 47998
MEL_BANKS = 128
MIC_USED = 5  # mic number used
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
BATCH_SIZE = 16
CLASS_TYPE = "coarse_class"
RANDOM_SEED = 42
ENABLE_CLASS_COUNTING = True

# model parameters
CNN_CHANNELS = 32
RNN_IN_SIZE = 32
RNN_HH_SIZE = 32
DROPOUT = 0.5
MODEL_NAME = CLASS_TYPE + "_epochs_" + str(EPOCHS) +"_fmaps_" + str(CNN_CHANNELS) + "_dropout_" + str(DROPOUT) + ".pth"

test_cnn_channels = CNN_CHANNELS
test_rnn_in_size = RNN_IN_SIZE
test_rnn_hh_size = RNN_HH_SIZE
test_dropout = DROPOUT
test_model_name = CLASS_TYPE + "_epochs_" + str(EPOCHS) +"_fmaps_" + str(test_cnn_channels) + "_dropout_" + str(test_dropout) + ".pth"

if CLASS_TYPE == "fine_class":
    REF_LABELS = ["front", "left", "negative", "right"]
else:
    REF_LABELS = ["positive", "negative"]

