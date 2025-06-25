
from diffcrysgen.trainer import load_data, train_diffcrysgen

# provide the proper paths for training and test IRCR  
train_loader, test_loader, _, _ = load_data(
    "ircr-train.npy",
    "ircr-test.npy"
)

# train the model
train_diffcrysgen(train_loader, test_loader)


