### Load in data
import sys
import os
import numpy as np
import torch
import pickle

sys.path.append('../../')
sys.path.append('../../dMG')  # Add the dMG root directory.
sys.path.append(os.path.abspath('..'))  # Add the parent directory of `scripts` to the path.

from scripts import load_config
from dMG import load_nn_model
from src.dMG.models.phy_models.terzaghi import TerzaghiMultiLayer as dPLT


#------------------------------------------#
# Define model settings here.
CONFIG_PATH = '/projects/mhpi/leoglonz/dPLT/src/dMG/conf/config_ls.yaml'
TEST_SPLIT = 0.2
#------------------------------------------#



config = load_config(CONFIG_PATH)


# Load data
with open(config['observations']['train_path'], 'rb') as f:
    data_dict = pickle.load(f)


# only keep desired layers
layer_count = config['dpl_model']['phy_model']['layer_count']
data_dict['attributes'] = data_dict['attributes'][:, :layer_count, :]


# Normalize attributes for NN:
attrs = data_dict['attributes']
for i in range(attrs.shape[-1]):
    attrs[:, :, i] = (attrs[:, :, i] - attrs[:, :, i].mean()) \
        / attrs[:, :, i].std()
    
data_dict['xc_nn_norm'] = attrs
data_dict['x_phy'] = data_dict['forcing']


# Train-test split + convert to torch tensors
train_dataset = {}
test_dataset = {}
split = int(len(data_dict['x_phy']) * (1 - TEST_SPLIT))

for key in data_dict.keys():
    train_dataset[key] = torch.tensor(
        data_dict[key][:split,],
        dtype=config['dtype'],
        device=config['device'],
    )
    test_dataset[key] = torch.tensor(
        data_dict[key][split:,],
        dtype=config['dtype'],
        device=config['device'],
    )

# Reshape to 3d
shape = train_dataset['xc_nn_norm'].shape
train_dataset['xc_nn_norm'] = train_dataset['xc_nn_norm'].reshape(
    shape[0], 
    shape[1],
    shape[2] * shape[3],
)

shape = test_dataset['xc_nn_norm'].shape
test_dataset['xc_nn_norm'] = test_dataset['xc_nn_norm'].reshape(
    shape[0], 
    shape[1],
    shape[2] * shape[3],
)


# Model initialization
model = dPLT(config['dpl_model']['phy_model'], device=config['device'])
nn = load_nn_model(
    model,
    config['dpl_model'],
    device=config['device'],
)


# Model forward
parameters = nn(train_dataset['xc_nn_norm'])

predictions = model(
    train_dataset,
    parameters,
)