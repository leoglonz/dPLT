# Generic, Scalable Differentiable Modeling Framework on PyTorch

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![image](https://img.shields.io/pypi/l/ruff.svg)](https://github.com/astral-sh/ruff/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)]()
[![Actions status](https://github.com/astral-sh/ruff/workflows/CI/badge.svg)](https://github.com/astral-sh/ruff/actions)


A domain-agnostic, PyTorch-based framework for developing trainable differentiable models that merge neural networks with process-based equations.
Following as a generalization of *HydroDL*, `generic_deltaModel` (or *DeltaModel*) aims
to expand differentiable modeling and learning capabilities to a wide variety of domains where prior equations can bring in benefits.

Closely synergizes with deep learning tools and the scale advantage of PyTorch.

<br>

### Key Features
- **Hybrid Modeling**: Combines neural networks with physical process equations for enhanced interpretability and generalizability.

- **PyTorch Integration**: Easily scales with PyTorch, enabling efficient training and compatibility with modern deep learning tools.

- **Domain-agnostic Flexibility**: While originally designed for hydrology, DeltaModel extends to other fields where physics-guided learning can add value.

<br>

### Ecosystem Integration
For differentiable hydrology models used in MHPI research, DeltaModel seamlessly integrates with:

- **HydroDL2.0 ([`hydroDL2`](https://github.com/mhpi/hydroDL2))**: Home to MHPI's suite of physics-based hydrology models, and differentiable model augmentations (think variational data
      assimilation, model coupling, and additional physics-based hydrology tools).
- **HydroData ([`hydro_data_dev`](https://github.com/mhpi/hydro_data_dev))**: Data extraction, processing, and management tools optimized for hydrology applications. [*In development*]


Explore the project's [roadmap](https://github.com/orgs/mhpi/projects/4) for planned features and future improvements.

<br>

### The overall idea
We define a "differentiable model" class which describes how neural networks and the process-based model are coupled. This class holds NNs and process-based models as attributes and can be trained and forwarded just as any other PyTorch model (nn.Module). We define classes to handle datasets (dataset class), various train/test experiments (trainer), multimodel handling and multi-GPU training (model handler), data assimilation and streaming in a uniform and modular way. All training and simulations can be specified by a config file to be adapted to custom applications.

<br>
## Quick Start: Building a Differentiable HBV ($\delta$ HBV) Model

Here’s an example of how you can build a differentiable model, coupling a physics-based model with a neural network to intelligently learn model parameters. In this instance, we use an
LSTM with the [HBV](https://en.wikipedia.org/wiki/HBV_hydrology_model) hydrology model.
```python
# Load configuration dictionary of model parameters and options
config = load_config(CONFIG_PATH)

# Setup a dataset dictionary of NN and physics model inputs.
dataset = get_dataset_dict(config, train=True)

# Initialize physical model and neural network.
phy_model = load_model(config['phy_model']['model'])
nn = init_nn_model(phy_model, config['nn_model'])

# Create the differentiable model dHBV: 
# a torch.nn.Module that describes how nn is linked to the physical model.
dpl_model = dHBV(phy_model, nn)


# Now dpl_model can be run or trained as any torch.nn.Module model in a standard training loop.

# For example, to forward:
output = dpl_model.forward(dataset, config['dpl_model'])
```


### Contributing:
We welcome contributions! Please submit changes via a fork and pull request. For more details, refer to docs/CONTRIBUTING.md.


### Repository Structure:

    .
    ├── deltaMod/
    │   ├── __main__.py                 # Main entry point
    │   ├── conf/                       # Configuration files
    │   │   ├── config.py
    │   │   ├── config.yaml             # Main configuration file
    │   │   ├── hydra/                  
    │   │   └── observations/           # Observation data config
    │   ├── core/                       
    │   │   ├── calc/                   # Calculation utilities
    │   │   ├── data/                   # Data processing
    │   │   └── utils/                  # Helper functions
    │   ├── models/                     
    │   │   ├── differentiable_model.py # Differentiable model definition
    │   │   ├── model_handler.py        # High-level model manager
    │   │   ├── loss_functions/         # Custom loss functions
    │   │   └── neural_networks/        # Neural network architectures
    │   └── trainers/                   # Training routines
    ├── docs/                           
    ├── envs/                           # Environment configuration files
    └── example/                        # Example scripts and usage guides
