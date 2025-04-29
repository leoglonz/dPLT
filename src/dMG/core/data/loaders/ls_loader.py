import json
import logging
import os
import pickle
from typing import Any, Optional


import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.exceptions import DataDimensionalityWarning

from dMG.core.data.data import intersect
from dMG.core.data.loaders.base import BaseLoader

log = logging.getLogger(__name__)


class LsLoader(BaseLoader):
    """Data loader for land subsidence inputs (gw, geology, subsidence).

    All data is loaded as Pytorch tensors.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    test_split : bool, optional
        Whether to split data into training and testing sets. Default is False.
    overwrite : bool, optional
        Whether to overwrite existing normalization statistics. Default is False.
    """
    def __init__(
        self,
        config: dict[str, Any],
        test_split: Optional[bool] = False,
        overwrite: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.test_split = test_split
        self.overwrite = overwrite
        self.supported_data = ['ls_aggregate']
        self.data_name = config['observations']['name']
        self.nn_attributes = config['dpl_model']['nn_model'].get('attributes', [])
        self.nn_forcings = config['dpl_model']['nn_model'].get('forcings', [])
        self.phy_attributes = config['dpl_model']['phy_model'].get('attributes', [])
        self.phy_forcings = config['dpl_model']['phy_model'].get('forcings', [])
        self.all_forcings = self.config['observations']['forcings_all']
        self.all_attributes = self.config['observations']['attributes_all']

        self.target = config['train']['target']
        self.log_norm_vars = config['dpl_model']['phy_model'].get('use_log_norm', [])
        self.device = config['device']
        self.dtype = config['dtype']
        
        self.train_dataset = None
        self.eval_dataset = None
        self.dataset = None
        self.norm_stats = None

        if self.data_name not in self.supported_data:
            raise ValueError(f"Data source '{self.data_name}' not supported.")

        self.load_dataset()

    def load_dataset(self) -> None:
        """Load dataset into dictionary of nn and physics model input arrays."""
        mode = self.config['mode']
        if mode == 'predict':
            self.dataset = self._preprocess_data(scope='predict')
        elif self.test_split:
            self.train_dataset = self._preprocess_data(scope='train')
            self.eval_dataset = self._preprocess_data(scope='test')
        elif mode in ['train', 'test']:
            self.train_dataset = self._preprocess_data(scope=mode)
        else:
            self.dataset = self._preprocess_data(scope='all')

    def _preprocess_data(
        self,
        scope: Optional[str],
    ) -> dict[str, NDArray[np.float32]]:
        """Read data from the dataset."""
        x_phy, c_phy, x_nn, c_nn, target = self.read_data(scope)

        # Normalize nn input data
        self.load_norm_stats()
        xc_nn_norm = self.normalize(x_nn, c_nn)

        # Build data dict of Torch tensors
        dataset = {
            'x_phy': self.to_tensor(x_phy),
            'c_phy': self.to_tensor(c_phy),
            'x_nn': self.to_tensor(x_nn),
            'c_nn': self.to_tensor(c_nn),
            'xc_nn_norm': self.to_tensor(xc_nn_norm),
            'target': self.to_tensor(target),
        }
        return dataset

    def read_data(self, scope: Optional[str]) -> tuple[NDArray[np.float32]]:
        """Read data from the data file."""
        try:
            if scope == 'train':
                data_path = self.config['observations']['train_path']
                time = self.config['train_time']
            elif scope == 'test':
                data_path = self.config['observations']['test_path']
                time = self.config['test_time']
            elif scope == 'predict':
                data_path = self.config['observations']['test_path']
                time = self.config['predict_time']
            elif scope == 'all':
                data_path = self.config['observations']['test_path']
                time = self.config['all_time']
            else:
                raise ValueError("Scope must be 'train', 'test', 'predict', or 'all'.")
        except KeyError as e:
            raise ValueError(f"Key {e} for data path not in dataset config.") from e
        
        # Get time indicies
        all_time = pd.date_range(
            self.config['all_time'][0],
            self.config['all_time'][-1],
            freq='MS',
        )
        idx_start = all_time.get_loc(time[0])
        idx_end = all_time.get_loc(time[-1]) + 1

        # Load data
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        forcings = data['forcing']  # [nt, nb, nforc]
        attributes = data['attributes']  # [nt, nb, nlayer, nattr]
        target = data['target']  # [nt, nb]
        del data

        if len(forcings.shape) < 3:
            forcings = forcings[:,:, np.newaxis]

        if len(target.shape) < 3:
            target = target[:, :, np.newaxis]

        # Forcing subset for phy model
        phy_forc_idx = []
        for forc in self.phy_forcings:
            if forc not in self.all_forcings:
                raise ValueError(f"Forcing {forc} not listed in available forcings.")
            phy_forc_idx.append(self.all_forcings.index(forc))
        
        # Attribute subset for phy model
        phy_attr_idx = []
        for attr in self.phy_attributes:
            if attr not in self.all_attributes:
                raise ValueError(f"Attribute {attr} not in the list of all attributes.")
            phy_attr_idx.append(self.all_attributes.index(attr))

        # Forcings subset for nn model
        nn_forc_idx = []
        for forc in self.nn_forcings:
            if forc not in self.all_forcings:
                raise ValueError(f"Forcing {forc} not in the list of all forcings.")
            nn_forc_idx.append(self.all_forcings.index(forc))

        # Attribute subset for nn model
        nn_attr_idx = []
        for attr in self.nn_attributes:
            if attr not in self.all_attributes:
                raise ValueError(f"Attribute {attr} not in the list of all attributes.")
            nn_attr_idx.append(self.all_attributes.index(attr))

        # For norm_stats file
        self.full_forc = forcings
        self.full_attr = attributes
        self.full_target = target

        x_phy = forcings[idx_start:idx_end, :, phy_forc_idx]
        c_phy = attributes[idx_start:idx_end, :, :, phy_attr_idx]
        x_nn = forcings[idx_start:idx_end, :, nn_forc_idx]
        c_nn = attributes[idx_start:idx_end, :, :, nn_attr_idx]
        target = target[idx_start:idx_end, :, :]

        # Collapse attribute and layer dimensions
        c_phy = c_phy.reshape(c_phy.shape[0], c_phy.shape[1], -1)
        c_nn = c_nn.reshape(c_nn.shape[0], c_nn.shape[1], -1)

        return x_phy, c_phy, x_nn, c_nn, target
        
    def load_norm_stats(self) -> None:
        """Load or calculate normalization statistics if necessary."""
        self.out_path = os.path.join(
            self.config['model_path'],
            'normalization_statistics.json',
        )

        if os.path.isfile(self.out_path) and not self.overwrite:
            if not self.norm_stats:
                with open(self.out_path) as f:
                    self.norm_stats = json.load(f)
        else:
            # Init normalization stats if file doesn't exist or overwrite is True.
            self.norm_stats = self._init_norm_stats(
                self.full_forc, self.full_attr, self.full_target
            )
        del self.full_forc, self.full_attr, self.full_target
    
    def _init_norm_stats(
        self,
        x_nn: NDArray[np.float32],
        c_nn: NDArray[np.float32],
        target: NDArray[np.float32],
    ) -> dict[str, list[float]]:
        """Compile calculations of data normalization statistics."""
        stat_dict = {}

        # Get basin areas from attributes.
        basin_area = self._get_basin_area(c_nn)

        # Forcing variable stats
        for k, var in enumerate(self.nn_forcings):
            if var in self.log_norm_vars:
                stat_dict[var] = self._calc_gamma_stats(x_nn[:, :, k])
            else:
                stat_dict[var] = self._calc_norm_stats(x_nn[:, :, k])

        # Attribute variable stats
        for k, var in enumerate(self.nn_attributes):
            stat_dict[var] = self._calc_norm_stats(c_nn[:, k])

        # Target variable stats
        for i, name in enumerate(self.target):
            stat_dict[name] = self._calc_norm_stats(
                np.swapaxes(target[:, :, i:i+1], 1, 0),
            )

        with open(self.out_path, 'w') as f:
            json.dump(stat_dict, f, indent=4)
        
        return stat_dict

    def _calc_norm_stats(
        self,
        x: NDArray[np.float32],
        basin_area: NDArray[np.float32] = None,
    ) -> list[float]:
        """
        Calculate statistics for normalization with optional basin
        area adjustment.
        """
        # Handle invalid values
        x[x == -999] = np.nan
        if basin_area is not None:
            x[x < 0] = 0  # Specific to basin normalization

        # Basin area normalization
        if basin_area is not None:
            nd = len(x.shape)
            if nd == 3 and x.shape[2] == 1:
                x = x[:, :, 0]  # Unsqueeze the original 3D matrix
            temparea = np.tile(basin_area, (1, x.shape[1]))
            flow = (x * 0.0283168 * 3600 * 24) / (temparea * (10 ** 6)) * 10 ** 3
            x = flow  # Replace x with flow for further calculations

        # Flatten and exclude NaNs and invalid values
        a = x.flatten()
        if basin_area is None:
            a = np.swapaxes(x, 1, 0).flatten() if len(x.shape) > 1 else x.flatten()
        b = a[(~np.isnan(a)) & (a != -999999)]
        if b.size == 0:
            b = np.array([0])

        # Calculate statistics
        transformed = np.log10(np.sqrt(b) + 0.1) if basin_area is not None else b
        p10, p90 = np.percentile(transformed, [10, 90]).astype(float)
        mean = np.mean(transformed).astype(float)
        std = np.std(transformed).astype(float)

        return [p10, p90, mean, max(std, 0.001)]
    
    def _calc_gamma_stats(self, x: NDArray[np.float32]) -> list[float]:
        """Calculate gamma statistics for streamflow and precipitation data."""
        a = np.swapaxes(x, 1, 0).flatten()
        b = a[(~np.isnan(a))]
        b = np.log10(
            np.sqrt(b) + 0.1
        )

        p10, p90 = np.percentile(b, [10,90]).astype(float)
        mean = np.mean(b).astype(float)
        std = np.std(b).astype(float)

        return [p10, p90, mean, max(std, 0.001)]
    
    def _get_basin_area(self, c_nn: NDArray[np.float32]) -> NDArray[np.float32]:
        """Get basin area from attributes."""
        try:
            area_name = self.config['observations']['area_name']
            basin_area = c_nn[:, self.nn_attributes.index(area_name)][:, np.newaxis]
        except KeyError:
            log.warning("No 'area_name' in observation config. Basin area norm will not be applied.")
            basin_area = None

        return basin_area

    def normalize(
        self,
        x_nn: NDArray[np.float32],
        c_nn: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Normalize data for neural network."""
        # TODO: Add np.swapaxes(x_nn, 1, 0) here and remove from _to_norm. This changes normalization, need to determine if it's detrimental.
        x_nn_norm = self._to_norm(
            np.swapaxes(x_nn, 1, 0).copy(),
            self.nn_forcings,
        )
        c_nn_norm = self._to_norm(
            c_nn,
            self.nn_attributes,
        )

        # Remove nans
        x_nn_norm[x_nn_norm != x_nn_norm] = 0
        c_nn_norm[c_nn_norm != c_nn_norm] = 0

        if self.nn_forcings == []:
            xc_nn_norm = c_nn_norm.copy()
        else:
            xc_nn_norm = np.concatenate((x_nn_norm, c_nn_norm), axis=2)
        del x_nn_norm, c_nn_norm, x_nn

        return xc_nn_norm

    def _to_norm(
        self,
        data: NDArray[np.float32],
        vars: list[str],
    ) -> NDArray[np.float32]:
        """Standard data normalization."""
        data_norm = np.zeros(data.shape)

        for k, var in enumerate(vars):
            stat = self.norm_stats[var]

            if len(data.shape) == 4:
                if var in self.log_norm_vars:
                    data[:, :, :, k] = np.log10(np.sqrt(data[:, :, :, k]) + 0.1)
                data_norm[:, :, :, k] = (data[:, :, :, k] - stat[2]) / stat[3]
            elif len(data.shape) == 3:
                if var in self.log_norm_vars:
                    data[:, :, k] = np.log10(np.sqrt(data[:, :, k]) + 0.1)
                data_norm[:, :, k] = (data[:, :, k] - stat[2]) / stat[3]
            else:
                raise DataDimensionalityWarning("Data dimension must be 3 or 4.")
        return data_norm

    def _from_norm(
        self,
        data_norm: NDArray[np.float32],
        vars: list[str],
    ) -> NDArray[np.float32]:
        """De-normalize data."""
        data = np.zeros(data_norm.shape)
                
        for k, var in enumerate(vars):
            stat = self.norm_stats[var]
            if len(data_norm.shape) == 3:
                data[:, :, k] = data_norm[:, :, k] * stat[3] + stat[2]
                if var in self.log_norm_vars:
                    data[:, :, k] = (np.power(10, data[:, :, k]) - 0.1) ** 2
            elif len(data_norm.shape) == 2:
                data[:, k] = data_norm[:, k] * stat[3] + stat[2]
                if var in self.log_norm_vars:
                    data[:, k] = (np.power(10, data[:, k]) - 0.1) ** 2
            else:
                raise DataDimensionalityWarning("Data dimension must be 2 or 3.")

        if len(data.shape) < 3:
            return data
        else:
            return np.swapaxes(data, 1, 0)
        
