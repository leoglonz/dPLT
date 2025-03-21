import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import zarr
from numpy.typing import NDArray
from sklearn.exceptions import DataDimensionalityWarning

from dMG.core.data.loaders.base import BaseLoader

log = logging.getLogger(__name__)


class MsHydroLoader(BaseLoader):
    """Data loader for multiscale hydrological data loading.
    
    All data is read from Zarr store and loaded as PyTorch tensors.

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
        config: Dict[str, Any],
        test_split: Optional[bool] = False,
        overwrite: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.test_split = test_split
        self.overwrite = overwrite
        self.supported_data = ['merit_forward']
        self.data_name = config['observations']['name']
        self.nn_attributes = config['dpl_model']['nn_model'].get('attributes', [])
        self.nn_forcings = config['dpl_model']['nn_model'].get('forcings', [])
        self.phy_attributes = config['dpl_model']['phy_model'].get('attributes', [])
        self.phy_forcings = config['dpl_model']['phy_model'].get('forcings', [])
        self.all_forcings = self.config['observations']['forcings_all']
        self.all_attributes = self.config['observations']['attributes_all']

        self.target = config['train']['target']
        self.log_norm_vars = config['dpl_model']['phy_model']['use_log_norm']
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
    ) -> Dict[str, NDArray[np.float32]]:
        """Read data from the dataset."""
        ac_all, elev_all, subbasin_id_all, x_nn, x_phy, c_nn = self.read_data(scope)
    
        # Remove nan
        x_phy = np.swapaxes(x_phy, 1, 0)
        x_phy[x_phy != x_phy] = 0

        # Normalize nn input data
        self.load_norm_stats(x_nn, c_nn, scope)
        xc_nn_norm, c_nn_norm = self.normalize(x_nn, c_nn)

        # Build data dict of Torch tensors
        # dataset = {
        #     'ac_all': self.to_tensor(ac_all),
        #     'elev_all': self.to_tensor(elev_all),
        #     'subbasin_id_all': self.to_tensor(subbasin_id_all),
        #     'c_nn': self.to_tensor(c_nn),
        #     'xc_nn_norm': self.to_tensor(xc_nn_norm),
        #     'c_nn_norm': self.to_tensor(c_nn_norm),
        #     'x_phy': self.to_tensor(x_phy),
        # }
        dataset = {
            'ac_all': ac_all,
            'elev_all': elev_all,
            'subbasin_id_all': subbasin_id_all,
            'c_nn': c_nn,
            'xc_nn_norm': xc_nn_norm,
            'c_nn_norm': c_nn_norm,
            'x_phy': x_phy,
        }
        return dataset

    def read_data(self, scope: Optional[str]) -> Tuple[NDArray[np.float32]]:
        """Read data from the data file."""
        try:
            if scope == 'train':
                time = self.config['train_time']
            elif scope == 'test':
                time = self.config['test_time']
            elif scope == 'predict':
                time = self.config['predict_time']                
            elif scope == 'all':
                time = self.config['all_time']
            else:
                raise ValueError("Scope must be 'train', 'test', 'predict', or 'all'.")
        except KeyError as e:
            raise ValueError(f"Key {e} for data path not in dataset config.")
        
        # Get time indicies
        all_time = pd.date_range(
            self.config['all_time'][0],
            self.config['all_time'][-1], 
            freq='d',
        )
        idx_start = all_time.get_loc(time[0])
        idx_end = all_time.get_loc(time[-1]) + 1

        # Load data
        root_zone = zarr.open_group(
            self.config['observations']['subbasin_data_path'],
            mode = 'r',
        )
        subbasin_id_all = np.array(
            root_zone[self.config['observations']['subbasin_id_name']][:]
        ).astype(int)

        # Forcing subset for phy model
        for i, forc in enumerate(self.nn_forcings):
            if forc not in self.all_forcings:
                raise ValueError(f"Forcing {forc} not listed in available forcings.")
            if i == 0:
                forc_array = np.expand_dims(root_zone[forc][:, idx_start:idx_end], -1) 
            else:
                forc_array = np.concatenate((
                    forc_array,
                    np.expand_dims(root_zone[forc][:, idx_start:idx_end], -1),
                ), axis = -1)
        forc_array = self._fill_nan(forc_array)

        # Attribute subset for nn model
        for i, attr in enumerate(self.nn_attributes):
            if attr not in self.all_attributes:
                raise ValueError(f"Attribute {attr} not in the list of all attributes.")
            if i == 0:
                attr_array = np.expand_dims(root_zone['attrs'][attr][:],-1)
            else:
                attr_array = np.concatenate((
                    attr_array,
                    np.expand_dims(root_zone['attrs'][attr][:], -1)
                ), axis = -1)
        
        # Get upstream area and elevation
        try:
            ac_name = self.config['observations']['upstream_area_name']
            ac_array = root_zone['attrs'][ac_name][:]
        except Exception:
            raise ValueError("Upstream area is not provided. This is needed for high-resolution streamflow model.")
        try:
            elevation_name = self.config['observations']['elevation_name']
            elev_array = root_zone['attrs'][elevation_name][:] 
        except Exception:
            raise ValueError("Elevation is not provided. This is needed for high-resolution streamflow model.")

        return [
            ac_array,
            elev_array,
            subbasin_id_all,
            forc_array,
            forc_array.copy(),
            attr_array,
        ]
        
    def load_norm_stats(
        self,
        x_nn: NDArray[np.float32],
        c_nn: NDArray[np.float32],
        scope: str,
    ) -> None:
        """Load or calculate normalization statistics if necessary.
        
        A different normalization file is loaded for training and testing data.
        """
        if scope == 'train':
            self.out_path = os.path.join(
                self.config['model_path'],
                'normalization_statistics.json',
            )
        elif scope in ['test', 'predict']:
            self.out_path = os.path.join(
                self.config['out_path'],
                'normalization_statistics.json',
            )
        else:
            raise ValueError(f"Unsupported scope {scope}.")

        if os.path.isfile(self.out_path) and not self.overwrite:
            if not self.norm_stats:
                with open(self.out_path, 'r') as f:
                    self.norm_stats = json.load(f)
        else:
            # Init normalization stats if file doesn't exist or overwrite is True.
            # NOTE: will be supported with release of multiscale training code.
            raise ValueError("Normalization statistics not found. Confirm 'normalization_statistics.json' is in your model directory.")

    def normalize(
        self,
        x_nn: NDArray[np.float32],
        c_nn: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Normalize data for neural network."""
        # TODO: Add np.swapaxes(x_nn, 1, 0) here and remove from _to_norm. This changes normalization, need to determine if it's detrimental.
        x_nn_norm = self._to_norm(x_nn, self.nn_forcings)
        c_nn_norm = self._to_norm(c_nn, self.nn_attributes)

        # Remove nans
        x_nn_norm[x_nn_norm != x_nn_norm] = 0    
        c_nn_norm[c_nn_norm != c_nn_norm] = 0

        c_nn_norm_repeat = np.repeat(
            np.expand_dims(c_nn_norm, 0),
            x_nn_norm.shape[0],
            axis=0,
        )

        xc_nn_norm = np.concatenate((x_nn_norm, c_nn_norm_repeat), axis=2)
        del x_nn_norm, x_nn

        return xc_nn_norm, c_nn_norm

    def _to_norm(
        self,
        data: NDArray[np.float32],
        vars: List[str],
    ) -> NDArray[np.float32]:
        """Standard data normalization."""
        data_norm = np.zeros(data.shape)

        for k, var in enumerate(vars):
            stat = self.norm_stats[var]

            if len(data.shape) == 3:
                if var in self.log_norm_vars:
                    data[:, :, k] = np.log10(np.sqrt(data[:, :, k]) + 0.1)
                data_norm[:, :, k] = (data[:, :, k] - stat[2]) / stat[3]
            elif len(data.shape) == 2:
                if var in self.log_norm_vars:
                    data[:, k] = np.log10(np.sqrt(data[:, k]) + 0.1)
                data_norm[:, k] = (data[:, k] - stat[2]) / stat[3]
            else:
                raise DataDimensionalityWarning("Data dimension must be 2 or 3.")
            
        # Should be external, except altering order of first two dims augments normalization.
        if len(data_norm.shape) < 3:
            return data_norm
        else:
            return np.swapaxes(data_norm, 1, 0)

    def _from_norm(
        self,
        data_norm: NDArray[np.float32],
        vars: List[str],
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

    def _fill_nan(self, array_3d):
        # Define the x-axis for interpolation
        x = np.arange(array_3d.shape[1])

        # Iterate over the first and third dimensions to interpolate the second dimension
        for i in range(array_3d.shape[0]):
            for j in range(array_3d.shape[2]):
                # Select the 1D slice for interpolation
                slice_1d = array_3d[i, :, j]

                # Find indices of NaNs and non-NaNs
                nans = np.isnan(slice_1d)
                non_nans = ~nans

                # Only interpolate if there are NaNs and at least two non-NaN values for reference
                if np.any(nans) and np.sum(non_nans) > 1:
                    # Perform linear interpolation using numpy.interp
                    array_3d[i, :, j] = np.interp(x, x[non_nans], slice_1d[non_nans], left=None, right=None)
        return array_3d
