import datetime as dt
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

log = logging.getLogger(__name__)


def intersect(tLst1: List[np.float32], tLst2: List[np.float32]) -> List[int]:
    C, ind1, ind2 = np.intersect1d(tLst1, tLst2, return_indices=True)
    return ind2


def time_to_date(t: int, hr: bool = False) -> Union[dt.date, dt.datetime]:
    """Convert time to date or datetime object.
    
    Adapted from Farshid Rahmani.
    
    Parameters
    ----------
    t : int, datetime, date
        Time object to convert.
    hr : bool
        If True, return datetime object.
    """
    tOut = None

    if type(t) is str:
        t = int(t.replace('/', ''))

    if type(t) is int:
        if t < 30000000 and t > 10000000:
            t = dt.datetime.strptime(str(t), "%Y%m%d").date()
            tOut = t if hr is False else t.datetime()

    if type(t) is dt.date:
        tOut = t if hr is False else t.datetime()

    if type(t) is dt.datetime:
        tOut = t.date() if hr is False else t

    if tOut is None:
        raise Exception("Failed to change time to date.")
    return tOut


def trange_to_array(tRange, *, step=np.timedelta64(1, "D")):
    sd = time_to_date(tRange[0])
    ed = time_to_date(tRange[1])
    tArray = np.arange(sd, ed, step)
    return tArray


def random_index(
    ngrid: int,
    nt: int,
    dim_subset: Tuple[int, int],
    warm_up: int = 0
) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    batch_size, rho = dim_subset
    i_grid = np.random.randint(0, ngrid, size=batch_size)
    i_t = np.random.randint(0 + warm_up, nt - rho, size=batch_size)
    return i_grid, i_t


def create_training_grid(
    x: NDArray[np.float32],
    config: Dict[str, Any],
    n_samples: int = None,
) -> Tuple[int, int, int]:
    """Define a training grid of samples x iterations per epoch x time.

    Parameters
    ----------
    x : NDArray[np.float32]
        The input data for a model.
    config : dict
        The configuration dictionary.
    n_samples : int, optional
        The number of samples to use in the training grid.
    
    Returns
    -------
    Tuple[int, int, int]
        The number of samples, the number of iterations per epoch, and the
        number of timesteps.
    """
    t_range = config['train_time']
    n_t = x.shape[0]

    if n_samples is None:
        n_samples = x.shape[1]

    t = trange_to_array(t_range)
    rho = min(t.shape[0], config['dpl_model']['rho'])

    # Calculate number of iterations per epoch.
    n_iter_ep = int(
        np.ceil(
            np.log(0.01)
            / np.log(1 - config['train']['batch_size'] * rho / n_samples
                     / (n_t - config['dpl_model']['phy_model']['warm_up']))
        )
    )
    return n_samples, n_iter_ep, n_t,


def select_subset(
    config: Dict,
    x: NDArray[np.float32],
    i_grid: NDArray[np.float32],
    i_t: NDArray[np.float32],
    rho: int,
    *,
    c: Optional[NDArray[np.float32]] = None,
    tuple_out: bool = False,
    has_grad: bool = False,
    warm_up: int = 0,
) -> torch.Tensor:
    """
    Select a subset of input array.
    """
    nx = x.shape[-1]
    nt = x.shape[0]
    if x.shape[0] == len(i_grid):   #hack
        i_grid = np.arange(0,len(i_grid))  # hack
    if nt <= rho:
        i_t.fill(0)

    batch_size = i_grid.shape[0]

    if i_t is not None:
        x_tensor = torch.zeros([rho + warm_up, batch_size, nx], requires_grad=has_grad)
        for k in range(batch_size):
            temp = x[np.arange(i_t[k] - warm_up, i_t[k] + rho), i_grid[k]:i_grid[k] + 1, :]
            x_tensor[:, k:k + 1, :] = torch.from_numpy(temp)
    else:
        if len(x.shape) == 2:
            # Used for local calibration kernel (x = Ngrid * Ntime).
            x_tensor = torch.from_numpy(x[i_grid, :]).float()
        else:
            # Used for rho equal to the whole length of time series.
            x_tensor = torch.from_numpy(x[:, i_grid, :]).float()
            rho = x_tensor.shape[0]

    if c is not None:
        nc = c.shape[-1]
        temp = np.repeat(np.reshape(c[i_grid, :], [batch_size, 1, nc]), rho + warm_up, axis=1)
        c_tensor = torch.from_numpy(temp).float()

        if tuple_out:
            if torch.cuda.is_available():
                x_tensor = x_tensor.cuda()
                c_tensor = c_tensor.cuda()
            return x_tensor, c_tensor
        return torch.cat((x_tensor, c_tensor), dim=2)

    return x_tensor.to(config['device']) if torch.cuda.is_available() else x_tensor


def numpy_to_torch_dict(
    data_dict: Dict[str, NDArray[np.float32]],
    device: str,
) -> Dict[str, torch.Tensor]:
    """Convert numpy data dictionary to torch tensor dictionary.

    Parameters
    ----------
    data_dict : Dict[str, NDArray[np.float32]]
        The numpy data dictionary.
    device : str
        The device to move the data to.
    """
    for key, value in data_dict.items():
        if type(value) is torch.Tensor:
            data_dict[key] = torch.tensor(
                value.copy(),
                dtype=torch.float32,
                device=device
            )
    return data_dict


def load_json(file_path: str) -> Dict:
    """Load JSON data from a file and return it as a dictionary.
    
    Parameters
    ----------
    file_path : str
        Path to the JSON file to load.
    
    Returns
    -------
    dict
        Dictionary containing the JSON data.
    """   
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)

            if isinstance(data, str):
                # If json is still a string, decode again.
                return json.loads(data) 
            return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from file '{file_path}'.")
        return None


def txt_to_array(txt_path: str):
    """Load txt file of gage ids to numpy array."""
    with open(txt_path, 'r') as f:
        lines = f.read().strip()  # Remove extra whitespace
        lines = lines.replace("[", "").replace("]", "")  # Remove brackets
    return np.array([int(x) for x in lines.split(",")])


def timestep_resample(
    data: Union[pd.DataFrame, NDArray[np.float32], torch.Tensor],
    resolution: str = 'M',
    method: str = 'mean',
) -> pd.DataFrame:
    """
    Resample the data to a given resolution.

    Parameters
    ----------
    data : Union[pd.DataFrame, NDArray[np.float32]]
        The data to resample.
    resolution : str
        The resolution to resample the data to. Default is 'M' (monthly).
    method : str
        The resampling method. Default is 'mean'.

    Returns
    -------
    pd.DataFrame
        The resampled data.
    """
    if isinstance(data, pd.DataFrame):
        pass
    elif isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()
        data = pd.DataFrame(data)
        data['time'] = pd.to_datetime(data['time'])
    elif isinstance(data, NDArray[np.float32]):
        data = pd.DataFrame(data)
        data['time'] = pd.to_datetime(data['time'])
    else:
        raise ValueError(f"Data type not supported: {type(data)}")
    
    data.set_index('time', inplace=True)
    data_resample = data.resample(resolution).agg(method)

    data_resample['time'] = data_resample.index

    return data_resample
