from typing import Literal, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from core.data import timestep_resample
from core.utils import format_resample_interval


def plot_hydrograph(
    timesteps: pd.DatetimeIndex,
    predictions: Union[np.ndarray, torch.Tensor],
    obs: Union[np.ndarray, torch.Tensor] = None,
    resample: Literal['D','W', 'M', 'Y'] = 'D',
    title = None,
    ylabel: str = 'Streamflow (ft$^3$/s)',
    minor_ticks: bool = False,
    figsize: tuple = (12, 6),
    fontsize: int = 12,
    dpi: int = 100,
) -> None:
    """Plot the hydrograph of model predictions and observations (if specified).

    Parameters
    ----------
    timesteps : pd.DatetimeIndex
        The timesteps of the predictions.
    predictions : Union[np.ndarray, torch.Tensor]
        The model predictions.
    obs : Union[np.ndarray, torch.Tensor], optional
        The observed streamflow values. Default is None.
    resample : Literal['D','W', 'M', 'Y'], optional
        The resampling interval for the data. Default is 'D'.
    title : str, optional
        The title of the plot. Default is None.
    ylabel : str, optional
        The y-axis label. Default is 'Streamflow (ft$^3$/s)'.
    minor_ticks : bool, optional
        Whether to show minor ticks on the plot. Default is False.
    figsize : tuple, optional
        The figure size. Default is (12, 6).
    fontsize : int, optional
        The font size of the plot. Default is 12.
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(obs, torch.Tensor):
        obs = obs.detach().cpu().numpy()
    else:
        obs = np.zeros_like(predictions)

    # Resample the data to the specified temporal resolution.
    data = pd.DataFrame({
        'time': timesteps,
        'pred': predictions,
        'obs': obs,
    })
    data = timestep_resample(data, resolution=resample, method='mean')

    plt.rcParams.update({'font.size': fontsize})
    
    # Create the figure.
    plt.figure(figsize=figsize)
    plt.plot(
        data['time'],
        list(data['pred']),
        label='Prediction',
        marker='o',
        color='r',
    )

    if obs.mean() != 0:
        plt.plot(
            data['time'],
            list(data['obs']),
            label='Observation',
            marker='o',
            color='b',
        )
    
    plt.title(title)
    # plt.xlabel('Time')
    plt.xlabel(f"Time ({format_resample_interval(resample)})")
    plt.ylabel(ylabel)

    # plt.annotate(
    #     f"Prediction Interval: {format_resample_interval(resample)}",
    #     xy=(0.03, 0.9),
    #     xycoords='axes fraction',
    #     color='black',
    #     bbox=dict(
    #         boxstyle='round,pad=0.3',
    #         edgecolor='gray',
    #         facecolor='lightgray',
    #         alpha=0.5
    #     ),
    # )

    if obs.mean() != 0:
        plt.legend(
            loc='upper right',
            frameon=True,
        )

    plt.xticks(rotation=45)    

    ax = plt.gca()  # Get the current axis

    if minor_ticks:
        ax.minorticks_on()
        
    # Align minor ticks with major ticks
    # ax.xaxis.set_minor_locator(AutoMinorLocator(2))  # One minor tick between major ticks

    # Optionally adjust major tick locator based on resampling interval
    # from matplotlib.ticker import AutoMinorLocator, MultipleLocator
    # from matplotlib.dates import DayLocator, WeekdayLocator, MonthLocator, YearLocator

    # len_data = len(data)
    # if 'D' in resample:
    #     ax.xaxis.set_major_locator(DayLocator(interval=len_data//5))
    # elif 'W' in resample:
    #     ax.xaxis.set_major_locator(WeekdayLocator(interval=len_data//10))
    # elif 'M' in resample:
    #     ax.xaxis.set_major_locator(MonthLocator(interval=1))
    # elif 'Y' in resample:
    #     ax.xaxis.set_major_locator(YearLocator(interval=1))

    # Add grid lines
    ax.grid(which='major', linestyle='--', linewidth=0.7, alpha=0.8)
    ax.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.6)

    plt.show()