from typing import Any, Optional, Union

import torch
import numpy as np


class TerzaghiMultiLayer(torch.nn.Module):
    """
    Differentiable form of Terzaghi's 1D Consolidation Model (TCM) for land
    subsidence prediction.

    Authors
    -------
    -   Leo Lonzarich, Nicholas Kraabel
    -   Karl von Terzaghi (1923)

    Parameters
    ----------
    config : dict, optional
        Configuration dictionary.
    device : torch.device, optional
        Device to run the model on.
    """
    def __init__(
            self,
            config: Optional[dict[str, Any]] = None,
            device: Optional[torch.device] = None,
        ) -> None:
        super().__init__()
        self.name = 'Terzaghi Multi-layer'
        self.config = config
        self.initialize = False
        self.dynamic_params = []
        self.dy_drop = 0.0
        self.lookback = 2
        self.layer_count = 1
        self.learnable_layers = []
        self.learnable_params = []
        self.variables = ['gw_level']
        self.nearzero = 1e-5
        self.nmul = 1
        self.device = device
        self.parameter_bounds = {
            'parLT': [0, 50],  # Layer thickness
            'parVR': [0, 1],  # Void Ratio
            'parVCI': [0, 5],  # Virgin Compression Index
            'parRC': [0, 1],  # Recompression Index
            'parOCR': [0, 1],  # Overconsolidation Ratio
            'parK': [0, 100],  # Percent coarseness
            'parMv': [0, 10],  # Coefficient of volume compressibility
            'parIES': [0, 100],  # Initial effective stress
        }
        self.water_unit_weight = 9.81  # kN/m³
        self.min_stress = 2.0  # kPa

        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if config is not None:
            # Overwrite defaults with config values.
            self.dy_drop = config.get('dy_drop', self.dy_drop)
            self.lookback = config.get('lookback', self.lookback)
            self.layer_count = config.get('layer_count', self.layer_count)
            self.learnable_layers = config.get('learnable_layers', self.learnable_layers)
            self.learnable_params = config.get('learnable_params', self.learnable_params)
            self.variables = config.get('variables', self.variables)
            self.nearzero = config.get('nearzero', self.nearzero)
            self.nmul = config.get('nmul', self.nmul)
        self.set_parameters()

    def set_parameters(self) -> None:
        """Get physical parameters.
        
        # of parameters = # of layers * # of parameters to learn per
        layer * # of layers
        """
        self.param_names = list(self.parameter_bounds.keys())
        self.learnable_param_count = len(self.learnable_params) \
            * len(self.learnable_layers) * self.nmul

    def unpack_parameters(
            self,
            parameters: torch.Tensor,
            raw_parameters: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
        """Extract learned physical parameters from NN output and overwrite
        estimated parameters (raw).
        
        Parameters
        ----------
        parameters : torch.Tensor
            Unprocessed, learned parameters from a neural network.
        raw_parameters : torch.Tensor
            Raw parameters with shape [time, sites, layers, parameters, nmul].

        Returns
        -------
        dict
            Dictionary of processed parameters.
        """
        # Shape of learned parameters: [time, sites, phy_param_count * nmul]
        # phy_param_count = len(self.parameter_bounds)
        
        # Reshape learned parameters to [time, sites, layers, parameters, nmul]
        n_learned_layers = len(self.learnable_layers)
        n_learned_params = len(self.learnable_params)
        learned_params = torch.sigmoid(parameters).view(
            parameters.shape[0],  # time
            parameters.shape[1],  # sites
            n_learned_layers,
            n_learned_params,
            self.nmul,
        )

        param_dict = raw_parameters.clone().view(
            raw_parameters.shape[0],  # time
            raw_parameters.shape[1],  # sites
            self.layer_count,
            len(self.param_names),
        ).unsqueeze(-1).repeat(1, 1, 1, 1, self.nmul)

        # Overwrite estimate params with learned params for specific layers/params.
        # Iterate over the learned layers and parameters.
        for i, layer_idx in enumerate(self.learnable_layers):
            for j, name in enumerate(self.learnable_params):
                descaled_param = self.descale_param(
                    learned_params[:, :, i, j, :], name
                )
                param_idx = self.param_names.index(name)
                param_dict[:, :, layer_idx, param_idx, :] = descaled_param
        return param_dict

    def change_param_range(self, param: torch.Tensor, bounds: list[float]) -> torch.Tensor:
        """Change the range of a parameter to the specified bounds.
        
        Parameters
        ----------
        param : torch.Tensor
            The parameter.
        bounds : list[float]
            The parameter bounds.
        
        Returns
        -------
        out : torch.Tensor
            The parameter with the specified bounds.
        """
        out = param * (bounds[1] - bounds[0]) + bounds[0]
        return out

    def descale_param(
            self,
            parameter: torch.Tensor,
            name: str,
        ) -> torch.Tensor:
        """Descale physical parameters.
        
        Parameters
        ----------
        phy_params : torch.Tensor
            Normalized physical parameters.
        name : str
            Name of the physical parameter.
        
        Returns
        -------
        dict
            Dictionary of descaled physical parameters.
        """
        param = self.change_param_range(
            param=parameter,
            bounds=self.parameter_bounds[name],
        )
        return param
    
    def _init_layer_weights(
        self,
        param_dict: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute layer weights per site."""
        weights = []

        for layer in range(self.layer_count):
            compressibility = param_dict['parVCI'][:, layer] \
                * (1 + param_dict['parVR'][:, layer])
            fine_frac = (100 - param_dict['parK'][:, layer]) / 100
            depth_factor = np.exp(-(layer + 1) / self.layer_count)

            weight = compressibility * fine_frac * depth_factor
            weights.append(weight)
        weights = torch.stack(weights, dim=1)  # Shape: [batch_size, layer_count]

        total_weight = torch.sum(weights, dim=1, keepdim=True)
        weights = torch.where(total_weight > 0, weights / total_weight, weights)

        return weights
    
    def _calculate_compression(
        self,
        param_dict: dict[str, torch.Tensor],
        stress_delta: float,
    ) -> torch.Tensor:
        """Calculate compression for a single layer."""
        if torch.abs(stress_delta) < 1e-6:
            return torch.tensor(0.0, dtype=torch.float32, device=self.device)

        stress_0 = torch.clamp(param_dict['parIES'], min=self.min_stress)
        stress_final = torch.clamp(stress_0 + stress_delta, min=self.min_stress)

        # Preconsolidation pressure
        precon_p = torch.clamp(stress_0 * param_dict['parOCR'], min=self.min_stress)

        if stress_final <= precon_p:
            compression = (
                param_dict['parLT'] * param_dict['parRC'] / (1 + param_dict['parVR'])
            ) * torch.log(stress_final / stress_0)
        else:
            compression = (param_dict['parLT'] / (1 + param_dict['parVR'])) * (
                param_dict['parRC'] * torch.log(precon_p / stress_0) +
                param_dict['parVCI'] * torch.log(stress_final / precon_p)
            )
        # Clamp between 0 and 1.
        fine_frac = torch.clamp((100 - param_dict['parK']) / 100, 0, 1)

        return compression * (0.5 + 0.5 * fine_frac)

    def forward(
            self,
            x_dict: dict[str, torch.Tensor],
            parameters: torch.Tensor
        ) -> Union[tuple, dict[str, torch.Tensor]]:
        """Forward pass for TCM.
        
        Parameters
        ----------
        x_dict : dict
            Dictionary of input forcing data.
        parameters : torch.Tensor
            Unprocessed, learned parameters from a neural network.
        
        Returns
        -------
        Union[tuple, dict]
            Tuple or dictionary of model outputs.
        """
        # all_results = []
        # station_metrics = {}

        # Unpack forcing data and expand for nmul models.
        GW = x_dict['x_phy']
        GWm = GW.repeat(1, 1, self.nmul)

        n_steps, n_sites, _ = GW.size()

        # Unpack parameters
        param_dict = self.unpack_parameters(parameters, x_dict['c_phy'])

        # Initialize model state
        vert_displacement = torch.zeros(
            [n_steps, n_sites, self.nmul],
            dtype=torch.float32,
            device=self.device,
        )

        # Station loop
        for i in range(n_sites):
            for m in range(self.nmul):
                site_params = {}
                for k, name in enumerate(self.param_names):
                    site_params[name] = param_dict[:, i, :, k, m]
                
                layer_weights = self._init_layer_weights(site_params)
                
                # Time loop (months)
                for j in range(n_steps):
                    gw_level = GWm[j, i, m]

                    if j >= self.lookback:
                        prev_gw_level = GWm[j - self.lookback:j, i, m]
                    else:
                        prev_gw_level = None

                    # Calculate subsidence
                    vert_displacement[j, i, m] = self.PBM(
                        {key: site_params[key][j,] for key in self.param_names},
                        gw_level,
                        prev_gw_level,
                        layer_weights[j],
                    )

        out_dict = {
            'v_disp': torch.mean(vert_displacement, dim=-1),
        }
        return out_dict
        
    def PBM(
        self,
        param_dict: dict[str, torch.Tensor],
        gw_level: torch.Tensor,
        prev_gw_level: torch.Tensor,
        layer_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate monthly subsidence (mm) using the Terzaghi model."""
        total_disp = 0.0

        if prev_gw_level is not None:
            # time_factors = torch.exp(-torch.arange(prev_gw_level.shape[0], device=self.device))
            # historical_effect = torch.sum(prev_gw_level * time_factors.unsqueeze(1), dim=0)
        # else:
        #     historical_effect = torch.tensor(0.0, device=self.device)
            historical_effect = torch.tensor(0.0, device=self.device)
            for i in range(prev_gw_level.shape[0]):
                historical_effect += prev_gw_level[i] * np.exp(-prev_gw_level.shape[0] - i)
            gw_level += 0.3 * historical_effect

        # Loop through layers
        for i in range(self.layer_count):
            # Effective stress change
            depth_factor = np.exp(-(i + 1) / self.layer_count)
            stress_delta = gw_level * self.water_unit_weight * depth_factor

            # Apply layer properties
            compression = self._calculate_compression(
                {key: param_dict[key][i] for key in param_dict.keys()},
                stress_delta,
            )

            # Weight each layer
            total_disp += compression * layer_weights[i]

        # Convert to mm
        return total_disp * 10
