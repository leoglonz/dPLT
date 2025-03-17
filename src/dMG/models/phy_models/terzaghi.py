from typing import Any, Dict, Optional, Tuple, Union, List

import torch
import pandas as pd


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
            config: Optional[Dict[str, Any]] = None,
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
        self.learnable_layers = [0]
        self.learnable_params = ['parLT']
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
            'parK': [0, 1],  # Percent coarseness
            'parMv': [0, 10],  # Coefficient of volume compressibility
            'parIES': [0, 100],  # Initial effective stress
        }
        self.water_unit_weight = 9.81  # kN/m³
        self.min_stress = 10.0  # kPa

        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if config is not None:
            # Overwrite defaults with config values.
            self.dy_drop = config.get('dy_drop', self.dy_drop)
            self.lookback = config.get('lookback', self.lookback)
            self.layer_count = config.get('layer_count', self.layer_count)
            self.learnable_layers = config.get('learnable_layers', self.learnable_layers)
            self.dynamic_params = config['dynamic_params'].get('Terzaghi', self.dynamic_params)
            self.variables = config.get('variables', self.variables)
            self.nearzero = config.get('nearzero', self.nearzero)
            self.nmul = config.get('nmul', self.nmul)
        self.set_parameters()

    def set_parameters(self) -> None:
        """Get physical parameters.
        
        # of parameters = # of layers * # of parameters to learn per
        layer * # of layers
        """
        self.phy_param_names = self.parameter_bounds.keys()
        self.learnable_param_count = len(self.learnable_params) \
            * len(self.learnable_layers) * self.nmul

    def unpack_parameters(
            self,
            parameters: torch.Tensor,
        ) -> Dict[str, torch.Tensor]:
        """Extract physical model parameters from NN output.
        
        Parameters
        ----------
        parameters : torch.Tensor
            Unprocessed, learned parameters from a neural network.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of physical parameters.
        """
        phy_param_count = len(self.parameter_bounds)
        
        # Physical parameters
        phy_params = torch.sigmoid(
            parameters[:, :, :phy_param_count * self.nmul]
        ).view(
            parameters.shape[0],
            parameters.shape[1],
            phy_param_count,
            self.nmul,
        )
        return phy_params

    def change_param_range(param: torch.Tensor, bounds: List[float]) -> torch.Tensor:
        """Change the range of a parameter to the specified bounds.
        
        Parameters
        ----------
        param : torch.Tensor
            The parameter.
        bounds : List[float]
            The parameter bounds.
        
        Returns
        -------
        out : torch.Tensor
            The parameter with the specified bounds.
        """
        out = param * (bounds[1] - bounds[0]) + bounds[0]
        return out

    def descale_phy_parameters(
            self,
            phy_params: torch.Tensor,
            dy_list:list,
        ) -> torch.Tensor:
        """Descale physical parameters.
        
        Parameters
        ----------
        phy_params : torch.Tensor
            Normalized physical parameters.
        dy_list : list
            List of dynamic parameters.
        
        Returns
        -------
        dict
            Dictionary of descaled physical parameters.
        """
        n_steps = phy_params.size(0)
        n_grid = phy_params.size(1)

        param_dict = {}
        pmat = torch.ones([1, n_grid, 1]) * self.dy_drop
        for i, name in enumerate(self.parameter_bounds.keys()):
            staPar = phy_params[-1, :, i,:].unsqueeze(0).repeat([n_steps, 1, 1])
            if name in dy_list:
                dynPar = phy_params[:, :, i,:]
                drmask = torch.bernoulli(pmat).detach_().cuda() 
                comPar = dynPar * (1 - drmask) + staPar * drmask
                param_dict[name] = self.change_param_range(
                    param=comPar,
                    bounds=self.parameter_bounds[name]
                )
            else:
                param_dict[name] = self.change_param_range(
                    param=staPar,
                    bounds=self.parameter_bounds[name]
                )
        return param_dict
    
    def _init_layer_weights(
        self,
        param_dict: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute layer weights per site."""
        weights = []

        for layer in range(self.layer_count):
            compressibility = param_dict['parVCI'][:, layer] \
                * (1 + param_dict['parVR'][:, layer])
            fine_fraction = (100 - param_dict['parK'][:, layer]) / 100
            depth_factor = torch.exp(- layer / self.layer_count)

            weight = compressibility * fine_fraction * depth_factor
            weights.append(weight)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        return weights / torch.sum(weights)
    
    def _calculate_compression(
        self,
        param_dict: Dict[str, torch.Tensor],
        stress_delta,
    ) -> torch.Tensor:
        """Calculate compression for a single layer."""
        if torch.abs(stress_delta) < 1e-6:
            return 0.0

        stress_0 = max(param_dict['parIES'], self.stress_min)
        stress_final = max(stress_0 + stress_delta, self.stress_min)

        # Preconsolidation pressure
        precon_p = stress_0 * param_dict['parOCR']

        if stress_final <= precon_p:
            compression = (
                param_dict['parLT'] * param_dict['parRC'] / (1 + param_dict['parVR'])
            ) * torch.log(stress_final / stress_0)
        else:
            compression = (param_dict['thickness'] / (1 + param_dict['parVR'])) * \
                (
                    param_dict['parRC'] * torch.log(precon_p / stress_0) +
                    param_dict['parVCI'] * torch.log(stress_final / precon_p)
                )
        fine_frac = (100 - param_dict['percent_coarse']) / 100

        return compression * (0.5 + 0.5 * fine_frac)

    def forward(
            self,
            x_dict: Dict[str, torch.Tensor],
            parameters: torch.Tensor
        ) -> Union[Tuple, Dict[str, torch.Tensor]]:
        """Forward pass for TCM.
        
        Parameters
        ----------
        x_dict : dict
            Dictionary of input forcing data.
        parameters : torch.Tensor
            Unprocessed, learned parameters from a neural network.
        
        Returns
        -------
        Union[Tuple, dict]
            Tuple or dictionary of model outputs.
        """
        all_results = []
        station_metrics = {}

        n_tsteps = x_dict['forcings'].shape[0]
        n_sites = x_dict['forcings'].shape[1]

        water_level = x_dict['forcings']

        # Unpack parameters # NOTE: needs to be fixed
        param_dict = self.unpack_parameters(parameters)

        # Initialize model state
        vert_displacement = torch.zeros(
            [n_tsteps, n_sites],
            dtype=torch.float32,
            device=self.device,
        )

        for i in range(n_sites):
            site_params = {}
            for key in param_dict.keys():
                site_params[key] = param_dict[key][:, i, :]

            layer_weights = self._init_layer_weights(site_params)

        out_dict = {
            'vert_displacement': vert_displacement,
        }
        return out_dict
        
    def PBM(self, station_id: int) -> torch.Tensor:
        station_layers = self.dfLayers[self.dfLayers['station_id'] == station_id].copy()
        station_subsidence = self.dfSubsidence[self.dfSubsidence['station_id'] == station_id].sort_values('Year')
        station_water = self.dfWaterlevel[self.dfWaterlevel['station_id'] == station_id].sort_values('year')
        
        
        layer_weights = self._init_layer_weights(station_layers)
        water_changes = station_water.set_index('year')['level_change_mean'].to_dict()

        results = []
        num_layers = len(station_layers)
        for _, row in station_subsidence.iterrows():
            year = row['Year']
            actual_change = row['yearly_change']
            current_water_change = water_changes.get(year)

            if pd.notna(current_water_change) and pd.notna(actual_change):
                prev_changes = [water_changes.get(year - i) for i in range(1, self.lookback + 1)]
                historical_effect = 0
                if prev_changes:
                    historical_effect = sum(change * torch.exp(torch.tensor(-(len(prev_changes) - i), dtype=torch.float32))
                                            for i, change in enumerate(prev_changes) if pd.notna(change))
                current_water_change += 0.3 * historical_effect

                total_subsidence = 0
                for i, layer in station_layers.iterrows():
                    depth_factor = torch.exp(torch.tensor(-layer['layer_number'] / num_layers, dtype=torch.float32))
                    stress_change = current_water_change * self.water_unit_weight * depth_factor
                    compression = self._calculate_compression(layer, stress_change)
                    total_subsidence += compression * layer_weights[i]

                results.append({
                    'year': year,
                    'predicted_subsidence': total_subsidence.item() * 1000,  # mm
                    'actual_subsidence': actual_change,
                    'water_change': current_water_change
                })
        return pd.DataFrame(results) if results else None
    