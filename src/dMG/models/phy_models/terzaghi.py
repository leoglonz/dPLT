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

        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if config is not None:
            # Overwrite defaults with config values.
            self.dy_drop = config.get('dy_drop', self.dy_drop)
            self.lookback = config.get('lookback', self.lookback)
            self.dynamic_params = config['dynamic_params'].get('Terzaghi', self.dynamic_params)
            self.variables = config.get('variables', self.variables)
            self.nearzero = config.get('nearzero', self.nearzero)
            self.nmul = config.get('nmul', self.nmul)
        self.set_parameters()

        # self.dfLayers = dfLayers
        # self.dfWaterlevel = dfWaterlevel
        # self.dfSubsidence = dfSubsidence
        # self.lookback = lookback
        # self.water_unit_weight = 9.81  # kN/m³
        # self.min_stress = 10.0  # kPa

    def set_parameters(self) -> None:
        """Get physical parameters."""
        self.phy_param_names = self.parameter_bounds.keys()
        self.learnable_param_count = len(self.phy_param_names) * self.nmul

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
    
    def _initialize_layer_weights(self, layer_data):
        weights = []
        num_layers = len(layer_data)
        for _, layer in layer_data.iterrows():
            compressibility = layer['compression_index'] * (1 + layer['void_ratio'])
            fine_fraction = (100 - layer['percent_coarse']) / 100
            depth_factor = torch.exp(-layer['layer_number'] / num_layers)
            weight = compressibility * fine_fraction * depth_factor
            weights.append(weight)

        weights = torch.tensor(weights, dtype=torch.float32)
        return weights / torch.sum(weights)
    

    def _calculate_compression(self, layer, stress_change):
        if torch.abs(stress_change) < 1e-6:
            return 0.0

        initial_stress = max(layer['initial_effective_stress'], self.min_stress)
        final_stress = max(initial_stress + stress_change, self.min_stress)

        cc = layer['compression_index']
        cr = layer['recompression_index']
        e0 = layer['void_ratio']
        ocr = layer['OCR']
        precons_pressure = initial_stress * ocr

        if final_stress <= precons_pressure:
            compression = (layer['thickness'] * cr / (1 + e0)) * \
                         torch.log(final_stress / initial_stress)
        else:
            compression = (layer['thickness'] / (1 + e0)) * \
                         (cr * torch.log(precons_pressure / initial_stress) +
                          cc * torch.log(final_stress / precons_pressure))

        fine_fraction = (100 - layer['percent_coarse']) / 100
        compression *= (0.5 + 0.5 * fine_fraction)

        return compression

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

        stations = self.dfLayers['station_id'].unique()

        for station in stations:
            print(f"Processing station: {station}")
            station_results = self.forward(station)

            if station_results is not None:
                station_results['station_id'] = station
                all_results.append(station_results)

                predictions = torch.tensor(station_results['predicted_subsidence'].values, dtype=torch.float32)
                actuals = torch.tensor(station_results['actual_subsidence'].values, dtype=torch.float32)

                rmse = torch.sqrt(torch.mean((predictions - actuals) ** 2)).item()
                r2 = 1 - torch.sum((predictions - actuals) ** 2).item() / torch.sum((actuals - actuals.mean()) ** 2).item()

                station_metrics[station] = {
                    'RMSE': rmse,
                    'R2': r2,
                    'n_predictions': len(station_results)
                }

        results_df = pd.concat(all_results, ignore_index=True)
        metrics_df = pd.DataFrame(station_metrics).T

        return results_df, metrics_df

    def PBM(self, station_id: int) -> torch.Tensor:
        station_layers = self.dfLayers[self.dfLayers['station_id'] == station_id].copy()
        station_subsidence = self.dfSubsidence[self.dfSubsidence['station_id'] == station_id].sort_values('Year')
        station_water = self.dfWaterlevel[self.dfWaterlevel['station_id'] == station_id].sort_values('year')
        
        
        layer_weights = self._initialize_layer_weights(station_layers)
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
    