########Measels#############

import os
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import wandb
from scipy.integrate import odeint

from .base_generator import BaseAnomalyGenerator
from .registry import AnomalyInjectionRegistry


def measles_sir_model(y: np.ndarray, t: float, params: Dict[str, Any]) -> list:
    """
    Defines the SIR model equations for measles transmission among adults and children.

    Args:
        y (np.ndarray): Current state of the compartments.
        t (float): Current time point.
        params (Dict[str, Any]): Parameters for the SIR model.

    Returns:
        list: Derivatives of each compartment.
    """
    S_a, E_a, I1_a, I2_a, I3_a, R_a = y
    N_a = params['population']['adults']
    N = N_a
    
    beta_a = params['transmission_rates']['adults']
    gamma_1_a = params['progression_rates']['adults']['exposed_to_infected']
    gamma_2_a = params['progression_rates']['adults']['infected_stage1_to_stage2']
    gamma_3_a = params['progression_rates']['adults']['infected_stage2_to_stage3']
    recovery_rate_a = params['progression_rates']['adults']['infected_stage3_to_recovered']
 
    
    # Adult equations
    dS_a = -beta_a * S_a * (I1_a + I2_a + I3_a) / N
    dE_a = beta_a * S_a * (I1_a + I2_a + I3_a) / N - gamma_1_a * E_a
    dI1_a = gamma_1_a * E_a - gamma_2_a * I1_a
    dI2_a = gamma_2_a * I1_a - gamma_3_a * I2_a
    dI3_a = gamma_3_a * I2_a - recovery_rate_a * I3_a
    dR_a = recovery_rate_a * I3_a
    
    
    return [dS_a, dE_a, dI1_a, dI2_a, dI3_a, dR_a]


@AnomalyInjectionRegistry.register("sir_measles")
class MeaslesSIRAnomalyGenerator(BaseAnomalyGenerator):
    """
    An anomaly generator based on the SIR model for measles transmission.
    This class simulates the spread of measles among adults and children and generates anomalies based on the reported symptoms.
    
    Attributes:
        sir_params (Dict[str, Any]): Parameters for the SIR model simulation.
        injection_start_index (int): Index to start injecting anomalies.
        logger: Logger for tracking and visualizing results.
    """
    def __init__(self, dataset: pd.DataFrame, sir_params: Dict[str, Any], injection_start_index: int, logger=None):
        """
        Initializes the MeaslesSIRAnomalyGenerator.

        Args:
            dataset (pd.DataFrame): The input dataset.
            sir_params (Dict[str, Any]): Parameters for the SIR model simulation.
            injection_start_index (int): Index to start injecting anomalies.
            logger: Logger for tracking and visualizing results (optional).
        """
        super().__init__(dataset)
        self.sir_params = sir_params
        self.injection_start_index = injection_start_index
        self.logger = logger

    def generate(self) -> Dict[str, np.ndarray]:
        """
        Generates anomalies based on the SIR model simulation.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing anomalies for each reported symptom.
        """
        time_points, solution = self._generate_sir_data()
        compartments = self._extract_compartments(solution)
        reported_symptoms = self._calculate_reported_symptoms(compartments)
        
        anomalies = {}
        for symptom, values in reported_symptoms.items():
            anomaly_indices = np.arange(self.injection_start_index, self.injection_start_index + len(values))
            anomalies[symptom] = {'indices': anomaly_indices, 'values': values}
        
        if self.logger:
            self.plot_results()
        return anomalies

    def _generate_sir_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates SIR model data over the specified time duration.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Time points and the corresponding solution of the SIR model.
        """
        start_day = self.sir_params['simulation_duration']['start_day']
        end_day = self.sir_params['simulation_duration']['end_day']
        time_points = np.linspace(start_day, end_day, end_day - start_day + 1)
        
        initial_state = [
            self.sir_params['initial_conditions']['adults']['susceptible'],
            self.sir_params['initial_conditions']['adults']['exposed'],
            self.sir_params['initial_conditions']['adults']['infected_stage1'],
            self.sir_params['initial_conditions']['adults']['infected_stage2'],
            self.sir_params['initial_conditions']['adults']['infected_stage3'],
            self.sir_params['initial_conditions']['adults']['recovered'],
        ]

        solution = odeint(measles_sir_model, initial_state, time_points, args=(self.sir_params,))
        return time_points, solution

    def _extract_compartments(self, solution: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extracts the compartments from the SIR model solution.

        Args:
            solution (np.ndarray): Solution array from the SIR model.
        
        Returns:
            Dict[str, np.ndarray]: A dictionary containing the values of each compartment over time.
        """
        compartment_names = [
            'Susceptible_Adult', 'Exposed_Adult', 'Infected1_Adult', 'Infected2_Adult', 'Infected3_Adult', 'Recovered_Adult',
        ]
        return dict(zip(compartment_names, solution.T))

    def _calculate_reported_symptoms(self, compartments: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Calculates the reported symptoms based on the compartment values and symptom probabilities.

        Args:
            compartments (Dict[str, np.ndarray]): A dictionary containing the compartment values over time.
        
        Returns:
            Dict[str, np.ndarray]: A dictionary containing the reported symptoms for each compartment.
        """
        symptom_probs = self.sir_params['symptom_probabilities']
        
        I1a, I2a, I3a = compartments['Infected1_Adult'], compartments['Infected2_Adult'], compartments['Infected3_Adult']
        return {
            'Fever_-_adult': symptom_probs['fever']['adults'] * 0.8* (I1a + I2a),
            'Fever_-_child': symptom_probs['fever']['children'] * 0.2* (I1a + I2a),
            'Cough_-_adult': symptom_probs['cough']['adults'] *0.8* (I1a + I2a),
            'Cough_-_child': symptom_probs['cough']['children'] * 0.2*(I1a + I2a),
            'Eye_problems_-_adult': symptom_probs['conjunctivitis']['adults'] *0.8* (I1a + I2a),
            'Eye_problems_-_children': symptom_probs['conjunctivitis']['children'] *0.2* (I1a + I2a),
            'Rash_-_adult': symptom_probs['rash']['adults'] *0.8* I3a,
            'Rash_-_children': symptom_probs['rash']['children'] *0.2* I3a,
        }

    def plot_results(self) -> None:
        """
        Generates a single Plotly figure showing the dynamics of adults and children compartments and logs it to wandb.
        """
        time_points, solution = self._generate_sir_data()
        compartments = self._extract_compartments(solution)
        adjusted_time = time_points + self.injection_start_index

        fig = go.Figure()

        # Adults
        fig.add_trace(go.Scatter(x=adjusted_time, y=compartments['Susceptible_Adult'], mode='lines', name='Susceptible'))
        fig.add_trace(go.Scatter(x=adjusted_time, y=compartments['Exposed_Adult'], mode='lines', name='Exposed'))
        infected_adults = compartments['Infected1_Adult'] + compartments['Infected2_Adult'] + compartments['Infected3_Adult']
        fig.add_trace(go.Scatter(x=adjusted_time, y=infected_adults, mode='lines', name='Infected'))
        fig.add_trace(go.Scatter(x=adjusted_time, y=compartments['Recovered_Adult'], mode='lines', name='Recovered'))


        # Now explicitly set both the 'plot_bgcolor' and 'paper_bgcolor' to white.
        fig.update_layout(
                    plot_bgcolor='white',  # Set the plot background color to white
                    paper_bgcolor='white',  # Set the paper background color (area around the plot) to white
                    font=dict(color='black'),  # Optional: Set font color to black for better contrast
                    
                    # Update the grid lines for both axes (x-axis and y-axis)
                    xaxis=dict(
                        showgrid=True,          # Show grid lines
                        gridcolor='black',      # Set the grid lines color to black
                        zeroline=True,          # Show the line at zero
                        zerolinecolor='black',   # Set the zero line color to black
                        linecolor='black'      # Set the x-axis line color to black
                    ),
                    yaxis=dict(
                        showgrid=True,          # Show grid lines
                        gridcolor='black',      # Set the grid lines color to black
                        zeroline=True,          # Show the line at zero
                        zerolinecolor='black',   # Set the zero line color to black
                        linecolor='black'      # Set the y-axis line color to black
                )
        )

        fig.update_layout(
            title='SIR Model for Measles',
            xaxis_title='Time',
            yaxis_title='Number of individuals',
            hovermode='closest',
            legend_title="Compartments",
            legend=dict(font=dict(size=20)),
            xaxis=dict(title=dict(text='Time', font=dict(size=20)), tickfont=dict(size=16)),
            yaxis=dict(title=dict(text='Number of individuals', font=dict(size=20)), tickfont=dict(size=16))
        )
        # Save the figure as HTML and log to wandb
        if self.logger:
            output_dir = self.logger.save_dir if hasattr(self.logger, 'save_dir') else './wandb_logs'
            os.makedirs(output_dir, exist_ok=True)
            plotly_html_path = os.path.join(output_dir, 'sir_measles_plot.html')
            fig.write_html(plotly_html_path, auto_play=False)

            table = wandb.Table(columns=["Plotly Figure"])
            table.add_data(wandb.Html(plotly_html_path))

            self.logger.experiment.log({"SIR Measles Plot": table})