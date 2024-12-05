from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import numpy as np
from agents import Farmer, Forecaster
from utils import gini

class FarmingModel(Model):
    def __init__(self, num_farmers, num_seasons, forecast_accuracy, subsidy_level, 
                 total_initial_wealth=None, total_initial_land=None,
                 wealth_distribution=None, land_size_distribution=None, dissemination_modes=None,
                 seed=None):  # Ensure seed parameter is included
        """
        Farming model initialization with customizable parameter distributions.
        Args:
        - num_farmers: Number of farmers in the model.
        - num_seasons: Number of seasons to simulate.
        - forecast_accuracy: Initial accuracy of weather forecasting.
        - subsidy_level: Government subsidy level.
        - wealth_distribution: Function to generate initial wealth (default: pareto).
        - land_size_distribution: Function to generate land sizes (default: pareto).
        - dissemination_modes: List of dissemination modes with their probabilities.
        - seed (int, optional): Seed for random number generators to ensure reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)  # Set the seed for numpy
        
        self.num_farmers = num_farmers
        self.num_seasons = num_seasons
        self.forecast_accuracy = forecast_accuracy
        self.subsidy_level = subsidy_level
        self.schedule = RandomActivation(self)
        
        # Default dissemination modes if none provided
        if dissemination_modes is None:
            dissemination_modes = [
                ("Radio", 0.50),            # Increased prevalence
                ("Mobile App", 0.20),
                ("Extension Officer", 0.20),
                ("Community Leaders", 0.10)
            ]
        
        # Generate dissemination modes for each farmer
        dissemination_mode_choices, dissemination_mode_probs = zip(*dissemination_modes)
        farmer_dissemination_modes = np.random.choice(dissemination_mode_choices, num_farmers, p=dissemination_mode_probs)
        
        self.forecaster = Forecaster(0, self, forecast_accuracy, dissemination_mode="Radio")  # Default mode for forecaster

        # Default distributions if none provided
        if wealth_distribution is None:
            wealth_distribution = lambda size: (np.random.pareto(a=1.5, size=size) + 1) * 1500
        if land_size_distribution is None:
            land_size_distribution = lambda size: (np.random.pareto(a=2.5, size=size) + 1) * 3

        # Generate initial wealth and land sizes using provided distributions
        if wealth_distribution:
            initial_wealths = wealth_distribution(size=num_farmers)
        else:
            initial_wealths = np.random.uniform(1000, 2000, size=num_farmers)
        
        if land_size_distribution:
            initial_land_sizes = land_size_distribution(size=num_farmers)
        else:
            initial_land_sizes = np.random.uniform(1, 5, size=num_farmers)
        
        # Scale initial wealths to match total_initial_wealth
        if total_initial_wealth is not None:
            scaling_factor = total_initial_wealth / initial_wealths.sum()
            initial_wealths *= scaling_factor
        
        # Scale initial land sizes to match total_initial_land
        if total_initial_land is not None:
            scaling_factor = total_initial_land / initial_land_sizes.sum()
            initial_land_sizes *= scaling_factor

        # Initialize farmers with adjusted wealth and land sizes
        for i in range(num_farmers):
            farmer = Farmer(
                unique_id=i,
                model=self,
                land_size=initial_land_sizes[i],
                initial_wealth=initial_wealths[i],
                dissemination_mode=farmer_dissemination_modes[i]  # Ensure correct dissemination mode
            )
            self.schedule.add(farmer)

        # Data collection
        self.datacollector = DataCollector(
            model_reporters={
                "Average Wealth": lambda m: np.mean([a.wealth for a in m.schedule.agents if isinstance(a, Farmer)]),
                "Average Trust": lambda m: np.mean([a.trust_level for a in m.schedule.agents if isinstance(a, Farmer)]),
                "Gini Coefficient": lambda m: gini([a.wealth for a in m.schedule.agents if isinstance(a, Farmer)])
            },
            agent_reporters={
                "Trust Level": "trust_level",
                "Crop Type": "crop_type",
                "Total Yield": "total_yield"  # Added to track total yield
            }
        )

    def step(self):
        """Advance the model by one step."""
        self.datacollector.collect(self)
        self.schedule.step()