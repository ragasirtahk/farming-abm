import numpy as np
import os
from agents import Farmer, Forecaster
from model import FarmingModel
from experiments import (
    analyze_dissemination_impact,
    simulate_economic_crisis,
    analyze_crop_yields,
    analyze_farmer_crop_choices,
    analyze_forecast_impact_on_inequality,
    analyze_forecast_impact_on_trust,
    simulate_forecast_system_failure,
    analyze_trust_recovery,
    analyze_accuracy_impact_on_yields,
    run_comparable_simulations,
    analyze_subsidy_allocation,
    analyze_peer_influence_on_trust,
    analyze_forecast_accuracy_improvement,
    analyze_forecast_impact
)
from sensitivity_tests import sensitivity_analysis, sensitivity_analysis_dist
from robustness_tests import (
    robustness_checks,
    robustness_check_time_and_scale,
    robustness_check_subsidy_levels,                # Added
    robustness_check_wealth_land_distributions      # Added
)
from summary_stats import generate_summary_statistics  # Import the summary stats function
import statistical_analysis
from statistical_analysis import analyze_dissemination_impact_stats  # Import the analysis function
from statistical_analysis import analyze_forecast_accuracy_relation
from statistical_analysis import analyze_simulate_forecast_system_failure
from statistical_analysis import analyze_trust_recovery_r
from statistical_analysis import analyze_robustness_time_and_scale

# Main parameters
num_farmers = 50
num_seasons = 40
forecast_accuracy = 0.8
subsidy_level = 10
crisis_season = 10
recovery_duration = 5
accuracy_levels = [0.6, 0.7, 0.8]
subsidy_levels = [0, 10, 20, 50]
distributions = {
    "Normal": {"func": np.random.normal, "params": {"loc": 1500, "scale": 500}},
    "Uniform": {"func": np.random.uniform, "params": {"low": 1000, "high": 2000}},
    "Log-Normal": {"func": np.random.lognormal, "params": {"mean": 7.188, "sigma": 0.5}},
    "Pareto": {"func": lambda size: (np.random.pareto(a=3.0, size=size) + 1) * 1000, "params": {}}
}
seeds = [42, 99, 123, 2024]
param_ranges = {
    "Forecast Accuracy": [0.2, 0.5, 0.8],
    "Subsidy Levels": [5, 10, 20],
    "Rainfall Variability": [20, 50, 100],
}

# Additional experiment-specific parameters
# Previously: [("Radio", 0.25), ("Mobile App", 0.25), ("Extension Officer", 0.25), ("Community Leaders", 0.25)]
# Updated to reflect more realistic distribution
dissemination_modes = [
    ("Radio", 0.50),            # Increased prevalence
    ("Mobile App", 0.20),
    ("Extension Officer", 0.20),
    ("Community Leaders", 0.10)
]

accuracy_crash_start = 10
accuracy_crash_duration = 5
extended_num_seasons = 40
crisis_scenario_num_seasons = 40
crisis_scenario_time = 20
extended_num_seasons = 40

# Experiment scenarios
scenarios = [
    {"Label": "Short Run, Small Scale", "num_farmers": 50, "num_seasons": 20},
    {"Label": "Short Run, Large Scale", "num_farmers": 100, "num_seasons": 20},
    {"Label": "Long Run, Small Scale", "num_farmers": 50, "num_seasons": 50},
    {"Label": "Long Run, Large Scale", "num_farmers": 100, "num_seasons": 50},
]

def main():
    run_comparable_simulations()
    # Summary Statistics
    generate_summary_statistics()
    
    # Experiments
    analyze_dissemination_impact(num_farmers, num_seasons, forecast_accuracy, subsidy_level, dissemination_modes)
    simulate_economic_crisis(num_farmers, crisis_scenario_num_seasons, forecast_accuracy, subsidy_level, crisis_scenario_time)
    analyze_crop_yields(num_farmers, num_seasons, forecast_accuracy, subsidy_level)
    analyze_farmer_crop_choices(num_farmers, num_seasons, forecast_accuracy, subsidy_level)
    analyze_forecast_impact_on_inequality(num_farmers, num_seasons, accuracy_levels, subsidy_level)
    analyze_forecast_impact_on_trust(num_farmers, num_seasons, accuracy_levels, subsidy_level)
    # simulate_forecast_system_failure(num_farmers, extended_num_seasons, forecast_accuracy, subsidy_level, accuracy_crash_start, accuracy_crash_duration)
    # analyze_trust_recovery(num_farmers, extended_num_seasons, forecast_accuracy, subsidy_level, crisis_season, recovery_duration)
    run_comparable_simulations()
    analyze_accuracy_impact_on_yields(num_farmers, num_seasons, accuracy_levels, subsidy_level)
    analyze_forecast_impact
    print("Base experiments completed.")

    # Run all statistical analyses after generating necessary data
    analyze_forecast_impact(
        num_farmers=50,
        num_seasons=40,
        forecast_accuracies=[0.6, 0.7, 0.8],
        subsidy_level=100
    )
    statistical_analysis.analyze_forecast_accuracy_relation("results/analysis/analyze_forecast_accuracy_relation.csv")

    # Perform statistical analysis on dissemination impact
    analysis_file = "results/analysis/analyze_dissemination_impact.csv"
    analyze_dissemination_impact_stats(analysis_file)

    analyze_simulate_forecast_system_failure()
    analyze_trust_recovery_r()

    # Additional Experiments
    # 1. Subsidy Allocation Strategies
    # for strategy in ["equal", "targeted_large", "targeted_trust"]:
    #     analyze_subsidy_allocation(num_farmers, num_seasons, forecast_accuracy, subsidy_level, strategy, dissemination_modes)
    
    # 2. Peer Influence on Trust Levels
    # analyze_peer_influence_on_trust(num_farmers, num_seasons, forecast_accuracy, subsidy_level, dissemination_modes, peer_influence=True)
    # analyze_peer_influence_on_trust(num_farmers, num_seasons, forecast_accuracy, subsidy_level, dissemination_modes, peer_influence=False)
    
    # 3. Long-Term Forecast Accuracy Improvements
    # analyze_forecast_accuracy_improvement(num_farmers, extended_num_seasons, forecast_accuracy, subsidy_level, dissemination_modes)
    
    print("Base stats completed.")

    # Sensitivity Analysis
    sensitivity_analysis(num_farmers, num_seasons, forecast_accuracy, subsidy_levels)
    sensitivity_analysis_dist(num_farmers, num_seasons, forecast_accuracy, subsidy_level, distributions, dissemination_modes)
    print("Sensitivity analysis completed.")

    # Robustness Checks
    robustness_checks(num_farmers, num_seasons, seeds, param_ranges, dissemination_modes)
    robustness_check_time_and_scale(scenarios, forecast_accuracy, subsidy_level)
    analyze_robustness_time_and_scale("results/robustness_checks/time_scale_results.csv")
    
    # Additional Robustness Tests
    # 1. Parameter Variation and Scenario Analysis
    robustness_checks(num_farmers, num_seasons, seeds, param_ranges, dissemination_modes)
    robustness_check_time_and_scale(scenarios, forecast_accuracy, subsidy_level)
    
    # 2. Subsidy Level Variation
    robustness_check_subsidy_levels(num_farmers, num_seasons, subsidy_levels, forecast_accuracy, dissemination_modes)
    
    # 3. Wealth and Land Size Distributions
    wealth_distributions = {
        "Pareto": lambda n: (np.random.pareto(a=1.5, size=n) + 1) * 1500,
        "Normal": lambda n: np.random.normal(loc=1500, scale=500, size=n),
        "Exponential": lambda n: np.random.exponential(scale=1500, size=n)
    }
    land_size_distributions = {
        "Pareto": lambda n: (np.random.pareto(a=2.5, size=n) + 1) * 3,
        "Normal": lambda n: np.random.normal(loc=3.0, scale=0.5, size=n),
        "Exponential": lambda n: np.random.exponential(scale=3.0, size=n)
    }
    robustness_check_wealth_land_distributions(num_farmers, num_seasons, wealth_distributions, land_size_distributions, forecast_accuracy, subsidy_level, dissemination_modes)
    
    print("Robustness tests completed.")
    
    print("All sections completed. Results and visualizations saved.")

if __name__ == "__main__":
    # Ensure the analysis directory exists
    os.makedirs("results/analysis", exist_ok=True)
    main()
