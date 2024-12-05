import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model import FarmingModel
from utils import gini
from agents import Farmer

# Ensure the results and analysis directories exist
os.makedirs("results/analysis", exist_ok=True)

# Define a consistent color palette
palette = sns.color_palette("Set2")

def analyze_dissemination_impact(num_farmers, num_seasons, forecast_accuracy, subsidy_level, dissemination_modes):
    results_dissemination_time = []

    # Define desired total initial wealth and land
    desired_total_wealth = num_farmers * 1500
    desired_total_land = num_farmers * 3

    # Define land size distribution (consistent with sensitivity_tests.py)
    land_size_distribution = lambda size: (np.random.pareto(a=2.5, size=size) + 1) * 3

    # Define wealth distribution (default or customized)
    wealth_distribution = lambda size: (np.random.pareto(a=1.5, size=size) + 1) * 1500

    # Scale wealth
    initial_wealths = wealth_distribution(num_farmers)
    scaling_factor_wealth = desired_total_wealth / initial_wealths.sum()
    scaled_initial_wealths = initial_wealths * scaling_factor_wealth

    # Scale land sizes
    initial_land_sizes = land_size_distribution(num_farmers)
    scaling_factor_land = desired_total_land / initial_land_sizes.sum()
    scaled_initial_land_sizes = initial_land_sizes * scaling_factor_land

    # Define distribution functions
    wealth_dist_func = lambda size: scaled_initial_wealths[:size]
    land_dist_func = lambda size: scaled_initial_land_sizes[:size]

    model = FarmingModel(
        num_farmers,
        num_seasons,
        forecast_accuracy,
        subsidy_level,
        wealth_distribution=wealth_dist_func,
        land_size_distribution=land_dist_func,
        dissemination_modes=dissemination_modes
    )

    for season in range(num_seasons):
        model.step()
        for mode, _ in dissemination_modes:
            avg_wealth = np.mean([a.wealth for a in model.schedule.agents if isinstance(a, Farmer) and a.dissemination_mode == mode])
            avg_trust = np.mean([a.trust_level for a in model.schedule.agents if isinstance(a, Farmer) and a.dissemination_mode == mode])
            gini_value = gini([a.wealth for a in model.schedule.agents if isinstance(a, Farmer) and a.dissemination_mode == mode])
            results_dissemination_time.append({
                "Season": season + 1,
                "Dissemination Mode": mode,
                "Average Wealth": avg_wealth,
                "Average Trust": avg_trust,
                "Gini Coefficient": gini_value,
            })

    results_dissemination_time = pd.DataFrame(results_dissemination_time)
    results_dissemination_time.to_csv("results/analysis/analyze_dissemination_impact.csv", index=False)

    # Plot Wealth over Time by Dissemination Mode
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=results_dissemination_time,
        x="Season",
        y="Average Wealth",
        hue="Dissemination Mode",
        palette=palette[:len(dissemination_modes)]  # Adjust palette to match categories
    )
    plt.title("Wealth by Dissemination Mode")
    plt.ylabel("Average Wealth")
    plt.xlabel("Season")
    plt.legend(title="Dissemination Mode")
    plt.tight_layout()
    plt.savefig("results/analysis/wealth_dissemination_modes.png")
    plt.close()

    # Plot Gini Coefficient over Time by Dissemination Mode
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=results_dissemination_time,
        x="Season",
        y="Gini Coefficient",
        hue="Dissemination Mode",
        palette=palette[:len(dissemination_modes)]  # Adjust palette to match categories
    )
    plt.title("Inequality by Dissemination Mode")
    plt.ylabel("Gini Coefficient")
    plt.xlabel("Season")
    plt.legend(title="Dissemination Mode")
    plt.tight_layout()
    plt.savefig("results/analysis/gini_dissemination_modes.png")
    plt.close()

    # Plot Trust over Time by Dissemination Mode
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=results_dissemination_time,
        x="Season",
        y="Average Trust",
        hue="Dissemination Mode",
        palette=palette[:len(dissemination_modes)]  # Adjust palette to match categories
    )
    plt.title("Trust by Dissemination Mode")
    plt.ylabel("Average Trust")
    plt.xlabel("Season")
    plt.legend(title="Dissemination Mode")
    plt.tight_layout()
    plt.savefig("results/analysis/trust_dissemination_modes.png")
    plt.close()

    # Experiment 2: Inequality Trends
    model = FarmingModel(num_farmers, num_seasons, forecast_accuracy, subsidy_level)
    inequality_trend = []
    for season in range(num_seasons):
        model.step()
        inequality_trend.append(gini([a.wealth for a in model.schedule.agents if isinstance(a, Farmer)]))

    # Visualize inequality trend
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_seasons), inequality_trend, marker='o', label="Gini Coefficient")
    plt.title("Inequality Trends Over Seasons")
    plt.xlabel("Season")
    plt.ylabel("Gini Coefficient")
    plt.grid()
    plt.legend()
    plt.savefig("results/inequality_trends.png")
    plt.close()

    # Experiment 3: Trust Dynamics
    trust_trend = []
    model = FarmingModel(num_farmers, num_seasons, forecast_accuracy, subsidy_level)
    for season in range(num_seasons):
        model.step()
        avg_trust = np.mean([a.trust_level for a in model.schedule.agents if isinstance(a, Farmer)])
        trust_trend.append(avg_trust)

    # Visualize trust dynamics
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_seasons), trust_trend, marker='o', color="green", label="Average Trust")
    plt.title("Trust Dynamics Over Seasons")
    plt.xlabel("Season")
    plt.ylabel("Average Trust Level")
    plt.grid()
    plt.legend()
    plt.savefig("results/trust_dynamics.png")
    plt.close()

def simulate_economic_crisis(num_farmers, num_seasons, forecast_accuracy, subsidy_level, crisis_time):
    # Define desired total initial wealth and land
    desired_total_wealth = num_farmers * 1500
    desired_total_land = num_farmers * 3

    # Define land size distribution
    land_size_distribution = lambda size: (np.random.pareto(a=2.5, size=size) + 1) * 3

    # Define wealth distribution
    wealth_distribution = lambda size: (np.random.pareto(a=1.5, size=size) + 1) * 1500

    # Scale wealth
    initial_wealths = wealth_distribution(num_farmers)
    scaling_factor_wealth = desired_total_wealth / initial_wealths.sum()
    scaled_initial_wealths = initial_wealths * scaling_factor_wealth

    # Scale land sizes
    initial_land_sizes = land_size_distribution(num_farmers)
    scaling_factor_land = desired_total_land / initial_land_sizes.sum()
    scaled_initial_land_sizes = initial_land_sizes * scaling_factor_land

    # Define distribution functions
    wealth_dist_func = lambda size: scaled_initial_wealths[:size]
    land_dist_func = lambda size: scaled_initial_land_sizes[:size]

    model = FarmingModel(
        num_farmers,
        num_seasons,
        forecast_accuracy,
        subsidy_level,
        wealth_distribution=wealth_dist_func,
        land_size_distribution=land_dist_func
    )

    wealths, ginis = [], []
    for season in range(num_seasons):
        model.step()
        wealths.append(np.mean([a.wealth for a in model.schedule.agents if isinstance(a, Farmer)]))
        ginis.append(gini([a.wealth for a in model.schedule.agents if isinstance(a, Farmer)]))

        # Introduce a crisis at the specified time
        if season == crisis_time:
            for agent in model.schedule.agents:
                if isinstance(agent, Farmer):
                    agent.wealth *= 0.75  # Halve their wealth as a simulated shock

    results_crisis = pd.DataFrame({
        "Season": range(1, num_seasons + 1),
        "Average Wealth": wealths,
        "Gini Coefficient": ginis
    })
    results_crisis.to_csv("results/analysis/simulate_economic_crisis.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(wealths, color=palette[0], label="Average Wealth")  # Use consistent color
    ax1.set_ylabel("Average Wealth", color=palette[0])
    ax1.axvline(x=crisis_time, color='red', linestyle='--', label="Crisis Event")

    ax2 = ax1.twinx()
    ax2.plot(ginis, color=palette[1], label="Gini Coefficient")  # Use consistent color
    ax2.set_ylabel("Gini Coefficient", color=palette[1])

    fig.suptitle("Wealth and Gini Dynamics During Crisis")
    ax1.set_xlabel("Seasons")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("results/crisis_scenario_wealth_gini.png")
    plt.close()

def analyze_crop_yields(num_farmers, num_seasons, forecast_accuracy, subsidy_level):
    # Define desired total initial wealth and land
    desired_total_wealth = num_farmers * 1500
    desired_total_land = num_farmers * 3

    # Define distributions
    wealth_distribution = lambda size: (np.random.pareto(a=1.5, size=size) + 1) * 1500
    land_size_distribution = lambda size: (np.random.pareto(a=2.5, size=size) + 1) * 3

    # Scale wealth
    initial_wealths = wealth_distribution(num_farmers)
    scaling_factor_wealth = desired_total_wealth / initial_wealths.sum()
    scaled_initial_wealths = initial_wealths * scaling_factor_wealth

    # Scale land sizes
    initial_land_sizes = land_size_distribution(num_farmers)
    scaling_factor_land = desired_total_land / initial_land_sizes.sum()
    scaled_initial_land_sizes = initial_land_sizes * scaling_factor_land

    # Define distribution functions
    wealth_dist_func = lambda size: scaled_initial_wealths[:size]
    land_dist_func = lambda size: scaled_initial_land_sizes[:size]

    model = FarmingModel(
        num_farmers,
        num_seasons,
        forecast_accuracy,
        subsidy_level,
        wealth_distribution=wealth_dist_func,
        land_size_distribution=land_dist_func
    )
    rice_yields, wheat_yields = [], []

    for _ in range(num_seasons):
        model.step()
        rice_yield = np.mean([model.forecaster.get_yield("rice", {"rainfall": 60, "accuracy": forecast_accuracy})
                              for _ in model.schedule.agents if isinstance(_, Farmer)])
        wheat_yield = np.mean([model.forecaster.get_yield("wheat", {"rainfall": 40, "accuracy": forecast_accuracy})
                               for _ in model.schedule.agents if isinstance(_, Farmer)])
        rice_yields.append(rice_yield)
        wheat_yields.append(wheat_yield)

    rice_yields_df = pd.DataFrame({"Season": range(1, num_seasons+1), "Rice Yield": rice_yields, "Wheat Yield": wheat_yields})
    rice_yields_df.to_csv("results/analysis/analyze_crop_yields.csv", index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(rice_yields, label="Rice Yield")
    plt.plot(wheat_yields, label="Wheat Yield")
    plt.title("Crop Yields Over Time")
    plt.xlabel("Seasons")
    plt.ylabel("Average Yield")
    plt.legend()
    plt.savefig("results/crop_yield_dynamics.png")
    plt.close()

def analyze_farmer_crop_choices(num_farmers, num_seasons, forecast_accuracy, subsidy_level):
    # Define desired total initial wealth and land
    desired_total_wealth = num_farmers * 1500
    desired_total_land = num_farmers * 3

    # Define distributions
    wealth_distribution = lambda size: (np.random.pareto(a=1.5, size=size) + 1) * 1500
    land_size_distribution = lambda size: (np.random.pareto(a=2.5, size=size) + 1) * 3

    # Scale wealth
    initial_wealths = wealth_distribution(num_farmers)
    scaling_factor_wealth = desired_total_wealth / initial_wealths.sum()
    scaled_initial_wealths = initial_wealths * scaling_factor_wealth

    # Scale land sizes
    initial_land_sizes = land_size_distribution(num_farmers)
    scaling_factor_land = desired_total_land / initial_land_sizes.sum()
    scaled_initial_land_sizes = initial_land_sizes * scaling_factor_land

    # Define distribution functions
    wealth_dist_func = lambda size: scaled_initial_wealths[:size]
    land_dist_func = lambda size: scaled_initial_land_sizes[:size]

    # Corrected variable name from 'land_size_func' to 'land_dist_func'
    model = FarmingModel(
        num_farmers,
        num_seasons,
        forecast_accuracy,
        subsidy_level,
        wealth_distribution=wealth_dist_func,
        land_size_distribution=land_dist_func  # Changed here
    )
    crop_counts = {"rice": [], "wheat": []}

    for _ in range(num_seasons):
        model.step()
        rice_count = sum(1 for a in model.schedule.agents if isinstance(a, Farmer) and a.crop_type == "rice")
        wheat_count = sum(1 for a in model.schedule.agents if isinstance(a, Farmer) and a.crop_type == "wheat")
        crop_counts["rice"].append(rice_count)
        crop_counts["wheat"].append(wheat_count)

    crop_choices_df = pd.DataFrame(crop_counts)
    crop_choices_df.to_csv("results/analysis/analyze_farmer_crop_choices.csv", index=False)

    # Plotting Crop Choices with labels to fix legend error
    plt.figure(figsize=(12, 6))
    plt.plot(crop_counts["rice"], label="Rice", color="brown")
    plt.plot(crop_counts["wheat"], label="Wheat", color="gold")
    plt.title("Crop Choice Distribution Over Time")
    plt.xlabel("Seasons")
    plt.ylabel("Number of Farmers")
    plt.legend()  # Ensure labels are present for the legend
    plt.tight_layout()
    plt.savefig("results/crop_choice_distribution.png")
    plt.close()

def analyze_forecast_impact_on_inequality(num_farmers, num_seasons, accuracy_levels, subsidy_level):
    # Define desired total initial wealth and land
    desired_total_wealth = num_farmers * 1500
    desired_total_land = num_farmers * 3

    fig, ax1 = plt.subplots(figsize=(12, 6))
    results = []

    for accuracy in accuracy_levels:
        # Define distributions
        wealth_distribution = lambda size: (np.random.pareto(a=1.5, size=size) + 1) * 1500
        land_size_distribution = lambda size: (np.random.pareto(a=2.5, size=size) + 1) * 3

        # Scale wealth
        initial_wealths = wealth_distribution(num_farmers)
        scaling_factor_wealth = desired_total_wealth / initial_wealths.sum()
        scaled_initial_wealths = initial_wealths * scaling_factor_wealth

        # Scale land sizes
        initial_land_sizes = land_size_distribution(num_farmers)
        scaling_factor_land = desired_total_land / initial_land_sizes.sum()
        scaled_initial_land_sizes = initial_land_sizes * scaling_factor_land

        # Define distribution functions
        wealth_dist_func = lambda size: scaled_initial_wealths[:size]
        land_dist_func = lambda size: scaled_initial_land_sizes[:size]

        model = FarmingModel(
            num_farmers,
            num_seasons,
            accuracy,
            subsidy_level,
            wealth_distribution=wealth_dist_func,
            land_size_distribution=land_dist_func
        )

        gini_values = []
        for season in range(num_seasons):
            model.step()
            gini_coeff = gini([a.wealth for a in model.schedule.agents if isinstance(a, Farmer)])
            gini_values.append(gini_coeff)
            results.append({
                "Season": season + 1,
                "Accuracy Level": accuracy,
                "Gini Coefficient": gini_coeff
            })

        ax1.plot(gini_values, label=f"Accuracy {accuracy}")

    inequality_trends_df = pd.DataFrame(results)
    inequality_trends_df.to_csv("results/analysis/analyze_forecast_impact_on_inequality.csv", index=False)

    ax1.set_xlabel("Seasons")
    ax1.set_ylabel("Gini Coefficient")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    for accuracy in accuracy_levels:
        # Repeat scaling for average wealth
        wealth_distribution = lambda size: (np.random.pareto(a=1.5, size=size) + 1) * 1500
        land_size_distribution = lambda size: (np.random.pareto(a=2.5, size=size) + 1) * 3

        initial_wealths = wealth_distribution(num_farmers)
        scaling_factor_wealth = desired_total_wealth / initial_wealths.sum()
        scaled_initial_wealths = initial_wealths * scaling_factor_wealth

        initial_land_sizes = land_size_distribution(num_farmers)
        scaling_factor_land = desired_total_land / initial_land_sizes.sum()
        scaled_initial_land_sizes = initial_land_sizes * scaling_factor_land

        wealth_dist_func = lambda size: scaled_initial_wealths[:size]
        land_dist_func = lambda size: scaled_initial_land_sizes[:size]

        model = FarmingModel(
            num_farmers,
            num_seasons,
            accuracy,
            subsidy_level,
            wealth_distribution=wealth_dist_func,
            land_size_distribution=land_dist_func
        )

        wealth_over_time = []
        for season in range(num_seasons):
            model.step()
            avg_wealth = np.mean([a.wealth for a in model.schedule.agents if isinstance(a, Farmer)])
            wealth_over_time.append(avg_wealth)

        ax2.plot(wealth_over_time, label=f"Accuracy {accuracy}", linestyle='--')

    ax2.set_ylabel("Average Wealth")
    ax2.legend(loc="upper right")

    plt.title("Inequality Trends with Different Accuracy Levels")
    plt.tight_layout()
    plt.savefig("results/inequality_trends_accuracy.png")
    plt.close()

def analyze_forecast_impact_on_trust(num_farmers, num_seasons, accuracy_levels, subsidy_level):
    # Define desired total initial wealth and land
    desired_total_wealth = num_farmers * 1500
    desired_total_land = num_farmers * 3

    fig, ax1 = plt.subplots(figsize=(12, 6))
    results = []

    for accuracy in accuracy_levels:
        # Define distributions
        wealth_distribution = lambda size: (np.random.pareto(a=1.5, size=size) + 1) * 1500
        land_size_distribution = lambda size: (np.random.pareto(a=2.5, size=size) + 1) * 3

        # Scale wealth
        initial_wealths = wealth_distribution(num_farmers)
        scaling_factor_wealth = desired_total_wealth / initial_wealths.sum()
        scaled_initial_wealths = initial_wealths * scaling_factor_wealth

        # Scale land sizes
        initial_land_sizes = land_size_distribution(num_farmers)
        scaling_factor_land = desired_total_land / initial_land_sizes.sum()
        scaled_initial_land_sizes = initial_land_sizes * scaling_factor_land

        # Define distribution functions
        wealth_dist_func = lambda size: scaled_initial_wealths[:size]
        land_dist_func = lambda size: scaled_initial_land_sizes[:size]

        model = FarmingModel(
            num_farmers,
            num_seasons,
            accuracy,
            subsidy_level,
            wealth_distribution=wealth_dist_func,
            land_size_distribution=land_dist_func
        )

        avg_trusts = []
        for season in range(num_seasons):
            model.step()
            avg_trust = np.mean([a.trust_level for a in model.schedule.agents if isinstance(a, Farmer)])
            avg_trusts.append(avg_trust)
            results.append({
                "Season": season + 1,
                "Accuracy Level": accuracy,
                "Average Trust": avg_trust
            })

        ax1.plot(avg_trusts, label=f"Average Trust (Accuracy {accuracy})")

    trust_trends_df = pd.DataFrame(results)
    trust_trends_df.to_csv("results/analysis/analyze_forecast_impact_on_trust.csv", index=False)

    ax1.set_xlabel("Seasons")
    ax1.set_ylabel("Average Trust")
    ax1.legend(loc="lower right")

    plt.title("Trust Trends with Different Accuracy Levels")
    plt.tight_layout()
    plt.savefig("results/trust_trends_accuracy.png")
    plt.close()

def simulate_forecast_system_failure(num_farmers, num_seasons, forecast_accuracy, subsidy_level, accuracy_crash_start, accuracy_crash_duration, seed=None):
    # Remove hardcoded parameters and use passed values
    # accuracy_crash_start = 10
    # accuracy_crash_duration = 5
    results_crisis = []

    model = FarmingModel(num_farmers, num_seasons, forecast_accuracy, subsidy_level, seed=seed)  # Pass seed

    for season in range(num_seasons):
        if accuracy_crash_start <= season < accuracy_crash_start + accuracy_crash_duration:
            model.forecaster.accuracy = 0.5
        elif accuracy_crash_start + accuracy_crash_duration <= season:
            # Gradually restore forecast accuracy over 5 seasons
            if season < accuracy_crash_start + accuracy_crash_duration + 5:
                model.forecaster.accuracy = 0.5 + 0.06 * (season - accuracy_crash_start - accuracy_crash_duration)
            else:
                model.forecaster.accuracy = 0.8  # Fully restored forecast accuracy
        model.step()

        results_crisis.append({
            "Season": season + 1,
            "Average Yield": np.mean([a.total_yield for a in model.schedule.agents if isinstance(a, Farmer)]),
            "Average Trust": np.mean([a.trust_level for a in model.schedule.agents if isinstance(a, Farmer)]),
            # Removed "Trust Variance" from results_crisis
        })

    results_crisis = pd.DataFrame(results_crisis)
    results_crisis.to_csv("results/analysis/simulate_forecast_system_failure.csv", index=False)

    # Main Graph without Trust Variance
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    sns.lineplot(data=results_crisis, x="Season", y="Average Yield", ax=ax1, label="Average Yield")
    sns.lineplot(data=results_crisis, x="Season", y="Average Trust", ax=ax2, label="Average Trust", color="green")

    ax1.axvline(accuracy_crash_start, color="red", linestyle="--", label="Crisis Start")
    ax1.axvline(accuracy_crash_start + accuracy_crash_duration, color="red", linestyle="--", label="Crisis End")

    ax1.set_ylabel("Average Yield")
    ax2.set_ylabel("Average Trust")
    ax2.set_ylim(0, 1)  # Set y-axis for Average Trust to range from 0 to 1
    ax1.set_xlabel("Season")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.title("Crisis Scenario: Forecast Accuracy Crash")
    plt.tight_layout()
    plt.savefig("results/crisis_forecasting_crash_yield.png")
    plt.close()

    # Removed Baby Graph for Trust Variance

def analyze_trust_recovery(num_farmers, num_seasons, forecast_accuracy, subsidy_level, accuracy_crash_start, accuracy_crash_duration, government_intervention_duration, seed=None):
    results_recovery = []

    model = FarmingModel(num_farmers, num_seasons, forecast_accuracy, subsidy_level, seed=seed)  # Pass seed

    for season in range(num_seasons):
        if accuracy_crash_start <= season < accuracy_crash_start + accuracy_crash_duration:
            model.forecaster.accuracy = 0.5
        elif accuracy_crash_start + accuracy_crash_duration <= season < accuracy_crash_start + accuracy_crash_duration + government_intervention_duration:
            for agent in model.schedule.agents:
                if isinstance(agent, Farmer):
                    agent.trust_level += 0.3  # Increase trust during government intervention
        elif accuracy_crash_start + accuracy_crash_duration <= season:
            # Gradually restore forecast accuracy over 5 seasons
            if season < accuracy_crash_start + accuracy_crash_duration + 5:
                model.forecaster.accuracy = 0.5 + 0.06 * (season - accuracy_crash_start - accuracy_crash_duration)
            else:
                model.forecaster.accuracy = 0.8  # Fully restored forecast accuracy
        model.step()

        results_recovery.append({
            "Season": season + 1,
            "Average Yield": np.mean([a.total_yield for a in model.schedule.agents if isinstance(a, Farmer)]),
            "Average Trust": np.mean([a.trust_level for a in model.schedule.agents if isinstance(a, Farmer)]),
            "Trust Variance": np.var([a.trust_level for a in model.schedule.agents if isinstance(a, Farmer)]),
        })

    results_recovery = pd.DataFrame(results_recovery)
    results_recovery.to_csv("results/analysis/analyze_trust_recovery.csv", index=False)

    # Visualization with Trust Variance and Government Intervention shading
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    sns.lineplot(data=results_recovery, x="Season", y="Average Yield", ax=ax1, color="blue", label="Average Yield")
    sns.lineplot(data=results_recovery, x="Season", y="Average Trust", ax=ax2, color="green", label="Average Trust")

    # Add vertical lines for Crisis Start and Crisis End
    ax1.axvline(accuracy_crash_start, color="red", linestyle="--", label="Crisis Start")
    ax1.axvline(accuracy_crash_start + accuracy_crash_duration, color="orange", linestyle="--", label="Crisis End")

    # Shade the government intervention period
    intervention_start = accuracy_crash_start + accuracy_crash_duration
    intervention_end = intervention_start + government_intervention_duration
    ax1.axvspan(intervention_start, intervention_end, color='yellow', alpha=0.3, label="Government Intervention")

    ax1.set_ylabel("Average Yield")
    ax2.set_ylabel("Average Trust")
    ax2.set_ylim(0, 1)  # Set y-axis for Average Trust to range from 0 to 1
    ax1.set_xlabel("Season")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.title("Trust Recovery Post Forecast Accuracy Crash")
    plt.tight_layout()
    plt.savefig("results/trust_recovery_yield.png")
    plt.close()

    # Baby Graph for Trust Variance
    plt.figure(figsize=(4, 3))
    sns.lineplot(data=results_recovery, x="Season", y="Trust Variance", label="Trust Variance")
    plt.axvline(accuracy_crash_start, color="red", linestyle="--", label="Crisis Start")
    plt.axvline(accuracy_crash_start + accuracy_crash_duration, color="orange", linestyle="--", label="Crisis End")
    plt.axvspan(intervention_start, intervention_end, color='yellow', alpha=0.3, label="Government Intervention")
    plt.ylabel("Trust Variance")
    plt.xlabel("Season")
    plt.title("Trust Variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/trust_recovery_trust_variance.png")
    plt.close()

def analyze_accuracy_impact_on_yields(num_farmers, num_seasons, accuracy_levels, subsidy_level):
    """
    Tracks average yield across seasons for two different forecast accuracy levels.
    """
    results = []

    for accuracy in accuracy_levels:
        model = FarmingModel(num_farmers, num_seasons, forecast_accuracy=accuracy, subsidy_level=subsidy_level)

        for season in range(num_seasons):
            model.step()

            avg_yield = np.mean([a.total_yield for a in model.schedule.agents if isinstance(a, Farmer)])
            results.append({"Season": season + 1, "Accuracy Level": accuracy, "Average Yield": avg_yield})

    results_df = pd.DataFrame(results)
    results_df.to_csv("results/analysis/analyze_accuracy_impact_on_yields.csv", index=False)

    # Visualization
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_df, x="Season", y="Average Yield", hue="Accuracy Level", palette="Set1")
    plt.xlabel("Season")
    plt.ylabel("Average Yield")
    plt.title("Yield by Accuracy Level")  # Shortened title
    plt.tight_layout()
    plt.savefig("results/yield_tracking_accuracy.png")
    plt.close()

def analyze_forecast_accuracy_improvement(num_farmers, num_seasons, initial_accuracy, subsidy_level, dissemination_modes):
    """
    Analyzes the impact of peer influence on trust levels, wealth, and inequality.
    
    Args:
    - initial_accuracy (float): Starting forecast accuracy.
    """
    results_peer = []

    model = FarmingModel(num_farmers, num_seasons, initial_accuracy, subsidy_level, dissemination_modes=dissemination_modes)
    # Enable peer influence in the model
    for farmer in model.schedule.agents:
        if isinstance(farmer, Farmer):
            farmer.enable_peer_influence = True  # Set to True or False as needed
        
    for season in range(num_seasons):
        model.step()
        
        avg_wealth = np.mean([a.wealth for a in model.schedule.agents if isinstance(a, Farmer)])
        avg_trust = np.mean([a.trust_level for a in model.schedule.agents if isinstance(a, Farmer)])
        gini_coeff = gini([a.wealth for a in model.schedule.agents if isinstance(a, Farmer)])
        
        results_peer.append({
            "Season": season + 1,
            "Average Wealth": avg_wealth,
            "Average Trust": avg_trust,
            "Gini Coefficient": gini_coeff,
        })

    results_peer = pd.DataFrame(results_peer)
    results_peer.to_csv("results/analysis/analyze_forecast_accuracy_improvement.csv", index=False)
    
    # Visualization - Fixed by adding label and removing palette
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=results_peer, 
        x="Season", 
        y="Average Trust", 
        label="Average Trust"  # Added label
    )
    plt.title("Average Trust Over Seasons with Improving Forecast Accuracy")
    plt.xlabel("Season")
    plt.ylabel("Average Trust")
    plt.legend()  # Changed from plt.legend(title="Peer Influence")
    plt.tight_layout()
    plt.savefig("results/forecast_accuracy_improvement.png")
    plt.close()

def apply_subsidy_allocation(model, strategy):
    """
    Applies the specified subsidy allocation strategy to farmers.
    
    Args:
    - strategy (str): 'equal', 'targeted_large', or 'targeted_trust'.
    """
    if strategy == "equal":
        for farmer in model.schedule.agents:
            if isinstance(farmer, Farmer):
                farmer.wealth += model.subsidy_level
    elif strategy == "targeted_large":
        for farmer in model.schedule.agents:
            if isinstance(farmer, Farmer) and farmer.type == "Large":
                farmer.wealth += model.subsidy_level
    elif strategy == "targeted_trust":
        for farmer in model.schedule.agents:
            if isinstance(farmer, Farmer) and farmer.trust_level > 0.7:
                farmer.wealth += model.subsidy_level
    else:
        raise ValueError("Unknown subsidy allocation strategy.")

def analyze_subsidy_allocation(num_farmers, num_seasons, forecast_accuracy, subsidy_level, allocation_strategy, dissemination_modes):
    """
    Analyzes the impact of different subsidy allocation strategies on wealth, trust, and inequality.
    
    Args:
    - allocation_strategy (str): Strategy for allocating subsidies ('equal', 'targeted_large', 'targeted_trust').
    """
    results_subsidy = []

    model = FarmingModel(num_farmers, num_seasons, forecast_accuracy, subsidy_level, dissemination_modes=dissemination_modes)
    
    for season in range(num_seasons):
        model.step()
        
        # Apply subsidy allocation strategy
        if season == 0:
            apply_subsidy_allocation(model, allocation_strategy)
        
        avg_wealth = np.mean([a.wealth for a in model.schedule.agents if isinstance(a, Farmer)])
        avg_trust = np.mean([a.trust_level for a in model.schedule.agents if isinstance(a, Farmer)])
        gini_coeff = gini([a.wealth for a in model.schedule.agents if isinstance(a, Farmer)])
        
        results_subsidy.append({
            "Season": season + 1,
            "Allocation Strategy": allocation_strategy,
            "Average Wealth": avg_wealth,
            "Average Trust": avg_trust,
            "Gini Coefficient": gini_coeff,
        })
    
    results_subsidy = pd.DataFrame(results_subsidy)
    results_subsidy.to_csv("results/analysis/analyze_subsidy_allocation.csv", index=False)
    
    # Visualization
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_subsidy, x="Season", y="Average Wealth", hue="Allocation Strategy", palette="Set1")
    plt.title("Average Wealth Over Seasons by Subsidy Allocation Strategy")
    plt.xlabel("Season")
    plt.ylabel("Average Wealth")
    plt.legend(title="Allocation Strategy")
    plt.tight_layout()
    plt.savefig("results/analysis/subsidy_allocation_wealth.png")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_subsidy, x="Season", y="Gini Coefficient", hue="Allocation Strategy", palette="Set1")
    plt.title("Inequality Over Seasons by Subsidy Allocation Strategy")
    plt.xlabel("Season")
    plt.ylabel("Gini Coefficient")
    plt.legend(title="Allocation Strategy")
    plt.tight_layout()
    plt.savefig("results/analysis/subsidy_allocation_gini.png")
    plt.close()

def analyze_peer_influence_on_trust(num_farmers, num_seasons, forecast_accuracy, subsidy_level, dissemination_modes, peer_influence=True):
    """
    Analyzes the impact of peer influence on trust levels, wealth, and inequality.
    
    Args:
    - peer_influence (bool): Whether peer influence is enabled.
    """
    results_peer = []

    model = FarmingModel(num_farmers, num_seasons, forecast_accuracy, subsidy_level, dissemination_modes=dissemination_modes)
    # Enable peer influence in the model
    for farmer in model.schedule.agents:
        if isinstance(farmer, Farmer):
            farmer.enable_peer_influence = peer_influence  # Assume Farmer agent can handle this attribute
    
    for season in range(num_seasons):
        model.step()
        
        avg_wealth = np.mean([a.wealth for a in model.schedule.agents if isinstance(a, Farmer)])
        avg_trust = np.mean([a.trust_level for a in model.schedule.agents if isinstance(a, Farmer)])
        gini_coeff = gini([a.wealth for a in model.schedule.agents if isinstance(a, Farmer)])
        
        results_peer.append({
            "Season": season + 1,
            "Peer Influence": peer_influence,
            "Average Wealth": avg_wealth,
            "Average Trust": avg_trust,
            "Gini Coefficient": gini_coeff,
        })
    
    results_peer = pd.DataFrame(results_peer)
    results_peer.to_csv("results/analysis/analyze_peer_influence_on_trust.csv", index=False)
    
    # Visualization
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_peer, x="Season", y="Average Trust", hue="Peer Influence", palette="Set1")
    plt.title("Average Trust Over Seasons by Peer Influence")
    plt.xlabel("Season")
    plt.ylabel("Average Trust")
    plt.legend(title="Peer Influence")
    plt.tight_layout()
    plt.savefig("results/peer_influence_trust.png")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_peer, x="Season", y="Average Wealth", hue="Peer Influence", palette="Set1")
    plt.title("Average Wealth Over Seasons by Peer Influence")
    plt.xlabel("Season")
    plt.ylabel("Average Wealth")
    plt.legend(title="Peer Influence")
    plt.tight_layout()
    plt.savefig("results/peer_influence_wealth.png")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_peer, x="Season", y="Gini Coefficient", hue="Peer Influence", palette="Set1")
    plt.title("Inequality Over Seasons by Peer Influence")
    plt.xlabel("Season")
    plt.ylabel("Gini Coefficient")
    plt.legend(title="Peer Influence")
    plt.tight_layout()
    plt.savefig("results/peer_influence_gini.png")
    plt.close()

def run_comparable_simulations():
    common_seed = 42  # Define a common seed for reproducibility
    
    # Define simulation parameters
    num_farmers = 50
    num_seasons = 40
    forecast_accuracy = 0.8
    subsidy_level = 100
    accuracy_crash_start = 10
    accuracy_crash_duration = 5
    crisis_season = 10
    government_intervention_duration = 5
    
    # Run simulate_forecast_system_failure
    simulate_forecast_system_failure(
        num_farmers=num_farmers,
        num_seasons=num_seasons,
        forecast_accuracy=forecast_accuracy,
        subsidy_level=subsidy_level,
        accuracy_crash_start=accuracy_crash_start,
        accuracy_crash_duration=accuracy_crash_duration,
        seed=common_seed  # Use common seed
    )
    
    # Run analyze_trust_recovery
    analyze_trust_recovery(
        num_farmers=num_farmers,
        num_seasons=num_seasons,
        forecast_accuracy=forecast_accuracy,
        subsidy_level=subsidy_level,
        accuracy_crash_start=accuracy_crash_start,
        accuracy_crash_duration=accuracy_crash_duration,
        government_intervention_duration=government_intervention_duration,
        seed=common_seed  # Use same seed for comparability
    )

def analyze_forecast_impact(num_farmers, num_seasons, forecast_accuracies, subsidy_level):
    """
    Runs simulations for different forecast accuracies and collects Gini coefficients and yields.
    
    Args:
    - forecast_accuracies (list of float): List of forecast accuracy levels to test.
    """
    results = []
    
    for accuracy in forecast_accuracies:
        model = FarmingModel(
            num_farmers,
            num_seasons,
            forecast_accuracy=accuracy,
            subsidy_level=subsidy_level
        )
        
        gini_values = []
        average_yields = []
        
        for season in range(num_seasons):
            model.step()
            gini_val = gini([a.wealth for a in model.schedule.agents if isinstance(a, Farmer)])
            avg_yield = np.mean([a.total_yield for a in model.schedule.agents if isinstance(a, Farmer)])
            gini_values.append(gini_val)
            average_yields.append(avg_yield)
            
            results.append({
                "Forecast Accuracy": accuracy,
                "Season": season + 1,
                "Gini Coefficient": gini_val,
                "Average Yield": avg_yield
            })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv("results/analysis/analyze_forecast_accuracy_relation.csv", index=False)
    
    # Plotting the results
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=results_df, x="Season", y="Gini Coefficient", hue="Forecast Accuracy", marker='o')
    plt.title("Gini Coefficient Over Seasons by Forecast Accuracy")
    plt.xlabel("Season")
    plt.ylabel("Gini Coefficient")
    plt.legend(title="Forecast Accuracy")
    plt.tight_layout()
    plt.savefig("results/analysis/gini_by_forecast_accuracy.png")
    plt.close()
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=results_df, x="Season", y="Average Yield", hue="Forecast Accuracy", marker='o')
    plt.title("Average Yield Over Seasons by Forecast Accuracy")
    plt.xlabel("Season")
    plt.ylabel("Average Yield")
    plt.legend(title="Forecast Accuracy")
    plt.tight_layout()
    plt.savefig("results/analysis/yield_by_forecast_accuracy.png")
    plt.close()