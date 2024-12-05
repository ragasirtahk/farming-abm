import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model import FarmingModel
from utils import gini
from agents import Farmer

# Ensure the results directory exists
os.makedirs("results", exist_ok=True)

# Define a consistent color palette
palette = sns.color_palette("Set2")

def sensitivity_analysis(num_farmers, num_seasons, forecast_accuracy, subsidy_levels):
    wealth_over_time = {subsidy_level: [] for subsidy_level in subsidy_levels}
    gini_over_time = {subsidy_level: [] for subsidy_level in subsidy_levels}

    for subsidy_level in subsidy_levels:
        model = FarmingModel(num_farmers, num_seasons, forecast_accuracy, subsidy_level)
        for season in range(num_seasons):
            model.step()
            avg_wealth = np.mean([a.wealth for a in model.schedule.agents if isinstance(a, Farmer)])
            wealth_over_time[subsidy_level].append(avg_wealth)
            gini_value = gini([a.wealth for a in model.schedule.agents if isinstance(a, Farmer)])
            gini_over_time[subsidy_level].append(gini_value)

    # Plot wealth trends
    plt.figure(figsize=(12, 6))
    for subsidy_level, wealths in wealth_over_time.items():
        plt.plot(wealths, label=f"Subsidy {subsidy_level}", color=palette[subsidy_level % len(palette)])
    plt.title("Wealth by Subsidy Level")
    plt.xlabel("Seasons")
    plt.ylabel("Average Wealth")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/wealth_sensitivity_subsidy.png")
    plt.close()

    # Plot Gini trends
    plt.figure(figsize=(12, 6))
    for subsidy_level, ginis in gini_over_time.items():
        plt.plot(ginis, label=f"Subsidy {subsidy_level}", color=palette[subsidy_level % len(palette)])
    plt.title("Inequality by Subsidy Level")
    plt.xlabel("Seasons")
    plt.ylabel("Gini Coefficient")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/gini_sensitivity_subsidy.png")
    plt.close()

wealth_distributions = {
    "Normal": lambda size: np.random.normal(loc=1500, scale=300, size=size),
    "Uniform": lambda size: np.random.uniform(low=1000, high=2000, size=size),
    "Log-Normal": lambda size: np.random.lognormal(mean=np.log(1500), sigma=0.5, size=size),
    "Pareto": lambda size: (np.random.pareto(a=1.5, size=size) + 1) * 1000
}

land_size_distributions = {
    "Normal": lambda size: np.random.normal(loc=3, scale=0.5, size=size),
    "Uniform": lambda size: np.random.uniform(low=1, high=5, size=size),
    "Log-Normal": lambda size: np.random.lognormal(mean=np.log(3), sigma=0.3, size=size),
    "Pareto": lambda size: (np.random.pareto(a=2.0, size=size) + 1) * 1
}

def sensitivity_analysis_dist(num_farmers, num_seasons, forecast_accuracy, subsidy_level, distributions, dissemination_modes):
    """
    Runs sensitivity analysis using different parameter distributions.
    """
    results = []
    initial_wealths_dict = {}
    initial_land_sizes_dict = {}  # Added to store initial land sizes
    final_wealths_dict = {}
    final_land_sizes_dict = {}    # Added to store final land sizes

    # Desired total initial wealth and land
    desired_total_wealth = num_farmers * 1500  # Since mean is 1500
    desired_total_land = num_farmers * 3       # Assuming mean land size is 3

    for dist_name, dist_info in distributions.items():
        dist_func = dist_info["func"]
        params = dist_info["params"]

        # Generate initial wealth
        if dist_name in ["Pareto"]:
            initial_wealths = dist_func(num_farmers)
        else:
            initial_wealths = dist_func(size=num_farmers, **params)

        # Calculate current total wealth
        current_total_wealth = np.sum(initial_wealths)

        # Calculate scaling factor for wealth
        scaling_factor_wealth = desired_total_wealth / current_total_wealth

        # Scale initial wealths to match desired total wealth
        scaled_initial_wealths = initial_wealths * scaling_factor_wealth

        initial_wealths_dict[dist_name] = scaled_initial_wealths

        # **Add the following print statement to verify total initial wealth**
        print(f"{dist_name} Total Initial Wealth: {np.sum(scaled_initial_wealths)}")

        # Generate initial land sizes
        initial_land_sizes = land_size_distributions[dist_name](size=num_farmers)  # Ensure consistent distribution naming

        # Calculate current total land
        current_total_land = np.sum(initial_land_sizes)

        # Calculate scaling factor for land
        scaling_factor_land = desired_total_land / current_total_land

        # Scale initial land sizes to match desired total land
        scaled_initial_land_sizes = initial_land_sizes * scaling_factor_land

        initial_land_sizes_dict[dist_name] = scaled_initial_land_sizes

        # **Add the following print statement to verify total initial land**
        print(f"{dist_name} Total Initial Land: {np.sum(scaled_initial_land_sizes)}")

        # Define wealth_distribution with 'size' as the parameter
        wealth_distribution = lambda size: scaled_initial_wealths[:size]

        # Define land_size_distribution with 'size' as the parameter
        land_size_distribution = lambda size: scaled_initial_land_sizes[:size]

        model = FarmingModel(
            num_farmers,
            num_seasons,
            forecast_accuracy,
            subsidy_level,
            wealth_distribution=wealth_distribution,
            land_size_distribution=land_size_distribution,
            dissemination_modes=dissemination_modes
        )

        for season in range(num_seasons):
            model.step()

        final_wealths = [a.wealth for a in model.schedule.agents if isinstance(a, Farmer)]
        final_land_sizes = [a.land_size for a in model.schedule.agents if isinstance(a, Farmer)]  # Track final land sizes

        final_wealths_dict[dist_name] = final_wealths
        final_land_sizes_dict[dist_name] = final_land_sizes

        avg_wealth = np.mean(final_wealths)
        avg_land = np.mean(final_land_sizes)  # Calculate average land size
        gini_coeff = gini(final_wealths)

        results.append({
            "Distribution": dist_name,
            "Average Wealth": avg_wealth,
            "Average Land Size": avg_land,  # Include average land size
            "Gini Coefficient": gini_coeff
        })

    results_df = pd.DataFrame(results)

    # Visualization
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # Updated Seaborn barplot to include hue and avoid FutureWarning
    sns.barplot(
        data=results_df, 
        x="Distribution", 
        y="Average Wealth", 
        hue="Distribution",      # Assign hue to 'Distribution'
        ax=ax1, 
        palette="Set2", 
        alpha=0.8, 
        legend=False            # Disable legend to avoid redundancy
    )
    sns.lineplot(data=results_df, x="Distribution", y="Gini Coefficient", ax=ax2, color="green", marker="o")

    # Increase font size for axis labels
    ax1.set_ylabel("Average Wealth", fontsize=12)
    ax2.set_ylabel("Gini Coefficient", fontsize=12)
    ax1.set_xlabel("Distribution Type", fontsize=12)

    # Increase font size for tick labels
    ax1.tick_params(axis='y', labelsize=10)
    ax2.tick_params(axis='y', labelsize=10)
    ax1.tick_params(axis='x', labelsize=10)
    ax2.tick_params(axis='x', labelsize=10)

    # Update legends with increased font size
    ax1.legend(["Average Wealth"], loc="upper left", fontsize=10)
    ax2.legend(["Gini Coefficient"], loc="upper right", fontsize=10)

    # Set y-axis to logarithmic scale
    ax1.set_yscale('log')  # Retain log scale for better data distribution visualization

    # Add grid lines to ax1 for the y-axis
    ax1.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.7)

    # Remove grid lines from ax2 to avoid clutter
    ax2.grid(False)

    # Increase font size for the plot title
    plt.title("Wealth and Inequality by Distribution", fontsize=14)

    # Adjust layout to accommodate increased font sizes
    plt.tight_layout()

    # Save the updated figure
    plt.savefig("results/sensitivity_analysis_distributions.png")
    plt.ticklabel_format(style='plain', axis='y')  # Disable scientific notation on y-axis
    plt.close()

    # Plot initial and final wealth and land distributions
    fig, axes = plt.subplots(len(distributions), 2, figsize=(14, 6 * len(distributions)))  # Increased width for titles
    for i, (dist_name, initial_wealths) in enumerate(initial_wealths_dict.items()):
        final_wealths = final_wealths_dict[dist_name]
        initial_land_sizes = initial_land_sizes_dict[dist_name]
        final_land_sizes = final_land_sizes_dict[dist_name]

        sns.histplot(initial_wealths, kde=True, ax=axes[i, 0], color=palette[0])
        axes[i, 0].set_title(f"Initial Wealth ({dist_name})")
        axes[i, 0].set_xlabel("Wealth")
        axes[i, 0].set_ylabel("Frequency")

        sns.histplot(final_wealths, kde=True, ax=axes[i, 1], color=palette[1])
        axes[i, 1].set_title(f"Final Wealth ({dist_name})")  # Added title
        axes[i, 1].set_xlabel("Wealth")
        axes[i, 1].set_ylabel("Frequency")

        # Optional: Plot initial and final land sizes
        # You can create additional subplots if needed

    plt.suptitle("Wealth Distributions: Initial vs Final", fontsize=16, y=0.98)  # Adjusted y position to ensure visibility
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Modify rect to allow space for the title
    plt.savefig("results/wealth_distributions_initial_final.png")
    plt.close()