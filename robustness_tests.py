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
os.makedirs("results/robustness_checks", exist_ok=True)

# Define a consistent color palette
palette = sns.color_palette("Set2")

def robustness_checks(num_farmers, num_seasons, seeds, param_ranges, dissemination_modes):
    # Remove hardcoded parameters and use passed values
    # num_farmers = 50  # Commented out
    # num_seasons = 20  # Commented out
    # seeds = [42, 99, 123, 2024]  # Commented out
    # param_ranges = { ... }  # Commented out

    results_robustness = []

    for seed in seeds:
        np.random.seed(seed)  # Move seed setting outside inner loops
        for accuracy in param_ranges["Forecast Accuracy"]:
            for subsidy in param_ranges["Subsidy Levels"]:
                for variability in param_ranges["Rainfall Variability"]:
                    model = FarmingModel(num_farmers, num_seasons, accuracy, subsidy, dissemination_modes=dissemination_modes)
                    for _ in range(num_seasons):
                        model.step()

                    final_wealth = np.mean([a.wealth for a in model.schedule.agents if isinstance(a, Farmer)])
                    final_gini = gini([a.wealth for a in model.schedule.agents if isinstance(a, Farmer)])
                    
                    results_robustness.append({
                        "Seed": seed,
                        "Forecast Accuracy": accuracy,
                        "Subsidy": subsidy,
                        "Rainfall Variability": variability,
                        "Final Wealth": final_wealth,
                        "Gini Coefficient": final_gini,
                    })

    results_robustness = pd.DataFrame(results_robustness)
    
    # Removed palette assignment since no hue is used in pairplot
    g = sns.pairplot(
        results_robustness,
        diag_kind="kde",
        vars=["Final Wealth", "Gini Coefficient", "Forecast Accuracy", "Rainfall Variability"],
        corner=True,
        plot_kws={"alpha": 0.6},
        diag_kws={"fill": True}
        # Removed palette=adjusted_palette
    )

    # Explicitly set axis labels for clarity
    for ax in g.axes.flatten():
        if ax:
            ax.set_xlabel(ax.get_xlabel(), fontsize=10)
            ax.set_ylabel(ax.get_ylabel(), fontsize=10)
            ax.tick_params(axis='x', labelsize=8)
            ax.tick_params(axis='y', labelsize=8)

    # Set suptitle correctly
    g.fig.suptitle("Robustness Checks Pairplot", fontsize=16, y=0.98)  # Adjusted y position to ensure visibility
    g.fig.tight_layout()
    plt.savefig("results/robustness_checks/robustness_checks_pairplot.png")
    plt.close()

def robustness_check_time_and_scale(scenarios, forecast_accuracy, subsidy_level):
    num_runs = 10  # Number of runs per scenario
    
    # Define total initial wealth and land
    total_initial_wealth = 100000  # Set a common total initial wealth
    total_initial_land = 200        # Set a common total initial land area
    
    results = []

    for scenario in scenarios:
        avg_wealths = np.zeros(scenario["num_seasons"])
        avg_ginis = np.zeros(scenario["num_seasons"])
        avg_wealths = np.zeros(scenario["num_seasons"])
        avg_ginis = np.zeros(scenario["num_seasons"])

        for _ in range(num_runs):
            model = FarmingModel(
                num_farmers=scenario["num_farmers"],
                num_seasons=scenario["num_seasons"],
                forecast_accuracy=forecast_accuracy,
                subsidy_level=subsidy_level,
                total_initial_wealth=total_initial_wealth,
                total_initial_land=total_initial_land
            )

            wealths = []
            ginis = []

            for season in range(scenario["num_seasons"]):
                model.step()
                wealth = np.mean([
                    a.wealth for a in model.schedule.agents if isinstance(a, Farmer)
                ])
                gini_coeff = gini([
                    a.wealth for a in model.schedule.agents if isinstance(a, Farmer)
                ])
                wealths.append(wealth)
                ginis.append(gini_coeff)

            avg_wealths += np.array(wealths)
            avg_ginis += np.array(ginis)

        avg_wealths /= num_runs
        avg_ginis /= num_runs

        for season in range(scenario["num_seasons"]):
            results.append({
                "Season": season + 1,
                "Scenario": scenario["Label"],
                "Average Wealth": avg_wealths[season],
                "Gini Coefficient": avg_ginis[season],
            })

    results_df = pd.DataFrame(results)
    # Save results to CSV
    results_df.to_csv("results/robustness_checks/time_scale_results.csv", index=False)

    # Plotting the averaged results
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=results_df,
        x="Season",
        y="Average Wealth",
        hue="Scenario"
    )
    plt.title("Average Wealth Over Time for Different Scenarios")
    plt.xlabel("Season")
    plt.ylabel("Average Wealth")
    plt.legend(title="Scenario")
    plt.tight_layout()
    plt.savefig("results/robustness_checks/time_scale_wealth.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=results_df,
        x="Season",
        y="Gini Coefficient",
        hue="Scenario"
    )
    plt.title("Gini Coefficient Over Time for Different Scenarios")
    plt.xlabel("Season")
    plt.ylabel("Gini Coefficient")
    plt.legend(title="Scenario")
    plt.tight_layout()
    plt.savefig("results/robustness_checks/time_scale_gini.png")
    plt.close()

def robustness_check_subsidy_levels(num_farmers, num_seasons, subsidy_levels, forecast_accuracy, dissemination_modes):
    """
    Tests the impact of varying subsidy levels on wealth, trust, and inequality.
    """
    results_subsidy = []

    for subsidy in subsidy_levels:
        model = FarmingModel(num_farmers, num_seasons, forecast_accuracy, subsidy, dissemination_modes=dissemination_modes)
        for season in range(num_seasons):
            model.step()
            avg_wealth = np.mean([a.wealth for a in model.schedule.agents if isinstance(a, Farmer)])
            avg_trust = np.mean([a.trust_level for a in model.schedule.agents if isinstance(a, Farmer)])
            gini_coeff = gini([a.wealth for a in model.schedule.agents if isinstance(a, Farmer)])

            results_subsidy.append({
                "Season": season + 1,
                "Subsidy Level": subsidy,
                "Average Wealth": avg_wealth,
                "Average Trust": avg_trust,
                "Gini Coefficient": gini_coeff,
            })

    results_subsidy_df = pd.DataFrame(results_subsidy)

    # Visualization - Average Wealth by Subsidy Level
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_subsidy_df, x="Season", y="Average Wealth", hue="Subsidy Level", palette="Set1")
    plt.title("Average Wealth Over Seasons by Subsidy Level")
    plt.xlabel("Season")
    plt.ylabel("Average Wealth")
    plt.legend(title="Subsidy Level")
    plt.tight_layout()
    plt.savefig("results/robustness_checks/subsidy_levels_wealth.png")
    plt.close()

    # Visualization - Gini Coefficient by Subsidy Level
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_subsidy_df, x="Season", y="Gini Coefficient", hue="Subsidy Level", palette="Set1")
    plt.title("Inequality Over Seasons by Subsidy Level")
    plt.xlabel("Season")
    plt.ylabel("Gini Coefficient")
    plt.legend(title="Subsidy Level")
    plt.tight_layout()
    plt.savefig("results/robustness_checks/subsidy_levels_gini.png")
    plt.close()

def robustness_check_wealth_land_distributions(num_farmers, num_seasons, wealth_distributions, land_size_distributions, forecast_accuracy, subsidy_level, dissemination_modes):
    """
    Tests the impact of different wealth and land size distributions on model outcomes.
    """
    results_distributions = []

    wealth_distributions = {
        "Normal": lambda size, **kwargs: np.random.normal(loc=1500, scale=300, size=size),
        "Uniform": lambda size, **kwargs: np.random.uniform(low=1000, high=2000, size=size),
        "Log-Normal": lambda size, **kwargs: np.random.lognormal(mean=np.log(1500), sigma=0.5, size=size),
        "Pareto": lambda size, **kwargs: (np.random.pareto(a=1.5, size=size) + 1) * 1000
    }

    land_size_distributions = {
        "Normal": lambda size, **kwargs: np.random.normal(loc=3, scale=0.5, size=size),
        "Uniform": lambda size, **kwargs: np.random.uniform(low=1, high=5, size=size),
        "Log-Normal": lambda size, **kwargs: np.random.lognormal(mean=np.log(3), sigma=0.3, size=size),
        "Pareto": lambda size, **kwargs: (np.random.pareto(a=2.0, size=size) + 1) * 1
    }

    for wealth_dist_name, wealth_dist_func in wealth_distributions.items():
        for land_dist_name, land_dist_func in land_size_distributions.items():
            # Define total_initial_wealth and total_initial_land
            total_initial_wealth = 7500  # Set a common total initial wealth
            total_initial_land = 2000        # Set a common total initial land area

            model = FarmingModel(
                num_farmers,
                num_seasons,
                forecast_accuracy,
                subsidy_level,
                wealth_distribution=wealth_dist_func,
                land_size_distribution=land_dist_func,
                total_initial_land=total_initial_land,
                total_initial_wealth=total_initial_wealth,  # Ensure total_initial_wealth is also passed
                dissemination_modes=dissemination_modes
            )
            
            # **Begin Scaling Validation**
            # Calculate total initial wealth and land from agents
            total_wealth = sum(a.wealth for a in model.schedule.agents if isinstance(a, Farmer))
            total_land = sum(a.land_size for a in model.schedule.agents if isinstance(a, Farmer))

            # Output the totals
            print(f"{wealth_dist_name} - {land_dist_name} Total Initial Wealth: {total_wealth}")
            print(f"{wealth_dist_name} - {land_dist_name} Total Initial Land: {total_land}")

            # Validate scaling
            assert np.isclose(total_wealth, total_initial_wealth), f"Total wealth mismatch for {wealth_dist_name} - {land_dist_name}"
            assert np.isclose(total_land, total_initial_land), f"Total land size mismatch for {wealth_dist_name} - {land_dist_name}"
            # **End Scaling Validation**

            for season in range(num_seasons):
                model.step()
                avg_wealth = np.mean([a.wealth for a in model.schedule.agents if isinstance(a, Farmer)])
                avg_trust = np.mean([a.trust_level for a in model.schedule.agents if isinstance(a, Farmer)])
                gini_coeff = gini([a.wealth for a in model.schedule.agents if isinstance(a, Farmer)])

                results_distributions.append({
                    "Season": season + 1,
                    "Wealth Distribution": wealth_dist_name,
                    "Land Size Distribution": land_dist_name,
                    "Average Wealth": avg_wealth,
                    "Average Trust": avg_trust,
                    "Gini Coefficient": gini_coeff,
                })

    results_distributions_df = pd.DataFrame(results_distributions)

    # Visualization - Average Wealth by Wealth Distribution
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=results_distributions_df, x="Season", y="Average Wealth",
                 hue="Wealth Distribution", palette="Set2")
    plt.title("Average Wealth Over Seasons by Wealth Distribution")
    plt.xlabel("Season")
    plt.ylabel("Average Wealth")
    plt.legend(title="Wealth Distribution", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("results/robustness_checks/distributions_wealth_by_wealth_dist.png")
    plt.close()

    # Visualization - Gini Coefficient by Wealth Distribution
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=results_distributions_df, x="Season", y="Gini Coefficient",
                 hue="Wealth Distribution", palette="Set2")
    plt.title("Inequality Over Seasons by Wealth Distribution")
    plt.xlabel("Season")
    plt.ylabel("Gini Coefficient")
    plt.legend(title="Wealth Distribution", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("results/robustness_checks/distributions_gini_by_wealth_dist.png")
    plt.close()

    # Visualization - Average Wealth by Land Size Distribution
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=results_distributions_df, x="Season", y="Average Wealth",
                 hue="Land Size Distribution", palette="Set3")
    plt.title("Average Wealth Over Seasons by Land Size Distribution")
    plt.xlabel("Season")
    plt.ylabel("Average Wealth")
    plt.legend(title="Land Size Distribution", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("results/robustness_checks/distributions_wealth_by_land_dist.png")
    plt.close()

    # Visualization - Gini Coefficient by Land Size Distribution
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=results_distributions_df, x="Season", y="Gini Coefficient",
                 hue="Land Size Distribution", palette="Set3")
    plt.title("Inequality Over Seasons by Land Size Distribution")
    plt.xlabel("Season")
    plt.ylabel("Gini Coefficient")
    plt.legend(title="Land Size Distribution", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("results/robustness_checks/distributions_gini_by_land_dist.png")
    plt.close()