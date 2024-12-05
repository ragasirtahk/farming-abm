import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model import FarmingModel
from agents import Farmer

# Ensure the results directory exists
os.makedirs("results/summary_stats", exist_ok=True)

# Define a consistent color palette
palette = sns.color_palette("Set2")

def plot_initial_wealth_distribution(model):
    """
    Plots the distribution of initial wealth among farmers.
    """
    initial_wealths = [agent.wealth for agent in model.schedule.agents if isinstance(agent, Farmer)]
    
    plt.figure(figsize=(10, 6))
    sns.histplot(initial_wealths, kde=True, color=palette[0], bins=20)
    plt.title("Initial Wealth Distribution")
    plt.xlabel("Wealth")
    plt.ylabel("Number of Farmers")
    plt.tight_layout()
    plt.savefig("results/summary_stats/initial_wealth_distribution.png")
    plt.close()

def plot_land_size_distribution(model):
    """
    Plots the distribution of land sizes among farmers.
    """
    land_sizes = [agent.land_size for agent in model.schedule.agents if isinstance(agent, Farmer)]
    
    plt.figure(figsize=(10, 6))
    sns.histplot(land_sizes, kde=True, color=palette[1], bins=20)
    plt.title("Land Size Distribution")
    plt.xlabel("Land Size (Hectares)")
    plt.ylabel("Number of Farmers")
    plt.tight_layout()
    plt.savefig("results/summary_stats/land_size_distribution.png")
    plt.close()

def plot_initial_trust_distribution(model):
    """
    Plots the distribution of initial trust levels among farmers.
    """
    trust_levels = [agent.trust_level for agent in model.schedule.agents if isinstance(agent, Farmer)]
    
    plt.figure(figsize=(10, 6))
    sns.histplot(trust_levels, kde=True, color=palette[2], bins=20)
    plt.title("Initial Trust Level Distribution")
    plt.xlabel("Trust Level")
    plt.ylabel("Number of Farmers")
    plt.tight_layout()
    plt.savefig("results/summary_stats/initial_trust_distribution.png")
    plt.close()

def plot_initial_crop_distribution(model):
    """
    Plots the initial distribution of crop types chosen by farmers.
    """
    crop_types = [agent.crop_type for agent in model.schedule.agents if isinstance(agent, Farmer)]
    crop_counts = pd.Series(crop_types).value_counts()
    
    plt.figure(figsize=(8, 8))
    plt.pie(crop_counts.values, labels=crop_counts.index, colors=palette[:len(crop_counts)], autopct='%1.1f%%', startangle=140)
    plt.title("Initial Crop Type Distribution")
    plt.tight_layout()
    plt.savefig("results/summary_stats/initial_crop_distribution.png")
    plt.close()

def plot_dissemination_mode_distribution(model):
    """
    Plots the initial distribution of dissemination modes among farmers using a pie chart.
    """
    dissemination_modes = [agent.dissemination_mode for agent in model.schedule.agents if isinstance(agent, Farmer)]
    mode_counts = pd.Series(dissemination_modes).value_counts()
    
    plt.figure(figsize=(8, 8))
    plt.pie(
        mode_counts.values, 
        labels=mode_counts.index, 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=palette[:len(mode_counts)]
    )
    plt.title("Initial Dissemination Mode Distribution")
    plt.tight_layout()
    plt.savefig("results/summary_stats/dissemination_mode_distribution.png")
    plt.close()

def plot_land_size_vs_wealth_distribution(model):
    """
    Plots a scatter plot showing the relationship between land size and initial wealth.
    """
    land_sizes = [agent.land_size for agent in model.schedule.agents if isinstance(agent, Farmer)]
    initial_wealths = [agent.wealth for agent in model.schedule.agents if isinstance(agent, Farmer)]
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=land_sizes, y=initial_wealths, hue=[agent.type for agent in model.schedule.agents if isinstance(agent, Farmer)], palette="Set1")
    plt.title("Land Size vs. Initial Wealth")
    plt.xlabel("Land Size (Hectares)")
    plt.ylabel("Initial Wealth")
    plt.legend(title="Farmer Type")
    plt.tight_layout()
    plt.savefig("results/summary_stats/land_size_vs_wealth.png")
    plt.close()

def plot_wealth_boxplot_by_type(model):
    """
    Plots a boxplot comparing wealth distributions between Small and Large farmers.
    """
    farmer_types = [agent.type for agent in model.schedule.agents if isinstance(agent, Farmer)]
    wealths = [agent.wealth for agent in model.schedule.agents if isinstance(agent, Farmer)]
    
    data = pd.DataFrame({
        "Farmer Type": farmer_types,
        "Wealth": wealths
    })
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="Farmer Type", y="Wealth", data=data, hue="Farmer Type", palette="Set2")
    plt.legend().remove()
    plt.title("Wealth Distribution by Farmer Type")
    plt.xlabel("Farmer Type")
    plt.ylabel("Wealth")
    plt.tight_layout()
    plt.savefig("results/summary_stats/wealth_boxplot_by_type.png")
    plt.close()

def plot_correlation_heatmap(model):
    """
    Plots a heatmap showing correlations between initial variables.
    """
    initial_wealths = [agent.wealth for agent in model.schedule.agents if isinstance(agent, Farmer)]
    land_sizes = [agent.land_size for agent in model.schedule.agents if isinstance(agent, Farmer)]
    trust_levels = [agent.trust_level for agent in model.schedule.agents if isinstance(agent, Farmer)]
    
    data = pd.DataFrame({
        "Wealth": initial_wealths,
        "Land Size": land_sizes,
        "Trust Level": trust_levels
    })
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap of Initial Variables")
    plt.tight_layout()
    plt.savefig("results/summary_stats/correlation_heatmap.png")
    plt.close()

def export_summary_statistics(model):
    """
    Exports summary statistics to a text file.
    """
    initial_wealths = [agent.wealth for agent in model.schedule.agents if isinstance(agent, Farmer)]
    land_sizes = [agent.land_size for agent in model.schedule.agents if isinstance(agent, Farmer)]
    trust_levels = [agent.trust_level for agent in model.schedule.agents if isinstance(agent, Farmer)]
    crop_types = [agent.crop_type for agent in model.schedule.agents if isinstance(agent, Farmer)]
    dissemination_modes = [agent.dissemination_mode for agent in model.schedule.agents if isinstance(agent, Farmer)]

    with open("results/summary_stats/summary_statistics.txt", "w") as file:
        file.write("Summary Statistics\n")
        file.write("==================\n")
        file.write(f"Number of Farmers: {len(initial_wealths)}\n")
        file.write(f"Average Initial Wealth: {np.mean(initial_wealths):.2f}\n")
        file.write(f"Average Land Size: {np.mean(land_sizes):.2f} hectares\n")
        file.write(f"Average Trust Level: {np.mean(trust_levels):.2f}\n")
        file.write(f"Crop Type Distribution: {pd.Series(crop_types).value_counts().to_dict()}\n")
        file.write(f"Dissemination Mode Distribution: {pd.Series(dissemination_modes).value_counts().to_dict()}\n")

def generate_summary_statistics():
    """
    Initializes the model and generates all summary statistics visualizations.
    """
    # Initialize the model with initial parameters
    model = FarmingModel(
        num_farmers=50,
        num_seasons=0,  # No simulation steps needed for initial stats
        forecast_accuracy=0.8,
        subsidy_level=10,
        dissemination_modes=[
            ("Radio", 0.50),
            ("Mobile App", 0.20),
            ("Extension Officer", 0.20),
            ("Community Leaders", 0.10)
        ]
    )
    
    # Generate plots
    plot_initial_wealth_distribution(model)
    plot_land_size_distribution(model)
    plot_initial_trust_distribution(model)
    plot_initial_crop_distribution(model)
    plot_dissemination_mode_distribution(model)
    plot_land_size_vs_wealth_distribution(model)
    plot_wealth_boxplot_by_type(model)
    plot_correlation_heatmap(model)
    export_summary_statistics(model)
    
    print("Summary statistics completed.")

if __name__ == "__main__":
    generate_summary_statistics()