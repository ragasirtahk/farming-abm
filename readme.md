This repository contains a farming agent-based model (ABM) implemented in Python using the Mesa framework. The model simulates the interactions between farmers and forecasters in an agricultural setting, exploring the impact of forecast accuracy, dissemination modes, subsidy allocation, and other factors on farmers' wealth, trust levels, crop choices, and inequality.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Results and Visualization](#results-and-visualization)
- [Contributing](#contributing)
- [License](#license)

## Overview

The farming ABM simulates a population of farmers who make crop decisions based on forecasts provided by a forecaster agent. Farmers' wealth, trust levels, and crop choices evolve over time, influenced by factors such as forecast accuracy, subsidy levels, dissemination modes (e.g., Radio, Mobile App, Extension Officer, Community Leaders), and peer influence.

The model is designed to analyze various scenarios, including:

- The impact of different dissemination modes on farmers' trust and wealth.
- The effects of forecast accuracy on inequality and yields.
- Responses to economic crises or forecast system failures.
- Strategies for subsidy allocation and their effects on inequality.

## Features

- **Agent-Based Modeling**: Simulation of individual farmers with distinct attributes and behaviors.
- **Forecasting Mechanism**: A forecaster agent provides weather forecasts with adjustable accuracy levels.
- **Dissemination Modes**: Exploration of different dissemination methods and their influence on trust and decision-making.
- **Economic Analysis**: Calculation of Gini coefficients and average wealth to study inequality.
- **Statistical Analysis**: ANOVA and Tukey HSD tests to analyze the effects of different factors.
- **Visualization**: Generating plots and flowcharts to visualize results and model structure.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/ragasirtahk/farming-abm.git
   cd farming-abm
   ```

2. **Create a virtual environment (optional but recommended)**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages**:

   ```bash
   pip install -r requirements.txt
   ```

   If `requirements.txt` is not provided, install the packages manually:

   ```bash
   pip install numpy pandas matplotlib seaborn mesa statsmodels graphviz
   ```

4. **Install Graphviz** (Only needed to generate the flowcharts):

   The `graphviz` Python package is an interface to the Graphviz software. You need to install Graphviz separately:

   - On Ubuntu/Debian:

     ```bash
     sudo apt-get install graphviz
     ```

   - On MacOS (using Homebrew):

     ```bash
     brew install graphviz
     ```

   - On Windows:

     Download and install from [Graphviz website](https://graphviz.org/download/).

## Usage

1. **Generate Summary Statistics**:

   Run the `summary_stats.py` script to generate initial summary statistics and visualizations:

   ```bash
   python summary_stats.py
   ```

   Outputs will be saved in the `results/summary_stats` directory.

2. **Run Main Simulations**:

   Run the `main.py` script to execute the simulations and experiments:

   ```bash
   python main.py
   ```

   Outputs will be saved in the `results` directory.

3. **Generate Flowcharts**:

   Use the `flowcharts/flowchart_generator.py` script to generate flowcharts illustrating the model's processes:

   ```bash
   python flowcharts/flowchart_generator.py
   ```

   Flowcharts will be generated in the `flowcharts` directory.

4. **Perform Statistical Analysis**:

   Analyze the results using the `statistical_analysis.py` script:

   ```bash
   python statistical_analysis.py
   ```

   Analysis outputs will be saved in the `results/analysis` directory.

## Project Structure

```
farming-abm/
├── agents.py
├── experiments.py
├── flowcharts/
│   ├── flowchart_generator.py
│   └── farming_abm_flowchart.gv
├── main.py
├── model.py
├── robustness_tests.py
├── sensitivity_tests.py
├── statistical_analysis.py
├── summary_stats.py
├── utils.py
├── results/
│   ├── analysis/
│   ├── robustness_checks/
│   └── summary_stats/
└── README.md
```

- `agents.py`: Defines the `Farmer` and `Forecaster` agent classes.
- `model.py`: Contains the `FarmingModel` class, defining the overall model.
- `summary_stats.py`: Generates initial summary statistics and visualizations.
- `experiments.py`: Contains functions to run various experiments and scenarios.
- `main.py`: The main script to run simulations and experiments.
- `statistical_analysis.py`: Performs statistical analyses on the simulation results.
- `flowcharts/`: Contains scripts and files for generating flowcharts of the model processes.
- `results/`: Directory where all results, visualizations, and analyses are saved.

## Dependencies

- Python 3.x
- [Mesa](https://mesa.readthedocs.io/en/master/) (Agent-based modeling framework)
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Statsmodels
- Graphviz

Install the dependencies using the provided `requirements.txt` file or manually as described in the [Installation](#installation) section.

## Results and Visualization

The results of the simulations and analyses are saved in the `results` directory, organized into subdirectories:

- `results/summary_stats/`: Initial summary statistics and plots.
- `results/analysis/`: Outputs from experiments and statistical analyses.
- `results/robustness_checks/`: Results from robustness tests.
- `results/flowcharts/`: Generated flowcharts visualizing the model's processes.

Visualizations include plots of wealth distribution, trust dynamics, Gini coefficients over time, and the impact of different dissemination modes or forecast accuracies.

## Contributing

Contributions are welcome! Please:

- Fork the repository.
- Create a new branch for your feature or bug fix.
- Submit a pull request with detailed descriptions of your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

*Note*: Ensure that all the paths and module imports are correctly set up relative to your project structure. Adjust the commands and instructions if necessary based on your specific setup.