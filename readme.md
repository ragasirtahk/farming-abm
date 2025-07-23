# Farming Agent-Based Model for Development Economics

This repository contains a sophisticated agent-based model (ABM) that simulates the impact of weather forecasting systems on agricultural decision-making and economic outcomes in rural communities. The model examines how forecast accuracy, information dissemination methods, subsidies, and other factors influence farmers' wealth accumulation, trust dynamics, crop choices, and economic inequality.

## Table of Contents

- [Overview](#overview)
- [Model Components](#model-components)
- [Key Research Questions](#key-research-questions)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Results and Visualization](#results-and-visualization)
- [Contributing](#contributing)
- [License](#license)

## Overview

The farming ABM uses the Mesa framework to simulate interactions between heterogeneous farmers and a weather forecasting system. Each farmer agent makes crop decisions (rice or wheat) based on weather forecasts, with outcomes that affect their wealth, subsequent trust in forecasts, and future decision-making. The model tracks economic inequality using Gini coefficients and examines how different policy interventions can improve outcomes across farmer types.

## Model Components

### Agents

#### Farmers
- Categorized as "Small" or "Large" based on land size (≤2.5 hectares = Small)
- Possess varying land sizes and initial wealth (typically following Pareto distributions)
- Receive weather forecasts through different dissemination channels
- Make crop choices (rice or wheat) based on forecast information and risk tolerance
- Update trust in forecasts based on economic outcomes and community influence
- Accumulate wealth through crop yields and government subsidies

#### Forecaster
- Provides weather predictions with configurable accuracy levels (0.0-1.0)
- Delivers forecasts through different dissemination modes (Radio, Mobile App, Extension Officer, Community Leaders)
- Each dissemination mode has different effects on perceived forecast accuracy
- Can experience temporary accuracy drops (simulating forecast system failures)

### Environment
- Simulates agricultural seasons with varying rainfall conditions
- Different crops perform optimally under different rainfall conditions
- Economic performance depends on crop-rainfall match and forecast accuracy

## Key Research Questions

The simulation framework addresses several important questions relevant to development economics:

1. **Forecast Impact**: How do weather forecasts of varying accuracy affect agricultural productivity, economic welfare, and inequality?
2. **Information Dissemination**: Which forecast delivery methods are most effective for different farmer segments?
3. **Trust Dynamics**: How does trust in forecasting systems evolve over time, and what factors influence trust formation?
4. **Resilience**: How do communities recover from forecast system failures or economic shocks?
5. **Policy Effectiveness**: What subsidy allocation strategies best reduce inequality while promoting productivity?
6. **Wealth Distribution**: How do initial wealth and land distributions affect long-term economic outcomes?

## Features

- **Heterogeneous Agents**: Farmers with different land sizes, wealth, risk tolerance, and information access
- **Dynamic Trust Mechanism**: Trust levels evolve based on forecast performance and peer influence
- **Multiple Dissemination Channels**: Four distinct forecast delivery methods with different effectiveness
- **Robust Experimentation Framework**: Parametric sensitivity analysis, robustness checks, and statistical validation
- **Economic Analysis**: Wealth tracking, Gini coefficient calculation, and distribution analysis
- **Crisis Simulation**: Economic shocks and forecast system failures with recovery monitoring
- **Statistical Analysis**: ANOVA, Tukey HSD tests, and correlation analysis
- **Comprehensive Visualization**: Time series plots, distribution comparisons, and boxplots

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/ragasirtahk/farming-abm.git
   cd farming-abm
   ```

2. **Create a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install required packages**:

   ```bash
   pip install numpy pandas matplotlib seaborn mesa statsmodels scipy graphviz
   ```

4. **Install Graphviz** (for flowchart generation):

   - **Ubuntu/Debian**:
     ```bash
     sudo apt-get install graphviz
     ```
   - **MacOS**:
     ```bash
     brew install graphviz
     ```
   - **Windows**: Download from [Graphviz website](https://graphviz.org/download/)

## Usage

### Running Experiments

The main simulation experiments can be run using the following scripts:

1. **Generate summary statistics for the model**:
   ```bash
   python summary_stats.py
   ```

2. **Run the main simulation suite**:
   ```bash
   python main.py
   ```

3. **Perform sensitivity analysis**:
   ```bash
   python sensitivity_tests.py
   ```

4. **Conduct robustness checks**:
   ```bash
   python robustness_tests.py
   ```

5. **Run statistical analysis on results**:
   ```bash
   python statistical_analysis.py
   ```

### Key Experiments

The model includes several specific experiments that can be run individually:

- **Dissemination Impact Analysis**: Compares how different forecast delivery methods affect farmer outcomes
  ```python
  analyze_dissemination_impact(num_farmers=50, num_seasons=40, forecast_accuracy=0.8, subsidy_level=10, dissemination_modes=dissemination_modes)
  ```

- **Economic Crisis Simulation**: Tests community resilience to economic shocks
  ```python
  simulate_economic_crisis(num_farmers=50, num_seasons=40, forecast_accuracy=0.8, subsidy_level=10, crisis_time=20)
  ```

- **Forecast System Failure**: Simulates the impact of a temporary drop in forecast accuracy
  ```python
  simulate_forecast_system_failure(num_farmers=50, num_seasons=40, forecast_accuracy=0.8, subsidy_level=10, accuracy_crash_start=10, accuracy_crash_duration=5)
  ```

- **Trust Recovery Analysis**: Examines how trust recovers after forecast failures
  ```python
  analyze_trust_recovery(num_farmers=50, num_seasons=40, forecast_accuracy=0.8, subsidy_level=10, accuracy_crash_start=10, accuracy_crash_duration=5, government_intervention_duration=3)
  ```

### Customizing Parameters

Key parameters that can be adjusted include:

- `num_farmers`: Population size (default: 50)
- `num_seasons`: Simulation duration (default: 40)
- `forecast_accuracy`: Baseline accuracy of weather forecasts (default: 0.8)
- `subsidy_level`: Government agricultural subsidy amount (default: 10)
- `dissemination_modes`: Distribution of forecast delivery methods among farmers

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

## Project Structure

```
farming-abm/
├── agents.py                # Defines Farmer and Forecaster agent classes
├── experiments.py           # Contains specific experimental scenarios
├── flowcharts/              # Visualization of model logic and processes
│   ├── flowchart_generator.py
│   └── various flowcharts (.pdf, .gv)
├── main.py                  # Main execution script
├── model.py                 # Core FarmingModel implementation
├── robustness_tests.py      # Tests for model stability across parameters
├── sensitivity_tests.py     # Parameter sensitivity analysis
├── statistical_analysis.py  # Statistical tests on simulation results
├── summary_stats.py         # Generates descriptive statistics
├── utils.py                 # Utility functions (e.g., Gini coefficient)
├── results/                 # Generated output and visualizations
│   ├── analysis/            # Detailed analysis results
│   ├── robustness_checks/   # Robustness test results
│   └── summary_stats/       # Descriptive statistics
└── README.md
```

### Key Files

- **agents.py**: Implements `Farmer` and `Forecaster` classes with decision-making logic
- **model.py**: Defines the `FarmingModel` class that orchestrates agent interactions
- **experiments.py**: Contains specialized experiments examining specific research questions
- **utils.py**: Includes the Gini coefficient calculation for inequality measurement

## Dependencies

- **Python 3.8+**
- **Mesa**: Agent-based modeling framework
- **NumPy & Pandas**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Data visualization
- **SciPy & Statsmodels**: Statistical analysis
- **Graphviz**: Flowchart generation

## Results and Visualization

Simulation outputs are organized into:

1. **Summary Statistics** (`results/summary_stats/`):
   - Initial wealth, land, and trust distributions
   - Demographic breakdowns of farmer types
   - Baseline model characteristics

2. **Analysis Results** (`results/analysis/`):
   - Time series of wealth, trust, and inequality measures
   - Comparative outcomes across dissemination modes
   - Impact of forecast accuracy on economic indicators
   - Trust dynamics through crisis and recovery periods

3. **Robustness Checks** (`results/robustness_checks/`):
   - Parameter sensitivity testing
   - Distribution variation effects
   - Time and scale effects

4. **Flowcharts** (`flowcharts/`):
   - Model process visualizations
   - Trust dynamics representations
   - Decision-making flowcharts

### Key Visualizations

The model generates several important visualizations:

- **Wealth Dynamics**: Time series showing wealth accumulation by farmer type and dissemination mode
- **Trust Evolution**: Changes in forecast trust over time with different accuracy levels
- **Gini Coefficient Trends**: Inequality measures across experimental conditions
- **Yield Comparisons**: Agricultural productivity under different forecast scenarios
- **Distribution Comparisons**: Initial vs. final wealth distributions
- **Recovery Patterns**: Economic and trust recovery after system failures

## Experimental Findings

Some key findings from the model include:

1. **Dissemination Effectiveness**: Extension officers and community leaders generally lead to better economic outcomes than radio or mobile app dissemination
2. **Trust-Yield Relationship**: Higher trust levels correlate with improved agricultural yields
3. **Inequality Dynamics**: Forecast accuracy improvements can initially increase inequality before decreasing it in the long run
4. **Recovery Patterns**: Trust recovers more slowly than economic indicators after forecast failures
5. **Subsidy Impact**: Well-targeted subsidies can significantly reduce inequality without compromising productivity

## Contributing

Contributions to the model are welcome. Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

*This agent-based model was developed for research in development economics focusing on agricultural decision-making and information systems.*