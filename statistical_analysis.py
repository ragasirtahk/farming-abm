import os
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import pearsonr
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Ensure the analysis results directory exists
os.makedirs("results/analysis", exist_ok=True)

def analyze_forecast_accuracy_relation(csv_path):
    """
    Analyzes the relationship between forecast accuracy, Gini coefficient, and yield.
    
    Args:
    - csv_path (str): Path to the CSV file containing simulation results.
    """
    data = pd.read_csv(csv_path)
    
    # Correlation between Forecast Accuracy and Gini Coefficient
    corr_gini, p_gini = pearsonr(data['Forecast Accuracy'], data['Gini Coefficient'])
    print(f"Correlation between Forecast Accuracy and Gini Coefficient: {corr_gini:.2f} (p-value: {p_gini:.4f})")
    
    # Correlation between Forecast Accuracy and Average Yield
    corr_yield, p_yield = pearsonr(data['Forecast Accuracy'], data['Average Yield'])
    print(f"Correlation between Forecast Accuracy and Average Yield: {corr_yield:.2f} (p-value: {p_yield:.4f})")
    
    # Write correlations to a file
    with open("results/analysis/forecast_accuracy_correlations.txt", "w") as f:
        f.write(f"Correlation between Forecast Accuracy and Gini Coefficient: {corr_gini:.2f} (p-value: {p_gini:.4f})\n")
        f.write(f"Correlation between Forecast Accuracy and Average Yield: {corr_yield:.2f} (p-value: {p_yield:.4f})\n")
    
    # Visualization
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data, x="Forecast Accuracy", y="Gini Coefficient", hue="Forecast Accuracy", palette="viridis")
    sns.regplot(data=data, x="Forecast Accuracy", y="Gini Coefficient", scatter=False, color='red')
    plt.title("Forecast Accuracy vs. Gini Coefficient")
    plt.xlabel("Forecast Accuracy")
    plt.ylabel("Gini Coefficient")
    plt.tight_layout()
    plt.savefig("results/analysis/forecast_accuracy_vs_gini.png")
    plt.close()
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data, x="Forecast Accuracy", y="Average Yield", hue="Forecast Accuracy", palette="viridis")
    sns.regplot(data=data, x="Forecast Accuracy", y="Average Yield", scatter=False, color='red')
    plt.title("Forecast Accuracy vs. Average Yield")
    plt.xlabel("Forecast Accuracy")
    plt.ylabel("Average Yield")
    plt.tight_layout()
    plt.savefig("results/analysis/forecast_accuracy_vs_yield.png")
    plt.close()

def perform_descriptive_statistics(file_path):
    data = pd.read_csv(file_path)
    desc_stats = data.groupby('Dissemination Mode').agg({
        'Gini Coefficient': ['mean', 'std', 'count'],
        'Average Trust': ['mean', 'std', 'count']
    })
    print("Descriptive Statistics:\n", desc_stats)
    
    # Save Descriptive Statistics to the common file
    with open("results/analysis/dissemination_analysis_results.txt", "a") as f:
        f.write("Descriptive Statistics:\n")
        f.write(desc_stats.to_string())
        f.write("\n\n")

def perform_anova(file_path, dependent_var):
    data = pd.read_csv(file_path)
    
    # Rename columns to remove spaces
    data.rename(columns={
        'Dissemination Mode': 'Dissemination_Mode',
        dependent_var: dependent_var.replace(' ', '_')
    }, inplace=True)
    
    # Update dependent_var to match the renamed column
    dependent_var = dependent_var.replace(' ', '_')
    
    # Update the ANOVA formula without backticks
    formula = f'{dependent_var} ~ C(Dissemination_Mode)'
    model = ols(formula, data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(f"ANOVA results for {dependent_var}:\n", anova_table)
    
    # Save ANOVA results to the common file
    with open("results/analysis/dissemination_analysis_results.txt", "a") as f:
        f.write(f"ANOVA results for {dependent_var}:\n")
        f.write(anova_table.to_string())
        f.write("\n\n")
    
    return anova_table

def perform_posthoc_tests(file_path, dependent_var):
    data = pd.read_csv(file_path)
    
    # Rename columns to remove spaces
    data.rename(columns={
        'Dissemination Mode': 'Dissemination_Mode',
        dependent_var: dependent_var.replace(' ', '_')
    }, inplace=True)
    
    # Update dependent_var to match the renamed column
    dependent_var = dependent_var.replace(' ', '_')
    
    # Perform Tukey HSD test with updated column names
    tukey = pairwise_tukeyhsd(
        endog=data[dependent_var],
        groups=data['Dissemination_Mode'],
        alpha=0.05
    )
    print(f"Post-hoc Tukey HSD results for {dependent_var}:\n", tukey)
    
    # Save Post-hoc Test results to the common file
    with open("results/analysis/dissemination_analysis_results.txt", "a") as f:
        f.write(f"Post-hoc Tukey HSD results for {dependent_var}:\n")
        f.write(tukey.summary().as_text())
        f.write("\n\n")

def check_assumptions(file_path, dependent_var):
    data = pd.read_csv(file_path)
    
    # Rename columns to remove spaces
    data.rename(columns={
        'Dissemination Mode': 'Dissemination_Mode',
        dependent_var: dependent_var.replace(' ', '_')
    }, inplace=True)
    
    # Update dependent_var to match the renamed column
    dependent_var = dependent_var.replace(' ', '_')
    
    # Group data based on the renamed column
    groups = data.groupby('Dissemination_Mode')[dependent_var].apply(list)
    
    # Normality Test
    normality_results = f"Shapiro-Wilk Test for {dependent_var}:\n"
    for group, values in groups.items():
        stat, p = stats.shapiro(values)
        normality_results += f"{group}: Statistics={stat:.3f}, p={p:.3f}\n"
    print(normality_results)
    
    # Homogeneity of Variances
    stat, p = stats.levene(*groups)
    homogeneity_results = f"\nLevene’s Test for Homogeneity of Variances:\nStatistics={stat:.3f}, p={p:.3f}\n"
    print(homogeneity_results)
    
    # Save Assumption Checks to the common file
    with open("results/analysis/dissemination_analysis_results.txt", "a") as f:
        f.write(f"Assumption Checks for {dependent_var}:\n")
        f.write(normality_results)
        f.write(homogeneity_results)
        f.write("\n")

def visualize_data(file_path):
    data = pd.read_csv(file_path)
    
    # Boxplot for Gini Coefficient
    plt.figure(figsize=(8,6))
    sns.boxplot(x='Dissemination Mode', y='Gini Coefficient', data=data)
    plt.title('Gini Coefficient by Dissemination Mode')
    plt.savefig("results/analysis/gini_boxplot.png")
    plt.close()
    
    # Boxplot for Average Trust
    plt.figure(figsize=(8,6))
    sns.boxplot(x='Dissemination Mode', y='Average Trust', data=data)
    plt.title('Average Trust by Dissemination Mode')
    plt.savefig("results/analysis/trust_boxplot.png")
    plt.close()

def analyze_dissemination_impact_stats(file_path):
    # Clear the common results file before starting
    open("results/analysis/dissemination_analysis_results.txt", "w").close()
    
    perform_descriptive_statistics(file_path)
    print("\n----- ANOVA for Gini Coefficient -----")
    anova_gini = perform_anova(file_path, 'Gini Coefficient')
    if (anova_gini['PR(>F)'][0] < 0.05):
        perform_posthoc_tests(file_path, 'Gini Coefficient')
    print("\n----- ANOVA for Average Trust -----")
    anova_trust = perform_anova(file_path, 'Average Trust')
    if (anova_trust['PR(>F)'][0] < 0.05):
        perform_posthoc_tests(file_path, 'Average Trust')
    print("\n----- Assumption Checks for Gini Coefficient -----")
    check_assumptions(file_path, 'Gini Coefficient')
    print("\n----- Assumption Checks for Average Trust -----")
    check_assumptions(file_path, 'Average Trust')
    print("\n----- Visualizing Data -----")
    visualize_data(file_path)
    # Optionally, append visualization summaries to the common file
    with open("results/analysis/dissemination_analysis_results.txt", "a") as f:
        f.write("Visualizations have been saved to the results/analysis directory.\n\n")

def analyze_simulate_forecast_system_failure():
    # Load data
    data = pd.read_csv("results/analysis/simulate_forecast_system_failure.csv")
    
    # Split data into pre-crisis and post-crisis
    pre_crisis = data[data['Season'] < 10]
    post_crisis = data[data['Season'] >= 10]
    
    # Perform t-test on Average Yield
    t_stat, p_value = stats.ttest_ind(pre_crisis['Average Yield'], post_crisis['Average Yield'])
    print("T-test for Average Yield between pre-crisis and post-crisis:")
    print(f"T-statistic: {t_stat}, P-value: {p_value}")
    
    # Correlation between Average Yield and Average Trust
    correlation, corr_p = stats.pearsonr(data['Average Yield'], data['Average Trust'])
    print("Correlation between Average Yield and Average Trust:")
    print(f"Correlation coefficient: {correlation}, P-value: {corr_p}")
    
    # Write outputs to file
    with open("results/analysis/analyze_forecast_crash.txt", "a") as f:
        f.write("----- Analyze Simulate Forecast System Failure -----\n")
        f.write(f"T-test for Average Yield between pre-crisis and post-crisis:\nT-statistic: {t_stat}, P-value: {p_value}\n")
        f.write(f"Correlation between Average Yield and Average Trust:\nCorrelation coefficient: {correlation}, P-value: {corr_p}\n\n")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x='Season', y='Average Yield', label='Average Yield')
    sns.lineplot(data=data, x='Season', y='Average Trust', label='Average Trust')
    plt.axvline(x=10, color='red', linestyle='--', label='Crisis Start')
    plt.title('Average Yield and Trust Over Seasons')
    plt.xlabel('Season')
    plt.ylabel('Values')
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/analysis/simulate_forecast_system_failure_analysis.png")
    plt.close()

def analyze_trust_recovery_r():
    # Load data
    data = pd.read_csv("results/analysis/analyze_trust_recovery.csv")
    
    # Analyze changes in Average Trust
    pre_intervention = data[data['Season'] < 15]
    post_intervention = data[data['Season'] >= 15]
    
    # T-test on Average Trust
    t_stat, p_value = stats.ttest_ind(pre_intervention['Average Trust'], post_intervention['Average Trust'])
    print("T-test for Average Trust before and after intervention:")
    print(f"T-statistic: {t_stat}, P-value: {p_value}")
    
    # Regression analysis
    slope, intercept, r_value, p_value, std_err = stats.linregress(data['Season'], data['Average Trust'])
    print("Regression analysis on Average Trust over Seasons:")
    print(f"Slope: {slope}, R-squared: {r_value**2}")
    
    # Write outputs to file
    with open("results/analysis/analyze_forecast_crash.txt", "a") as f:
        f.write("----- Analyze Trust Recovery -----\n")
        f.write(f"T-test for Average Trust before and after intervention:\nT-statistic: {t_stat}, P-value: {p_value}\n")
        f.write(f"Regression analysis on Average Trust over Seasons:\nSlope: {slope}, R-squared: {r_value**2}\n\n")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x='Season', y='Average Trust', label='Average Trust')
    plt.axvline(x=10, color='red', linestyle='--', label='Crisis Start')
    plt.axvline(x=15, color='orange', linestyle='--', label='Intervention Start')
    plt.title('Average Trust Over Seasons')
    plt.xlabel('Season')
    plt.ylabel('Average Trust')
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/analysis/analyze_trust_recovery_analysis.png")
    plt.close()

def analyze_robustness_time_and_scale(file_path):
    data = pd.read_csv(file_path)
    
    # Rename columns to remove spaces
    data.rename(columns={
        'Average Wealth': 'Average_Wealth',
        'Gini Coefficient': 'Gini_Coefficient'
    }, inplace=True)
    
    # Adjusted formulas without backticks
    model_wealth = ols('Average_Wealth ~ C(Scenario)', data=data).fit()
    anova_wealth = sm.stats.anova_lm(model_wealth, typ=2)
    print("ANOVA results for Average Wealth across Scenarios:\n", anova_wealth)
    
    model_gini = ols('Gini_Coefficient ~ C(Scenario)', data=data).fit()
    anova_gini = sm.stats.anova_lm(model_gini, typ=2)
    print("ANOVA results for Gini Coefficient across Scenarios:\n", anova_gini)
    
    # Save ANOVA results to a file
    with open("results/analysis/robustness_time_scale_analysis.txt", "w") as f:
        f.write("ANOVA results for Average Wealth across Scenarios:\n")
        f.write(anova_wealth.to_string())
        f.write("\n")
        
        # Perform Tukey HSD post-hoc test for Average Wealth if ANOVA is significant
        if anova_wealth['PR(>F)'][0] < 0.05:
            tukey_wealth = pairwise_tukeyhsd(endog=data['Average_Wealth'], groups=data['Scenario'], alpha=0.05)
            print("Post-hoc Tukey HSD results for Average Wealth:\n", tukey_wealth)
            f.write("\nPost-hoc Tukey HSD results for Average Wealth:\n")
            f.write(tukey_wealth.summary().as_text())
        
        f.write("\nANOVA results for Gini Coefficient across Scenarios:\n")
        f.write(anova_gini.to_string())
        f.write("\n")
        
        # Perform Tukey HSD post-hoc test for Gini Coefficient if ANOVA is significant
        if anova_gini['PR(>F)'][0] < 0.05:
            tukey_gini = pairwise_tukeyhsd(endog=data['Gini_Coefficient'], groups=data['Scenario'], alpha=0.05)
            print("Post-hoc Tukey HSD results for Gini Coefficient:\n", tukey_gini)
            f.write("\nPost-hoc Tukey HSD results for Gini Coefficient:\n")
            f.write(tukey_gini.summary().as_text())
    
    # Visualization
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x='Season', y='Average_Wealth', hue='Scenario')
    plt.title('Average Wealth Over Seasons by Scenario')
    plt.xlabel('Season')
    plt.ylabel('Average Wealth')
    plt.legend(title='Scenario')
    plt.tight_layout()
    plt.savefig("results/analysis/robustness_time_scale_wealth.png")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x='Season', y='Gini_Coefficient', hue='Scenario')
    plt.title('Gini Coefficient Over Seasons by Scenario')
    plt.xlabel('Season')
    plt.ylabel('Gini Coefficient')
    plt.legend(title='Scenario')
    plt.tight_layout()
    plt.savefig("results/analysis/robustness_time_scale_gini.png")
    plt.close()

if __name__ == "__main__":
    
    analyze_forecast_accuracy_relation("results/analysis/analyze_forecast_accuracy_relation.csv")
    
    # Example usage:
    # analyze_dissemination_impact_stats("results/analysis/analyze_dissemination_impact.csv")
