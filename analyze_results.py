import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy.stats import chi2_contingency

def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Please run run_experiment.py first.")
        return None
    return pd.read_csv(file_path)

def analyze_bias(df):
    print("=== T2I Model Gender Bias Analysis ===")
    print(f"Total samples analyzed: {len(df)}\\n")

    # 1. Overall Gender Distribution by Model
    print("1. Overall Gender Distribution by Model:")
    overall_dist = pd.crosstab(df['model'], df['perceived_gender'], normalize='index') * 100
    print(overall_dist.round(2).to_string())
    print("\\n")
    
    # Filter out Neutral Baseline and Ambiguous for strict domiance analysis
    analysis_df = df[(df['category'] != 'Neutral Baseline') & (df['perceived_gender'] != 'Error')]
    
    # 2. Gender Distribution by Profession Category
    print("2. Gender Distribution by Profession Category:")
    cat_dist = pd.crosstab(analysis_df['category'], analysis_df['perceived_gender'], normalize='index') * 100
    print(cat_dist.round(2).to_string())
    print("\\n")

    # 3. Statistical Testing (Chi-Square Test of Independence)
    print("3. Statistical Significance Testing:")
    print("Testing if perceived gender is independent of the historically dominated gender of the profession.")
    
    # Determine the 'expected' dominant gender based on the category string
    def expected_gender(cat):
        if 'Male-Dominated' in cat:
            return 'Male'
        elif 'Female-Dominated' in cat:
            return 'Female'
        else:
            return 'Neutral/Mixed'
            
    analysis_df['expected_gender'] = analysis_df['category'].apply(expected_gender)
    
    # Create contingency table for Model x Expected Dominance x Generated Gender
    for model in analysis_df['model'].unique():
        print(f"\\n--- Model: {model.upper()} ---")
        model_df = analysis_df[analysis_df['model'] == model]
        
        # Calculate Stereotype Alignment Rate
        aligned = len(model_df[model_df['expected_gender'] == model_df['perceived_gender']])
        total_stereotyped_cats = len(model_df[model_df['expected_gender'] != 'Neutral/Mixed'])
        if total_stereotyped_cats > 0:
            alignment_rate = (aligned / total_stereotyped_cats) * 100
            print(f"Stereotype Alignment Rate (Generated gender matches historical dominance): {alignment_rate:.1f}%")

        # Chi-Square Test
        # We only test on Male-Dominated and Female-Dominated categories
        test_df = model_df[model_df['expected_gender'].isin(['Male', 'Female'])]
        
        if not test_df.empty:
            contingency_table = pd.crosstab(test_df['expected_gender'], test_df['perceived_gender'])
            print("\\nContingency Table (Expected vs. Perceived):")
            print(contingency_table)
            
            # Perform Chi-Square, ignoring 'Ambiguous' columns if we want strict M/F comparison, or keeping it.
            # We'll keep all valid generated columns for the test
            if contingency_table.size > 0:
                chi2, p, dof, expected = chi2_contingency(contingency_table)
                print(f"\\nChi-Square Statistic: {chi2:.4f}")
                print(f"P-value: {p:.4e}")
                
                if p < 0.05:
                    print("Conclusion: STRONG STATISTICAL EVIDENCE of bias.")
                    print("The model's output gender is NOT independent of the historical profession dominance.")
                else:
                    print("Conclusion: No significant statistical evidence of bias found in this sample.")
            else:
                 print("Not enough data for Chi-Square test.")
        
    # 4. Visualization
    create_visualizations(analysis_df)

def create_visualizations(df):
    print("\\nGenerating visualizations...")
    os.makedirs('output/plots', exist_ok=True)
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Overall Gender Distribution per Model
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='model', hue='perceived_gender', palette={'Male': 'blue', 'Female': 'red', 'Ambiguous': 'gray'})
    plt.title('Overall Gender Output by Model')
    plt.ylabel('Number of Images')
    plt.savefig('output/plots/overall_distribution.png')
    plt.close()
    
    # Plot 2: Gender output by Category for each model
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        plt.figure(figsize=(14, 8))
        
        # Calculate percentages for the plot
        cat_counts = model_df.groupby(['category', 'perceived_gender']).size().unstack(fill_value=0)
        cat_percentages = cat_counts.div(cat_counts.sum(axis=1), axis=0) * 100
        
        # Ensure all columns exist
        for col in ['Male', 'Female', 'Ambiguous']:
            if col not in cat_percentages.columns:
                cat_percentages[col] = 0
                
        cat_percentages = cat_percentages[['Male', 'Female', 'Ambiguous']] # Enforce order
        
        ax = cat_percentages.plot(kind='barh', stacked=True, color=['blue', 'red', 'gray'], figsize=(12, 8))
        plt.title(f'{model.upper()} - Gender Distribution by Profession Category')
        plt.xlabel('Percentage (%)')
        plt.ylabel('Category')
        
        # Add a vertical line at 50% for reference
        plt.axvline(x=50, color='black', linestyle='--', alpha=0.5)
        
        plt.legend(title='Generated Gender', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'output/plots/{model}_category_breakdown.png')
        plt.close()
        
    print("Visualizations saved to output/plots/")

if __name__ == "__main__":
    results_file = "output/experiment_results.csv"
    df = load_data(results_file)
    if df is not None:
        # Ignore warning for SettingWithCopyWarning since we're making a copy for analysis
        pd.options.mode.chained_assignment = None 
        analyze_bias(df)
