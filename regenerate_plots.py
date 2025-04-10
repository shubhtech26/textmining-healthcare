import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os
import textwrap # Import textwrap for potential label wrapping

def generate_clinical_entity_visualizations_adjusted(analysis_results_file, output_dir="clinical_plots"):
    """
    Generate visualizations for top clinical entities with adjustments for label visibility.
    
    Parameters:
    - analysis_results_file: Path to the JSON file with clinical entity counts
    - output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data from the JSON file
    try:
        with open(analysis_results_file, 'r') as f:
            analysis_results = json.load(f)
    except FileNotFoundError:
        print(f"Error: Analysis results file not found at {analysis_results_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {analysis_results_file}")
        return

    # Common visualization parameters
    n_top = 15  # Number of top entities to display
    fig_size = (12, 10) # Adjusted figure size
    label_wrap_length = 60 # Max length before wrapping y-axis labels

    # Get data for each entity type
    entities = {
        'symptom': ('Symptom', 'Frequency', analysis_results.get('symptom_counts', {})),
        'diagnosis': ('Diagnosis', 'Frequency', analysis_results.get('diagnosis_counts', {})),
        'treatment': ('Treatment', 'Frequency', analysis_results.get('treatment_counts', {}))
    }

    for entity_key, (entity_label, count_label, counts_dict) in entities.items():
        if not counts_dict:
            print(f"No data found for {entity_key}_counts. Skipping plot.")
            continue

        plt.figure(figsize=fig_size)
        
        # Create DataFrame and sort
        df = pd.DataFrame(list(counts_dict.items()), 
                          columns=[entity_label, count_label])
        # Ensure counts are numeric, handle potential errors
        df[count_label] = pd.to_numeric(df[count_label], errors='coerce')
        df.dropna(subset=[count_label], inplace=True) # Remove rows where count couldn't be converted
        
        if df.empty:
             print(f"No valid data to plot for {entity_key} after filtering.")
             plt.close()
             continue
             
        df[count_label] = df[count_label].astype(int) # Convert valid counts to integer
        df = df.sort_values(count_label, ascending=False).head(n_top)

        if df.empty:
             print(f"No valid data remaining to plot for {entity_key} after sorting/filtering.")
             plt.close()
             continue
             
        # Wrap long labels for the y-axis
        wrapped_labels = [textwrap.fill(str(label), label_wrap_length) for label in df[entity_label]] # Ensure label is string

        # Create horizontal bar chart
        ax = sns.barplot(x=count_label, y=wrapped_labels, data=df, orient='h')
        ax.set_title(f'Top {n_top} {entity_label.capitalize()}s in Medical Transcriptions', fontsize=16)
        ax.set_xlabel(count_label, fontsize=12)
        ax.set_ylabel(entity_label, fontsize=12)
        
        # Add count labels to bars
        # Adjust label positioning slightly for better visibility
        max_count = df[count_label].max() if not df.empty else 1 # Avoid division by zero
        for i, v in enumerate(df[count_label]):
             # Position text slightly to the right of the bar end
             ax.text(v + max_count * 0.01, i, str(v), va='center', fontsize=10) 

        # Adjust layout to prevent label cutoff
        plt.tight_layout(pad=1.0) # Use tight_layout first
        # If labels are still cut, try adjusting further:
        # plt.subplots_adjust(left=0.35) # Increase left margin significantly if needed

        plot_filename = f"{output_dir}/top_{entity_key}s.png"
        try:
            plt.savefig(plot_filename)
            print(f"Saved adjusted plot: {plot_filename}")
        except Exception as e:
            print(f"Error saving plot {plot_filename}: {e}")
        finally:
             plt.close() # Close the figure to free memory

# Specify the path to the analysis results file
results_file = "clinical_plots/analysis_results.json"

# --- Main execution part ---
if __name__ == "__main__":
    print("Starting plot regeneration...")
    generate_clinical_entity_visualizations_adjusted(results_file)
    print("Finished regenerating clinical entity plots with adjustments.") 