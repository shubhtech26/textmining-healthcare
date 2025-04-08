import pandas as pd
import json
from clinical_analysis import ClinicalPatternAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def plot_top_entities(stats, entity_type, title, n=10):
    """Plot top n entities of a given type."""
    counts = stats[f'{entity_type}_counts']
    top_n = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:n])
    
    plt.figure(figsize=(12, 6))
    plt.bar(top_n.keys(), top_n.values())
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Top {n} {title}')
    plt.tight_layout()
    return plt

def main():
    # Load the medical transcriptions data
    print("Loading medical transcriptions...")
    df = pd.read_csv('data/mtsamples.csv')
    
    # Remove any rows with NaN values in the transcription column
    df = df.dropna(subset=['transcription'])
    
    # Initialize the clinical pattern analyzer
    analyzer = ClinicalPatternAnalyzer()
    
    # Analyze a sample of records
    print("Analyzing clinical patterns...")
    sample_size = min(100, len(df))  # Analyze 100 records or all if less
    sample_texts = df['transcription'].head(sample_size).tolist()
    
    # Perform analysis
    results = analyzer.analyze_multiple_records(sample_texts)
    
    # Get statistics
    stats = analyzer.get_pattern_statistics(results)
    
    # Create visualizations directory
    import os
    os.makedirs('clinical_plots', exist_ok=True)
    
    # Plot top symptoms
    print("Generating visualization for top symptoms...")
    plt_symptoms = plot_top_entities(stats, 'symptom', 'Symptoms')
    plt_symptoms.savefig('clinical_plots/top_symptoms.png')
    plt_symptoms.close()
    
    # Plot top diagnoses
    print("Generating visualization for top diagnoses...")
    plt_diagnoses = plot_top_entities(stats, 'diagnosis', 'Diagnoses')
    plt_diagnoses.savefig('clinical_plots/top_diagnoses.png')
    plt_diagnoses.close()
    
    # Plot top treatments
    print("Generating visualization for top treatments...")
    plt_treatments = plot_top_entities(stats, 'treatment', 'Treatments')
    plt_treatments.savefig('clinical_plots/top_treatments.png')
    plt_treatments.close()
    
    # Save detailed analysis results
    print("Saving detailed analysis results...")
    with open('clinical_plots/analysis_results.json', 'w') as f:
        # Convert defaultdict to dict for JSON serialization
        stats_dict = {
            'symptom_counts': dict(stats['symptom_counts']),
            'diagnosis_counts': dict(stats['diagnosis_counts']),
            'treatment_counts': dict(stats['treatment_counts']),
            'common_symptom_diagnosis_pairs': {str(k): v for k, v in stats['common_symptom_diagnosis_pairs'].items()},
            'common_diagnosis_treatment_pairs': {str(k): v for k, v in stats['common_diagnosis_treatment_pairs'].items()}
        }
        json.dump(stats_dict, f, indent=2)
    
    print("Analysis complete! Check the 'clinical_plots' directory for results.")
    
    # Print some example findings
    print("\nExample findings from the analysis:")
    print("\nTop 5 Most Common Symptoms:")
    for symptom, count in sorted(stats['symptom_counts'].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"- {symptom}: {count} occurrences")
    
    print("\nTop 5 Most Common Diagnoses:")
    for diagnosis, count in sorted(stats['diagnosis_counts'].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"- {diagnosis}: {count} occurrences")
    
    print("\nTop 5 Most Common Treatments:")
    for treatment, count in sorted(stats['treatment_counts'].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"- {treatment}: {count} occurrences")

if __name__ == "__main__":
    main() 