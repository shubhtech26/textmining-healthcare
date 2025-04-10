import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import re
import nltk
from collections import Counter
import textwrap

# Download required NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class EnhancedClinicalAnalyzer:
    def __init__(self):
        # Load class names
        self.classes = self._load_class_names()
        
        # Define indicator terms for different clinical entities
        self.symptom_indicators = [
            "complains of", "reports", "experiencing", "presents with", "symptoms include",
            "noted", "exhibits", "showing", "demonstrating", "pain", "ache", "discomfort",
            "swelling", "redness", "fatigue", "weakness", "dizziness", "nausea"
        ]
        
        self.diagnosis_indicators = [
            "diagnosed with", "assessment", "impression", "confirms", "consistent with",
            "suggestive of", "indicative of", "probable", "suspected", "diagnosis",
            "condition", "disorder", "disease", "syndrome", "pathology"
        ]
        
        self.treatment_indicators = [
            "treated with", "prescribed", "administered", "therapy", "procedure",
            "intervention", "surgery", "medication", "recommended", "plan",
            "management", "advised", "initiated on", "continue", "follow-up"
        ]
        
        # Create simple term normalization dictionary
        self.term_normalization = {
            "microscopic hematuria": "microhematuria",
            "difficulty urinating": "dysuria",
            "high blood pressure": "hypertension",
            "heart attack": "myocardial infarction",
            "kidney stone": "nephrolithiasis",
            "urinary tract infection": "UTI",
            "unable to urinate": "urinary retention",
            "blood in urine": "hematuria",
            "shortness of breath": "dyspnea"
        }
    
    def _load_class_names(self):
        try:
            with open('data/classes.txt', 'r') as f:
                classes = [line.strip() for line in f if line.strip()]
            return classes
        except Exception as e:
            print(f"Error loading class names: {e}")
            return ["Surgery", "Medical Records", "Internal Medicine", "Other"]
    
    def load_and_sample_data(self, file_path, samples_per_class=15):
        """
        Load data from CSV and create a balanced sample across medical classes
        """
        try:
            df = pd.read_csv(file_path)
            
            # Ensure the medical_specialty column exists
            if 'medical_specialty' not in df.columns:
                print("Warning: 'medical_specialty' column not found. Using random sampling.")
                if len(df) > samples_per_class * len(self.classes):
                    return df.sample(samples_per_class * len(self.classes))
                return df
            # Clean up class names in the dataframe to match our class list
            df['medical_specialty'] = df['medical_specialty'].str.strip()
            
            # Map similar specialties to our main classes
            specialty_mapping = {
                'Cardiovascular': 'Internal Medicine', 
                'Neurology': 'Internal Medicine',
                'Gastroenterology': 'Internal Medicine',
                'Nephrology': 'Internal Medicine',
                'Orthopedic': 'Surgery',
                'Urology': 'Surgery',
                'ENT': 'Surgery',
                'Discharge Summary': 'Medical Records',
                'Emergency': 'Other',
                'Pediatrics': 'Other',
                'General Medicine': 'Internal Medicine'
            }
            
            # Apply mapping where possible
            for specialty, main_class in specialty_mapping.items():
                df.loc[df['medical_specialty'].str.contains(specialty, case=False), 'mapped_specialty'] = main_class
            
            # For unmapped specialties, use 'Other'
            df.loc[df['mapped_specialty'].isna(), 'mapped_specialty'] = 'Other'
            
            # Select balanced samples from each class
            balanced_samples = []
            for class_name in self.classes:
                class_df = df[df['mapped_specialty'] == class_name]
                if len(class_df) >= samples_per_class:
                    balanced_samples.append(class_df.sample(samples_per_class))
                else:
                    # If not enough samples, take all available
                    balanced_samples.append(class_df)
                    print(f"Warning: Only {len(class_df)} samples available for {class_name}")
            
            return pd.concat(balanced_samples)
            
        except Exception as e:
            print(f"Error loading or sampling data: {e}")
            return None
    
    def normalize_entity(self, entity_text):
        """
        Clean and normalize entity text
        """
        if not entity_text:
            return ""
            
        # Convert to lowercase and remove extra whitespace
        text = entity_text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing punctuation and common articles
        text = re.sub(r'^[.,;:\s()]+', '', text)
        text = re.sub(r'[.,;:\s()]+$', '', text)
        
        # Check if we have a direct normalization
        if text in self.term_normalization:
            return self.term_normalization[text]
            
        return text
    
    def _extract_entities(self, text, indicators):
        """
        Extract clinical entities based on indicator terms with improved context handling
        """
        if not text:
            return []
            
        entities = []
        sentences = nltk.sent_tokenize(text)
        
        for sentence in sentences:
            for indicator in indicators:
                pattern = r'\b' + re.escape(indicator) + r'\b'
                if re.search(pattern, sentence, re.IGNORECASE):
                    # Find the indicator position
                    match = re.search(pattern, sentence, re.IGNORECASE)
                    if match:
                        # Extract text after the indicator (limited to 10 words)
                        start_pos = match.end()
                        after_text = sentence[start_pos:].strip()
                        # Extract up to 10 words, stop at period if present
                        words = after_text.split()[:10]
                        entity_text = ' '.join(words)
                        # If entity ends with a period, stop there
                        period_pos = entity_text.find('.')
                        if period_pos > 0:
                            entity_text = entity_text[:period_pos]
                        
                        # Normalize the entity
                        normalized_entity = self.normalize_entity(entity_text)
                        
                        # Only include if entity has substance (at least 2 chars)
                        if len(normalized_entity) > 2:
                            entities.append(normalized_entity)
        
        # For short entities, also look for specific medical terms directly
        medical_terms_patterns = [
            r'\b(pain|ache)\b', 
            r'\b(tumor|mass|lesion)\b',
            r'\b(infection|inflammation)\b',
            r'\b(fracture|break)\b',
            r'\b(surgery|procedure)\b'
        ]
        
        for pattern in medical_terms_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity_text = match.group(0)
                entities.append(self.normalize_entity(entity_text))
        
        return entities
    
    def analyze_text(self, text):
        """
        Analyze clinical text to extract symptoms, diagnoses, and treatments
        """
        symptoms = self._extract_entities(text, self.symptom_indicators)
        diagnoses = self._extract_entities(text, self.diagnosis_indicators)
        treatments = self._extract_entities(text, self.treatment_indicators)
        
        # Only consider entities of reasonable length and frequency
        return {
            "symptoms": [s for s in symptoms if 2 < len(s) < 100],
            "diagnoses": [d for d in diagnoses if 2 < len(d) < 100],
            "treatments": [t for t in treatments if 2 < len(t) < 100]
        }
    
    def analyze_corpus(self, df, sample_size=None):
        """
        Analyze a corpus of clinical texts
        """
        if sample_size and len(df) > sample_size:
            df = df.sample(sample_size)
            
        all_results = []
        
        # Check for expected column names
        text_column = 'transcription' if 'transcription' in df.columns else 'text'
        if text_column not in df.columns:
            print(f"Error: Neither 'transcription' nor 'text' column found in dataframe")
            return {}
            
        for idx, row in df.iterrows():
            if idx % 10 == 0:
                print(f"Processing document {idx+1}/{len(df)}...")
                
            text = row[text_column]
            result = self.analyze_text(text)
            
            # Add class information if available
            if 'mapped_specialty' in df.columns:
                result['class'] = row['mapped_specialty']
            elif 'medical_specialty' in df.columns:
                result['class'] = row['medical_specialty']
            else:
                result['class'] = 'Unknown'
                
            all_results.append(result)
            
        # Aggregate results
        symptom_counts = Counter()
        diagnosis_counts = Counter()
        treatment_counts = Counter()
        
        # Track counts by class
        class_symptoms = {cls: Counter() for cls in self.classes}
        class_diagnoses = {cls: Counter() for cls in self.classes}
        class_treatments = {cls: Counter() for cls in self.classes}
        
        for result in all_results:
            symptom_counts.update(result['symptoms'])
            diagnosis_counts.update(result['diagnoses'])
            treatment_counts.update(result['treatments'])
            
            # Update class-specific counters
            if 'class' in result:
                doc_class = result['class']
                if doc_class in class_symptoms:
                    class_symptoms[doc_class].update(result['symptoms'])
                    class_diagnoses[doc_class].update(result['diagnoses'])
                    class_treatments[doc_class].update(result['treatments'])
        
        return {
            'symptom_counts': dict(symptom_counts),
            'diagnosis_counts': dict(diagnosis_counts),
            'treatment_counts': dict(treatment_counts),
            'class_symptoms': {cls: dict(counter) for cls, counter in class_symptoms.items()},
            'class_diagnoses': {cls: dict(counter) for cls, counter in class_diagnoses.items()},
            'class_treatments': {cls: dict(counter) for cls, counter in class_treatments.items()}
        }
        
    def generate_visualizations(self, analysis_results, output_dir="enhanced_plots"):
        """
        Generate enhanced visualizations from analysis results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save analysis results to JSON
        with open(f"{output_dir}/enhanced_analysis_results.json", 'w') as f:
            json.dump(analysis_results, f, indent=2)
            
        # Generate overall entity visualizations
        self._plot_top_entities(analysis_results, output_dir)
        
        # Generate class-specific visualizations
        self._plot_class_specific_entities(analysis_results, output_dir)
        
    def _plot_top_entities(self, analysis_results, output_dir, n_top=15):
        """
        Plot top entities overall
        """
        entity_types = {
            'symptom': 'Symptoms',
            'diagnosis': 'Diagnoses', 
            'treatment': 'Treatments'
        }
        
        for entity_key, entity_label in entity_types.items():
            counts_dict = analysis_results.get(f"{entity_key}_counts", {})
            if not counts_dict:
                print(f"No data found for {entity_key}_counts. Skipping plot.")
                continue
                
            # Convert to DataFrame and get top N
            df = pd.DataFrame(list(counts_dict.items()), columns=[entity_label, 'Frequency'])
            df = df.sort_values('Frequency', ascending=False).head(n_top)
            
            if df.empty:
                print(f"No data to plot for {entity_key} after filtering.")
                continue
                
            # Create enhanced visualization
            plt.figure(figsize=(12, 10))
            
            # Wrap long labels for better readability
            wrapped_labels = [textwrap.fill(str(label), 40) for label in df[entity_label]]
            
            # Create gradient colors based on frequency
            colors = plt.cm.Blues(df['Frequency'] / df['Frequency'].max())
            
            # Plot horizontal bar chart with gradient colors
            bars = plt.barh(wrapped_labels, df['Frequency'], color=colors)
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                        f'{width:.0f}', va='center')
            
            plt.title(f'Top {n_top} {entity_label} in Medical Transcriptions', fontsize=16)
            plt.xlabel('Frequency', fontsize=12)
            plt.ylabel(entity_label, fontsize=12)
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(f"{output_dir}/enhanced_top_{entity_key}s.png")
            plt.close()
    
    def _plot_class_specific_entities(self, analysis_results, output_dir, n_top=10):
        """
        Plot top entities by medical class
        """
        entity_types = {
            'symptom': ('Symptoms', 'class_symptoms'),
            'diagnosis': ('Diagnoses', 'class_diagnoses'),
            'treatment': ('Treatments', 'class_treatments')
        }
        
        for entity_key, (entity_label, class_dict_key) in entity_types.items():
            class_data = analysis_results.get(class_dict_key, {})
            if not class_data:
                print(f"No class-specific data found for {entity_key}. Skipping plot.")
                continue
                
            # Create subplots for each class
            fig, axes = plt.subplots(2, 2, figsize=(16, 14))
            axes = axes.flatten()
            
            for i, class_name in enumerate(self.classes):
                if class_name not in class_data:
                    continue
                    
                # Get data for this class
                counts_dict = class_data[class_name]
                if not counts_dict:
                    axes[i].text(0.5, 0.5, f"No {entity_label} data for {class_name}", 
                                ha='center', va='center', fontsize=12)
                    axes[i].set_title(f"Top {entity_label} in {class_name}", fontsize=14)
                    continue
                
                # Convert to DataFrame and get top N
                df = pd.DataFrame(list(counts_dict.items()), columns=[entity_label, 'Frequency'])
                df = df.sort_values('Frequency', ascending=False).head(n_top)
                
                if df.empty:
                    axes[i].text(0.5, 0.5, f"No valid {entity_label} data for {class_name}", 
                                ha='center', va='center', fontsize=12)
                    axes[i].set_title(f"Top {entity_label} in {class_name}", fontsize=14)
                    continue
                    
                # Wrap long labels
                wrapped_labels = [textwrap.fill(str(label), 30) for label in df[entity_label]]
                
                # Create bar chart
                bars = axes[i].barh(wrapped_labels, df['Frequency'], 
                                   color=plt.cm.Blues(0.6 + i*0.1))
                
                # Add labels
                for bar in bars:
                    width = bar.get_width()
                    axes[i].text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                                f'{width:.0f}', va='center')
                
                axes[i].set_title(f"Top {entity_label} in {class_name}", fontsize=14)
                axes[i].set_xlabel('Frequency', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/enhanced_{entity_key}s_by_class.png")
            plt.close()

# Main execution function
def main():
    analyzer = EnhancedClinicalAnalyzer()
    
    print("Loading and sampling data...")
    data_path = "data/mtsamples.csv"
    df = analyzer.load_and_sample_data(data_path, samples_per_class=15)
    
    if df is None or len(df) == 0:
        print("Error: Could not load or sample data. Exiting.")
        return
        
    print(f"Analyzing {len(df)} medical documents...")
    results = analyzer.analyze_corpus(df)
    
    print("Generating enhanced visualizations...")
    analyzer.generate_visualizations(results, "enhanced_plots")
    
    print("Analysis and visualization complete.")
    
if __name__ == "__main__":
    main() 