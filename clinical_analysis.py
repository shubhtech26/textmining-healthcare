import pandas as pd
import numpy as np
from collections import defaultdict
import re
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from typing import Dict, List, Tuple

class ClinicalPatternAnalyzer:
    def __init__(self):
        """Initialize the clinical pattern analyzer with necessary models and resources."""
        # Load medical NER model for clinical entity recognition
        self.ner = pipeline("ner", model="samrawal/bert-base-uncased_clinical-ner")
        
        # Download required NLTK resources
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # Common medical terms and patterns
        self.symptom_indicators = [
            "complains of", "reports", "presents with", "symptoms include",
            "experiencing", "suffered from", "noted", "exhibits"
        ]
        
        self.diagnosis_indicators = [
            "diagnosed with", "assessment:", "impression:", "conclusion:",
            "findings suggest", "consistent with", "confirms", "reveals"
        ]
        
        self.treatment_indicators = [
            "treated with", "prescribed", "recommended", "therapy",
            "administered", "management includes", "plan:", "treatment:"
        ]

    def identify_symptoms(self, text: str) -> List[Dict[str, str]]:
        """
        Identify symptoms mentioned in the text.
        
        Args:
            text (str): Clinical text to analyze
            
        Returns:
            List[Dict[str, str]]: List of identified symptoms with context
        """
        if not isinstance(text, str) or pd.isna(text):
            return []
            
        symptoms = []
        sentences = sent_tokenize(text.lower())
        
        # Use NER to identify medical entities
        entities = self.ner(text)
        
        # Extract symptoms based on medical entity recognition
        for entity in entities:
            if entity['entity'].endswith('Problem') or entity['entity'].endswith('Symptom'):
                symptoms.append({
                    'symptom': entity['word'],
                    'confidence': entity['score'],
                    'context': self._get_context(text, entity['start'], entity['end'])
                })
        
        # Look for symptom patterns in each sentence
        for sentence in sentences:
            for indicator in self.symptom_indicators:
                matches = re.finditer(fr"{indicator}[^\S\r\n]+(.*?)(?=[.;])", sentence)
                for match in matches:
                    symptom_text = match.group(1).strip()
                    symptoms.append({
                        'symptom': symptom_text,
                        'confidence': 1.0,
                        'context': sentence
                    })
        
        return symptoms

    def extract_diagnoses(self, text: str) -> List[Dict[str, str]]:
        """
        Extract diagnoses from the clinical text.
        
        Args:
            text (str): Clinical text to analyze
            
        Returns:
            List[Dict[str, str]]: List of identified diagnoses with context
        """
        if not isinstance(text, str) or pd.isna(text):
            return []
            
        diagnoses = []
        sentences = sent_tokenize(text.lower())
        
        # Use NER to identify diagnostic entities
        entities = self.ner(text)
        
        # Extract diagnoses based on medical entity recognition
        for entity in entities:
            if entity['entity'].endswith('Diagnosis') or entity['entity'].endswith('Disease'):
                diagnoses.append({
                    'diagnosis': entity['word'],
                    'confidence': entity['score'],
                    'context': self._get_context(text, entity['start'], entity['end'])
                })
        
        # Look for diagnosis patterns in each sentence
        for sentence in sentences:
            for indicator in self.diagnosis_indicators:
                matches = re.finditer(fr"{indicator}[^\S\r\n]+(.*?)(?=[.;])", sentence)
                for match in matches:
                    diagnosis_text = match.group(1).strip()
                    diagnoses.append({
                        'diagnosis': diagnosis_text,
                        'confidence': 1.0,
                        'context': sentence
                    })
        
        return diagnoses

    def find_treatment_patterns(self, text: str) -> List[Dict[str, str]]:
        """
        Identify treatment patterns in the clinical text.
        
        Args:
            text (str): Clinical text to analyze
            
        Returns:
            List[Dict[str, str]]: List of identified treatments with context
        """
        if not isinstance(text, str) or pd.isna(text):
            return []
            
        treatments = []
        sentences = sent_tokenize(text.lower())
        
        # Use NER to identify treatment entities
        entities = self.ner(text)
        
        # Extract treatments based on medical entity recognition
        for entity in entities:
            if entity['entity'].endswith('Treatment') or entity['entity'].endswith('Procedure'):
                treatments.append({
                    'treatment': entity['word'],
                    'confidence': entity['score'],
                    'context': self._get_context(text, entity['start'], entity['end'])
                })
        
        # Look for treatment patterns in each sentence
        for sentence in sentences:
            for indicator in self.treatment_indicators:
                matches = re.finditer(fr"{indicator}[^\S\r\n]+(.*?)(?=[.;])", sentence)
                for match in matches:
                    treatment_text = match.group(1).strip()
                    treatments.append({
                        'treatment': treatment_text,
                        'confidence': 1.0,
                        'context': sentence
                    })
        
        return treatments

    def _get_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Get surrounding context for an entity."""
        text_start = max(0, start - window)
        text_end = min(len(text), end + window)
        return text[text_start:text_end].strip()

    def analyze_clinical_patterns(self, text_data: str) -> Dict:
        """
        Comprehensive analysis of clinical patterns in text data.
        
        Args:
            text_data (str): Clinical text to analyze
            
        Returns:
            Dict: Dictionary containing identified symptoms, diagnoses, and treatments
        """
        return {
            'symptoms': self.identify_symptoms(text_data),
            'diagnoses': self.extract_diagnoses(text_data),
            'treatments': self.find_treatment_patterns(text_data)
        }

    def analyze_multiple_records(self, texts: List[str]) -> List[Dict]:
        """
        Analyze multiple clinical records.
        
        Args:
            texts (List[str]): List of clinical texts to analyze
            
        Returns:
            List[Dict]: List of analysis results for each text
        """
        results = []
        for text in texts:
            results.append(self.analyze_clinical_patterns(text))
        return results

    def get_pattern_statistics(self, analysis_results: List[Dict]) -> Dict:
        """
        Generate statistics from multiple analysis results.
        
        Args:
            analysis_results (List[Dict]): List of analysis results
            
        Returns:
            Dict: Statistics about symptoms, diagnoses, and treatments
        """
        stats = {
            'symptom_counts': defaultdict(int),
            'diagnosis_counts': defaultdict(int),
            'treatment_counts': defaultdict(int),
            'common_symptom_diagnosis_pairs': defaultdict(int),
            'common_diagnosis_treatment_pairs': defaultdict(int)
        }
        
        for result in analysis_results:
            # Count symptoms
            for symptom in result['symptoms']:
                stats['symptom_counts'][symptom['symptom']] += 1
            
            # Count diagnoses
            for diagnosis in result['diagnoses']:
                stats['diagnosis_counts'][diagnosis['diagnosis']] += 1
            
            # Count treatments
            for treatment in result['treatments']:
                stats['treatment_counts'][treatment['treatment']] += 1
            
            # Track symptom-diagnosis pairs
            for symptom in result['symptoms']:
                for diagnosis in result['diagnoses']:
                    pair = (symptom['symptom'], diagnosis['diagnosis'])
                    stats['common_symptom_diagnosis_pairs'][pair] += 1
            
            # Track diagnosis-treatment pairs
            for diagnosis in result['diagnoses']:
                for treatment in result['treatments']:
                    pair = (diagnosis['diagnosis'], treatment['treatment'])
                    stats['common_diagnosis_treatment_pairs'][pair] += 1
        
        return stats 