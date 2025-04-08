# Medical Text Analysis and Visualization

This project analyzes medical transcriptions data using Natural Language Processing (NLP) and creates various visualizations to understand the patterns and distributions in the medical text data.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── utils.py
├── visualizations.py
├── analyze_medical_data.py
└── data/
    ├── mtsamples.csv
    ├── clinical-stopwords.txt
    ├── vocab.txt
    ├── X.csv
    ├── classes.txt
    ├── train.csv
    └── test.csv
```

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Analysis

To run the complete analysis and generate all visualizations:

```bash
python analyze_medical_data.py
```

This will create a `plots` directory containing the following visualizations:
- Class distribution plot
- Word cloud of medical documents
- Document length distribution by class
- Overall document length distribution
- Top terms by class
- Interactive document embeddings visualization (t-SNE)

## Features

1. **Data Preprocessing**
   - Text cleaning and normalization
   - Clinical stopwords removal
   - Tokenization

2. **Visualizations**
   - Distribution of medical document classes
   - Word cloud visualization
   - Document length analysis
   - Top terms analysis by class
   - Document embeddings visualization using Bio_ClinicalBERT

3. **Pre-trained Model Integration**
   - Uses Bio_ClinicalBERT for generating document embeddings
   - t-SNE dimensionality reduction for visualization

## Requirements

- Python 3.7+
- See requirements.txt for all Python package dependencies

## Data Sources

- Medical transcriptions dataset from Kaggle
- Clinical stopwords from Dr. Kavita Ganesan's clinical-concepts repository
- SNMI vocabulary data 