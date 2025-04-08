import pandas as pd
import numpy as np
from utils import load_and_preprocess_data, load_classes
import visualizations as viz
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.manifold import TSNE
import plotly.express as px

def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    train_df, test_df = load_and_preprocess_data()
    classes = load_classes()
    
    # Create output directory for plots
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Plot class distribution
    print("Generating class distribution plot...")
    plt_dist = viz.plot_class_distribution(train_df)
    plt_dist.savefig('plots/class_distribution.png')
    plt_dist.close()
    
    # Create word cloud
    print("Generating word cloud...")
    plt_cloud = viz.create_wordcloud(train_df['processed_text'])
    plt_cloud.savefig('plots/wordcloud.png')
    plt_cloud.close()
    
    # Plot document lengths
    print("Generating document length plots...")
    plt_lengths = viz.plot_document_lengths(train_df)
    plt_lengths.savefig('plots/doc_lengths_by_class.png')
    plt_lengths.close()
    
    # Plot length distribution
    plt_dist = viz.plot_length_distribution(train_df)
    plt_dist.savefig('plots/doc_length_distribution.png')
    plt_dist.close()
    
    # Plot top terms by class
    print("Generating top terms plot...")
    plt_terms = viz.plot_top_terms_by_class(train_df)
    plt_terms.savefig('plots/top_terms_by_class.png')
    plt_terms.close()
    
    # Load pre-trained model for embeddings
    print("Generating document embeddings using pre-trained model...")
    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    
    # Generate embeddings for a sample of documents
    sample_size = min(1000, len(train_df))
    sample_df = train_df.sample(n=sample_size, random_state=42)
    
    # Get embeddings
    embeddings = []
    for text in sample_df['processed_text']:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
    
    # Convert list of embeddings to numpy array
    embeddings = np.array(embeddings)
    
    # Reduce dimensionality with t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create interactive scatter plot
    plot_df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'Class': sample_df['label'].reset_index(drop=True)
    })
    
    fig = px.scatter(plot_df, x='x', y='y', color='Class',
                    title='Document Embeddings Visualization (t-SNE)')
    fig.write_html('plots/embeddings_visualization.html')
    
    print("Analysis complete! Check the 'plots' directory for visualizations.")

if __name__ == "__main__":
    main() 