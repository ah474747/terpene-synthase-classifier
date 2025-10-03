#!/usr/bin/env python3
"""
Comprehensive attention analysis across multiple sequences to understand model behavior.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import logging
from Bio import SeqIO
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_multiple_sequences():
    """Analyze attention patterns across multiple sequences"""
    
    logger.info("Starting comprehensive attention analysis...")
    
    try:
        from models.saprot_encoder import SaProtEncoder
        from prediction_tool import TerpenePredictor
        
        # Initialize components
        encoder = SaProtEncoder()
        predictor = TerpenePredictor()
        
        # Load sample sequences
        sequences = []
        sequence_ids = []
        
        with open('sample_sequences.fasta', 'r') as f:
            for record in SeqIO.parse(f, 'fasta'):
                sequences.append(str(record.seq))
                sequence_ids.append(record.id)
        
        logger.info(f"Loaded {len(sequences)} sequences for analysis")
        
        # Analyze each sequence
        attention_data = []
        prediction_data = []
        
        for i, (seq_id, sequence) in enumerate(zip(sequence_ids, sequences)):
            logger.info(f"Analyzing sequence {i+1}/{len(sequences)}: {seq_id}")
            
            try:
                # Get prediction
                prediction = predictor.predict_single_sequence(sequence, seq_id)
                prediction_data.append({
                    'sequence_id': seq_id,
                    'predicted_product': prediction.predicted_product,
                    'confidence': prediction.confidence,
                    'sequence_length': len(sequence)
                })
                
                # Get attention weights
                attention_result = encoder.encode_sequence(sequence, return_attention=True)
                
                if attention_result and attention_result.attention_weights is not None:
                    attention_weights = attention_result.attention_weights
                    
                    # Analyze attention patterns
                    attention_stats = analyze_sequence_attention(
                        attention_weights, sequence, seq_id, prediction.predicted_product
                    )
                    attention_data.append(attention_stats)
                
            except Exception as e:
                logger.error(f"Failed to analyze sequence {seq_id}: {e}")
                continue
        
        # Create comprehensive analysis
        create_comprehensive_analysis(attention_data, prediction_data)
        
        logger.info("Comprehensive attention analysis completed!")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

def analyze_sequence_attention(attention_weights, sequence, seq_id, predicted_product):
    """Analyze attention patterns for a single sequence"""
    
    # attention_weights shape: (num_layers, num_heads, seq_len, seq_len)
    num_layers, num_heads, seq_len, _ = attention_weights.shape
    
    # Average attention across heads for each layer
    avg_attention = np.mean(attention_weights, axis=1)  # Shape: (num_layers, seq_len, seq_len)
    
    # Calculate attention statistics
    attention_received = np.sum(avg_attention[0], axis=0)  # Total attention received by each position
    
    # Find most attended positions
    top_attended_indices = np.argsort(attention_received)[-10:]  # Top 10
    top_attended_aa = [sequence[i] if i < len(sequence) else 'X' for i in top_attended_indices]
    top_attended_scores = attention_received[top_attended_indices]
    
    # Calculate attention entropy (measure of attention spread)
    attention_entropy = -np.sum(avg_attention[0] * np.log(avg_attention[0] + 1e-10), axis=1)
    mean_entropy = np.mean(attention_entropy)
    
    # Calculate self-attention (diagonal elements)
    self_attention = np.diagonal(avg_attention[0])
    mean_self_attention = np.mean(self_attention)
    
    # Analyze amino acid attention patterns
    aa_attention = defaultdict(list)
    for i, aa in enumerate(sequence):
        if i < len(attention_received):
            aa_attention[aa].append(attention_received[i])
    
    # Calculate mean attention per amino acid
    aa_mean_attention = {aa: np.mean(scores) for aa, scores in aa_attention.items()}
    
    return {
        'sequence_id': seq_id,
        'predicted_product': predicted_product,
        'sequence_length': len(sequence),
        'num_layers': num_layers,
        'num_heads': num_heads,
        'mean_attention_entropy': mean_entropy,
        'mean_self_attention': mean_self_attention,
        'max_attention': np.max(attention_received),
        'min_attention': np.min(attention_received),
        'std_attention': np.std(attention_received),
        'top_attended_positions': top_attended_indices.tolist(),
        'top_attended_aa': top_attended_aa,
        'top_attended_scores': top_attended_scores.tolist(),
        'aa_mean_attention': aa_mean_attention,
        'attention_received': attention_received.tolist()
    }

def create_comprehensive_analysis(attention_data, prediction_data):
    """Create comprehensive analysis and visualizations"""
    
    logger.info("Creating comprehensive analysis...")
    
    # Create output directory
    output_dir = Path("comprehensive_attention_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Convert to DataFrames
    attention_df = pd.DataFrame(attention_data)
    prediction_df = pd.DataFrame(prediction_data)
    
    # Save raw data
    attention_df.to_csv(output_dir / "attention_analysis_data.csv", index=False)
    prediction_df.to_csv(output_dir / "prediction_data.csv", index=False)
    
    # Create visualizations
    create_attention_visualizations(attention_df, prediction_df, output_dir)
    
    # Create summary statistics
    create_summary_statistics(attention_df, prediction_df, output_dir)

def create_attention_visualizations(attention_df, prediction_df, output_dir):
    """Create comprehensive visualizations"""
    
    logger.info("Creating visualizations...")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Attention entropy by predicted product
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=attention_df, x='predicted_product', y='mean_attention_entropy')
    plt.title('Attention Entropy by Predicted Product')
    plt.xlabel('Predicted Product')
    plt.ylabel('Mean Attention Entropy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "attention_entropy_by_product.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Attention statistics distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Max attention
    axes[0,0].hist(attention_df['max_attention'], bins=20, alpha=0.7)
    axes[0,0].set_title('Distribution of Maximum Attention')
    axes[0,0].set_xlabel('Maximum Attention')
    axes[0,0].set_ylabel('Frequency')
    
    # Attention entropy
    axes[0,1].hist(attention_df['mean_attention_entropy'], bins=20, alpha=0.7)
    axes[0,1].set_title('Distribution of Attention Entropy')
    axes[0,1].set_xlabel('Mean Attention Entropy')
    axes[0,1].set_ylabel('Frequency')
    
    # Self-attention
    axes[1,0].hist(attention_df['mean_self_attention'], bins=20, alpha=0.7)
    axes[1,0].set_title('Distribution of Self-Attention')
    axes[1,0].set_xlabel('Mean Self-Attention')
    axes[1,0].set_ylabel('Frequency')
    
    # Attention std
    axes[1,1].hist(attention_df['std_attention'], bins=20, alpha=0.7)
    axes[1,1].set_title('Distribution of Attention Standard Deviation')
    axes[1,1].set_xlabel('Attention Std')
    axes[1,1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(output_dir / "attention_statistics_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Amino acid attention patterns
    create_amino_acid_attention_plot(attention_df, output_dir)
    
    # 4. Sequence length vs attention patterns
    plt.figure(figsize=(12, 8))
    plt.scatter(attention_df['sequence_length'], attention_df['mean_attention_entropy'], 
               alpha=0.7, s=60)
    plt.title('Sequence Length vs Attention Entropy')
    plt.xlabel('Sequence Length')
    plt.ylabel('Mean Attention Entropy')
    plt.tight_layout()
    plt.savefig(output_dir / "sequence_length_vs_attention.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_amino_acid_attention_plot(attention_df, output_dir):
    """Create amino acid attention analysis"""
    
    # Collect all amino acid attention scores
    all_aa_attention = defaultdict(list)
    
    for _, row in attention_df.iterrows():
        aa_attention = row['aa_mean_attention']
        for aa, score in aa_attention.items():
            all_aa_attention[aa].append(score)
    
    # Create amino acid attention plot
    aa_names = list(all_aa_attention.keys())
    aa_scores = [np.mean(all_aa_attention[aa]) for aa in aa_names]
    aa_stds = [np.std(all_aa_attention[aa]) for aa in aa_names]
    
    plt.figure(figsize=(15, 8))
    bars = plt.bar(aa_names, aa_scores, yerr=aa_stds, capsize=5, alpha=0.7)
    plt.title('Mean Attention Scores by Amino Acid')
    plt.xlabel('Amino Acid')
    plt.ylabel('Mean Attention Score')
    plt.xticks(rotation=0)
    
    # Color bars by amino acid properties
    colors = ['red' if aa in 'DE' else 'blue' if aa in 'KR' else 'green' if aa in 'ST' else 'gray' 
              for aa in aa_names]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.tight_layout()
    plt.savefig(output_dir / "amino_acid_attention.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_statistics(attention_df, prediction_df, output_dir):
    """Create summary statistics"""
    
    logger.info("Creating summary statistics...")
    
    summary_stats = {
        'total_sequences_analyzed': len(attention_df),
        'unique_predicted_products': attention_df['predicted_product'].nunique(),
        'predicted_products': attention_df['predicted_product'].value_counts().to_dict(),
        'attention_statistics': {
            'mean_entropy': attention_df['mean_attention_entropy'].mean(),
            'std_entropy': attention_df['mean_attention_entropy'].std(),
            'mean_max_attention': attention_df['max_attention'].mean(),
            'std_max_attention': attention_df['max_attention'].std(),
            'mean_self_attention': attention_df['mean_self_attention'].mean(),
            'std_self_attention': attention_df['mean_self_attention'].std(),
        },
        'sequence_statistics': {
            'mean_length': attention_df['sequence_length'].mean(),
            'std_length': attention_df['sequence_length'].std(),
            'min_length': attention_df['sequence_length'].min(),
            'max_length': attention_df['sequence_length'].max(),
        }
    }
    
    # Save summary
    import json
    with open(output_dir / "summary_statistics.json", 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("COMPREHENSIVE ATTENTION ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total sequences analyzed: {summary_stats['total_sequences_analyzed']}")
    print(f"Unique predicted products: {summary_stats['unique_predicted_products']}")
    print(f"Predicted products distribution: {summary_stats['predicted_products']}")
    print(f"\nAttention Statistics:")
    print(f"  Mean entropy: {summary_stats['attention_statistics']['mean_entropy']:.4f}")
    print(f"  Mean max attention: {summary_stats['attention_statistics']['mean_max_attention']:.4f}")
    print(f"  Mean self-attention: {summary_stats['attention_statistics']['mean_self_attention']:.4f}")
    print(f"\nSequence Statistics:")
    print(f"  Mean length: {summary_stats['sequence_statistics']['mean_length']:.1f}")
    print(f"  Length range: {summary_stats['sequence_statistics']['min_length']}-{summary_stats['sequence_statistics']['max_length']}")
    print("="*60)

def main():
    """Main function"""
    analyze_multiple_sequences()

if __name__ == "__main__":
    main()
