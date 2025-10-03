"""
Attention Analysis Tool

This module analyzes attention weights from the trained terpene synthase model
to understand what features the model focuses on for different predictions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
from prediction_tool import TerpenePredictor, PredictionResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AttentionAnalyzer:
    """Analyzes attention weights from the terpene synthase model"""
    
    def __init__(self, predictor: TerpenePredictor):
        self.predictor = predictor
        logger.info("Attention Analyzer initialized")
    
    def analyze_sequence_attention(self, sequence: str, sequence_id: str = "test") -> Dict:
        """Analyze attention weights for a single sequence"""
        
        # Get prediction with attention weights
        result = self.predictor.predict_single_sequence(sequence, sequence_id, return_attention=True)
        
        if result.attention_weights is None:
            logger.warning("No attention weights available")
            return {}
        
        # Analyze attention patterns
        attention_analysis = {
            'sequence_id': sequence_id,
            'sequence': sequence,
            'predicted_product': result.predicted_product,
            'confidence': result.confidence,
            'attention_weights': result.attention_weights,
            'attention_stats': self._compute_attention_stats(result.attention_weights),
            'top_attention_positions': self._get_top_attention_positions(result.attention_weights, sequence)
        }
        
        return attention_analysis
    
    def analyze_product_attention_patterns(self, sequences: List[str], 
                                         sequence_ids: List[str] = None,
                                         organisms: List[str] = None) -> pd.DataFrame:
        """Analyze attention patterns across multiple sequences grouped by predicted product"""
        
        if sequence_ids is None:
            sequence_ids = [f"seq_{i+1}" for i in range(len(sequences))]
        
        if organisms is None:
            organisms = [None] * len(sequences)
        
        # Get predictions with attention weights
        results = self.predictor.predict_multiple_sequences(
            sequences, sequence_ids, organisms, return_attention=True
        )
        
        # Analyze each sequence
        attention_data = []
        for result in results:
            if result.attention_weights is not None and result.predicted_product != 'ERROR':
                analysis = self.analyze_sequence_attention(result.sequence, result.sequence_id)
                attention_data.append({
                    'sequence_id': result.sequence_id,
                    'predicted_product': result.predicted_product,
                    'confidence': result.confidence,
                    'organism': result.organism or 'unknown',
                    'sequence_length': len(result.sequence),
                    'attention_mean': analysis['attention_stats']['mean'],
                    'attention_std': analysis['attention_stats']['std'],
                    'attention_max': analysis['attention_stats']['max'],
                    'attention_min': analysis['attention_stats']['min'],
                    'top_attention_positions': analysis['top_attention_positions']
                })
        
        return pd.DataFrame(attention_data)
    
    def plot_attention_heatmap(self, attention_weights: np.ndarray, sequence: str, 
                              sequence_id: str = "test", save_path: str = None):
        """Plot attention weights as a heatmap"""
        
        plt.figure(figsize=(12, 8))
        
        # Create heatmap
        sns.heatmap(attention_weights, 
                   cmap='viridis', 
                   cbar=True,
                   xticklabels=False,
                   yticklabels=False)
        
        plt.title(f'Attention Weights - {sequence_id}\nPredicted: {self.predictor.predict_single_sequence(sequence, sequence_id).predicted_product}')
        plt.xlabel('Sequence Position')
        plt.ylabel('Attention Head')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Attention heatmap saved to {save_path}")
        
        plt.show()
    
    def plot_attention_by_product(self, attention_df: pd.DataFrame, save_path: str = None):
        """Plot attention statistics grouped by predicted product"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Attention mean by product
        sns.boxplot(data=attention_df, x='predicted_product', y='attention_mean', ax=axes[0,0])
        axes[0,0].set_title('Mean Attention by Product')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Attention std by product
        sns.boxplot(data=attention_df, x='predicted_product', y='attention_std', ax=axes[0,1])
        axes[0,1].set_title('Attention Variability by Product')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Confidence vs attention mean
        sns.scatterplot(data=attention_df, x='attention_mean', y='confidence', 
                       hue='predicted_product', ax=axes[1,0])
        axes[1,0].set_title('Confidence vs Mean Attention')
        
        # Sequence length vs attention
        sns.scatterplot(data=attention_df, x='sequence_length', y='attention_mean', 
                       hue='predicted_product', ax=axes[1,1])
        axes[1,1].set_title('Sequence Length vs Mean Attention')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Attention analysis plot saved to {save_path}")
        
        plt.show()
    
    def analyze_amino_acid_attention(self, sequences: List[str], 
                                   sequence_ids: List[str] = None) -> pd.DataFrame:
        """Analyze which amino acids receive the most attention"""
        
        if sequence_ids is None:
            sequence_ids = [f"seq_{i+1}" for i in range(len(sequences))]
        
        # Get predictions with attention weights
        results = self.predictor.predict_multiple_sequences(
            sequences, sequence_ids, return_attention=True
        )
        
        # Analyze amino acid attention
        aa_attention_data = []
        
        for result in results:
            if result.attention_weights is not None and result.predicted_product != 'ERROR':
                # Average attention across all heads for each position
                avg_attention = np.mean(result.attention_weights, axis=0)
                
                # Get top attention positions
                top_positions = np.argsort(avg_attention)[-10:]  # Top 10 positions
                
                for pos in top_positions:
                    if pos < len(result.sequence):
                        aa_attention_data.append({
                            'sequence_id': result.sequence_id,
                            'predicted_product': result.predicted_product,
                            'position': pos,
                            'amino_acid': result.sequence[pos],
                            'attention_score': avg_attention[pos]
                        })
        
        return pd.DataFrame(aa_attention_data)
    
    def plot_amino_acid_attention(self, aa_attention_df: pd.DataFrame, save_path: str = None):
        """Plot attention scores by amino acid"""
        
        plt.figure(figsize=(12, 8))
        
        # Group by amino acid and calculate mean attention
        aa_means = aa_attention_df.groupby('amino_acid')['attention_score'].mean().sort_values(ascending=False)
        
        # Create bar plot
        sns.barplot(x=aa_means.index, y=aa_means.values)
        plt.title('Mean Attention Score by Amino Acid')
        plt.xlabel('Amino Acid')
        plt.ylabel('Mean Attention Score')
        plt.xticks(rotation=45)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Amino acid attention plot saved to {save_path}")
        
        plt.show()
    
    def _compute_attention_stats(self, attention_weights: np.ndarray) -> Dict:
        """Compute statistics for attention weights"""
        
        return {
            'mean': float(np.mean(attention_weights)),
            'std': float(np.std(attention_weights)),
            'max': float(np.max(attention_weights)),
            'min': float(np.min(attention_weights)),
            'shape': attention_weights.shape
        }
    
    def _get_top_attention_positions(self, attention_weights: np.ndarray, sequence: str, 
                                   top_k: int = 10) -> List[Dict]:
        """Get top attention positions with their amino acids"""
        
        # Average attention across all heads
        avg_attention = np.mean(attention_weights, axis=0)
        
        # Get top positions
        top_positions = np.argsort(avg_attention)[-top_k:][::-1]
        
        top_attention = []
        for pos in top_positions:
            if pos < len(sequence):
                top_attention.append({
                    'position': int(pos),
                    'amino_acid': sequence[pos],
                    'attention_score': float(avg_attention[pos])
                })
        
        return top_attention
    
    def generate_attention_report(self, sequences: List[str], 
                                sequence_ids: List[str] = None,
                                output_dir: str = "attention_analysis") -> str:
        """Generate comprehensive attention analysis report"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        logger.info("Generating comprehensive attention analysis report...")
        
        # Analyze attention patterns
        attention_df = self.analyze_product_attention_patterns(sequences, sequence_ids)
        
        # Analyze amino acid attention
        aa_attention_df = self.analyze_amino_acid_attention(sequences, sequence_ids)
        
        # Save data
        attention_df.to_csv(output_path / "attention_patterns.csv", index=False)
        aa_attention_df.to_csv(output_path / "amino_acid_attention.csv", index=False)
        
        # Generate plots
        self.plot_attention_by_product(attention_df, str(output_path / "attention_by_product.png"))
        self.plot_amino_acid_attention(aa_attention_df, str(output_path / "amino_acid_attention.png"))
        
        # Generate summary report
        report_path = output_path / "attention_report.txt"
        with open(report_path, 'w') as f:
            f.write("TERPENE SYNTHASE ATTENTION ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total sequences analyzed: {len(attention_df)}\n")
            f.write(f"Unique products predicted: {attention_df['predicted_product'].nunique()}\n\n")
            
            f.write("PRODUCT DISTRIBUTION:\n")
            product_counts = attention_df['predicted_product'].value_counts()
            for product, count in product_counts.items():
                f.write(f"  {product}: {count} sequences\n")
            
            f.write(f"\nATTENTION STATISTICS:\n")
            f.write(f"  Mean attention score: {attention_df['attention_mean'].mean():.4f}\n")
            f.write(f"  Attention std: {attention_df['attention_std'].mean():.4f}\n")
            f.write(f"  Max attention: {attention_df['attention_max'].max():.4f}\n")
            f.write(f"  Min attention: {attention_df['attention_min'].min():.4f}\n")
            
            f.write(f"\nTOP ATTENTION AMINO ACIDS:\n")
            aa_means = aa_attention_df.groupby('amino_acid')['attention_score'].mean().sort_values(ascending=False)
            for aa, score in aa_means.head(10).items():
                f.write(f"  {aa}: {score:.4f}\n")
        
        logger.info(f"Attention analysis report generated in {output_path}")
        return str(output_path)

def main():
    """Command-line interface for attention analysis"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze attention weights from terpene synthase model")
    parser.add_argument("--input", "-i", required=True, help="Input FASTA file")
    parser.add_argument("--output-dir", "-o", default="attention_analysis", help="Output directory")
    parser.add_argument("--model-path", default="data/cache/best_model.pth", help="Path to trained model")
    parser.add_argument("--pipeline-path", default="data/cache/terpene_predictor_pipeline.pkl", help="Path to pipeline data")
    
    args = parser.parse_args()
    
    # Initialize predictor and analyzer
    predictor = TerpenePredictor(args.model_path, args.pipeline_path)
    analyzer = AttentionAnalyzer(predictor)
    
    # Load sequences
    sequences, sequence_ids = analyzer.predictor._parse_fasta(args.input)
    
    # Generate analysis report
    output_path = analyzer.generate_attention_report(sequences, sequence_ids, args.output_dir)
    
    print(f"\nAttention analysis complete!")
    print(f"Results saved to: {output_path}")
    print(f"Files generated:")
    print(f"  - attention_patterns.csv")
    print(f"  - amino_acid_attention.csv") 
    print(f"  - attention_by_product.png")
    print(f"  - amino_acid_attention.png")
    print(f"  - attention_report.txt")

if __name__ == "__main__":
    main()
