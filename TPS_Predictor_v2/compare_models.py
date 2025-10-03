#!/usr/bin/env python3
"""
Comparison script for SaProt, ProtT5, and TerpeneMiner encoders.
Tests attention patterns, prediction diversity, and performance.
"""

import logging
import torch
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Import our encoders
from models.saprot_encoder import SaProtEncoder
from models.prott5_encoder import ProtT5Encoder
from models.terpeneminer_encoder import TerpeneMinerEncoder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def analyze_attention_patterns(attention_weights: np.ndarray, sequence: str, model_name: str) -> dict:
    """Analyze attention patterns for a given model and sequence"""
    if attention_weights is None:
        return {"error": "No attention weights available"}
    
    num_heads = attention_weights.shape[1]
    seq_len = attention_weights.shape[2]
    
    # Average attention across heads for each token (attention received)
    attention_received = attention_weights.mean(axis=1).sum(axis=1).squeeze()
    
    # Methionine attention (assuming it's the first actual amino acid after <cls>)
    methionine_attention = attention_received[1] if seq_len > 1 else 0.0
    
    # Mean attention across all actual amino acids (excluding <cls> and <eos>)
    mean_attention = attention_received[1:seq_len-1].mean() if seq_len > 2 else 0.0
    
    # Methionine bias ratio
    methionine_bias = methionine_attention / mean_attention if mean_attention > 0 else float('inf')
    
    # Attention entropy (how spread out attention is)
    attention_entropy = np.array([entropy(attention_weights[0, h, i, :]) for h in range(num_heads) for i in range(seq_len)])
    mean_attention_entropy = attention_entropy.mean()
    
    # Top 10 attended positions
    top_indices = np.argsort(attention_received)[::-1][:10]
    
    return {
        "model_name": model_name,
        "attention_shape": attention_weights.shape,
        "methionine_attention": float(methionine_attention),
        "mean_attention": float(mean_attention),
        "methionine_bias": float(methionine_bias),
        "attention_entropy": float(mean_attention_entropy),
        "top_attended_positions": top_indices.tolist(),
        "attention_received": attention_received.tolist()
    }

def test_model_attention(model, model_name: str, test_sequence: str, device: torch.device) -> dict:
    """Test attention patterns for a given model"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {model_name}")
    logger.info(f"{'='*60}")
    
    try:
        # Test basic encoding
        logger.info("Testing basic sequence encoding...")
        embedding_result = model.encode_sequence(test_sequence)
        
        if embedding_result:
            logger.info(f"✓ Successfully encoded sequence")
            logger.info(f"  Embedding shape: {embedding_result.embedding.shape}")
            logger.info(f"  Sequence length: {embedding_result.sequence_length}")
        else:
            logger.error("✗ Failed to encode sequence")
            return {"error": "Failed to encode sequence"}
        
        # Test attention weight extraction
        logger.info("Testing attention weight extraction...")
        attention_result = model.encode_sequence(test_sequence, return_attention=True)
        
        if attention_result and attention_result.attention_weights is not None:
            logger.info(f"✓ Successfully extracted attention weights")
            logger.info(f"  Attention shape: {attention_result.attention_weights.shape}")
            
            # Analyze attention patterns
            analysis = analyze_attention_patterns(
                attention_result.attention_weights, 
                test_sequence, 
                model_name
            )
            
            logger.info(f"\n--- Attention Analysis ---")
            logger.info(f"Methionine attention: {analysis['methionine_attention']:.4f}")
            logger.info(f"Mean attention: {analysis['mean_attention']:.4f}")
            logger.info(f"Methionine bias ratio: {analysis['methionine_bias']:.2f}")
            logger.info(f"Attention entropy: {analysis['attention_entropy']:.4f}")
            
            logger.info(f"\nTop 5 attended positions:")
            for i, pos in enumerate(analysis['top_attended_positions'][:5]):
                logger.info(f"  {i+1}. Position {pos}: {analysis['attention_received'][pos]:.4f}")
            
            return analysis
            
        else:
            logger.error("✗ Failed to extract attention weights")
            return {"error": "Failed to extract attention weights"}
            
    except Exception as e:
        logger.error(f"✗ Error testing {model_name}: {e}")
        return {"error": str(e)}

def create_comparison_visualization(results: dict, output_dir: Path):
    """Create visualization comparing attention patterns across models"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter out error results
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    
    if len(valid_results) < 2:
        logger.warning("Not enough valid results for comparison visualization")
        return
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Methionine bias comparison
    models = list(valid_results.keys())
    methionine_biases = [valid_results[model]['methionine_bias'] for model in models]
    
    axes[0, 0].bar(models, methionine_biases, color=['red', 'blue', 'green'][:len(models)])
    axes[0, 0].set_title('Methionine Attention Bias')
    axes[0, 0].set_ylabel('Bias Ratio')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Attention entropy comparison
    attention_entropies = [valid_results[model]['attention_entropy'] for model in models]
    
    axes[0, 1].bar(models, attention_entropies, color=['red', 'blue', 'green'][:len(models)])
    axes[0, 1].set_title('Attention Entropy (Higher = More Spread)')
    axes[0, 1].set_ylabel('Entropy')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Attention received patterns
    for i, (model, result) in enumerate(valid_results.items()):
        axes[1, 0].plot(result['attention_received'], label=model, alpha=0.7)
    
    axes[1, 0].set_title('Attention Received by Position')
    axes[1, 0].set_xlabel('Position')
    axes[1, 0].set_ylabel('Attention Score')
    axes[1, 0].legend()
    
    # 4. Summary statistics
    summary_data = []
    for model, result in valid_results.items():
        summary_data.append({
            'Model': model,
            'Methionine Bias': result['methionine_bias'],
            'Attention Entropy': result['attention_entropy'],
            'Max Attention': max(result['attention_received']),
            'Min Attention': min(result['attention_received'])
        })
    
    # Create summary table
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    
    table_data = []
    headers = ['Model', 'Methionine Bias', 'Attention Entropy', 'Max Attn', 'Min Attn']
    
    for data in summary_data:
        table_data.append([
            data['Model'],
            f"{data['Methionine Bias']:.2f}",
            f"{data['Attention Entropy']:.3f}",
            f"{data['Max Attention']:.3f}",
            f"{data['Min Attention']:.3f}"
        ])
    
    table = axes[1, 1].table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    axes[1, 1].set_title('Summary Statistics')
    
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Comparison visualization saved to {output_dir / 'model_comparison.png'}")

def main():
    """Main comparison function"""
    logger.info("Starting comprehensive model comparison...")
    
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Test sequence
    test_sequence = "MSITFNLKIAPFSGPGIQRSKETFPATEIQITASTKSTMTTKCSFNASTDFMGKLREKVGGKADKPPVVIHPVDISSNLCMIDTLQSLGVDRYFQSEINTLLEHTYRLWKEKKKNIIFKDVSCCAIAFRLLREKGYQVSSDKLAPFADYRIRDVATILELYRASQARLYEDEHTLEKLHDWSSNLLKQHLLNGSIPDHKLHKQVEYFLKNYHGILDRVAVRRSLDLYNINHHHRIPDVADGFPKEDFLEYSMQDFNICQAQQQEELHQLQRWYADCRLDTLNYGRDVVRIANFLTSAIFGEPEFSDARLAFAKHIILVTRIDDFFDHGGSREESYKILDLVQEWKEKPAEEYGSKEVEILFTAVYNTVNDLAEKAHIEQGRCVKPLLIKLWVEILTSFKKELDSWTEETALTLDEYLSSSWVSIGCRICILNSLQYLGIKLSEEMLSSQECTDLCRHVSSVDRLLNDVQTFKKERLENTINSVGLQLAAHKGERAMTEEDAMSKIKEMADYHRRKLMQIVYKEGTVFPRECKDVFLRVCRIGYYLYSSGDEFTSPQQMKEDMKSLVYQPVKIHPLEAINV"
    
    logger.info(f"Test sequence length: {len(test_sequence)} amino acids")
    
    # Initialize models
    models = {}
    
    try:
        logger.info("Initializing SaProt model...")
        models['SaProt'] = SaProtEncoder(model_name="westlake-repl/SaProt_650M_AF2", device=device)
    except Exception as e:
        logger.error(f"Failed to initialize SaProt: {e}")
    
    try:
        logger.info("Initializing ProtT5 model...")
        models['ProtT5'] = ProtT5Encoder(device=device)
    except Exception as e:
        logger.error(f"Failed to initialize ProtT5: {e}")
    
    try:
        logger.info("Initializing TerpeneMiner model...")
        models['TerpeneMiner'] = TerpeneMinerEncoder(device=device)
    except Exception as e:
        logger.error(f"Failed to initialize TerpeneMiner: {e}")
    
    if not models:
        logger.error("No models could be initialized. Aborting comparison.")
        return
    
    # Test each model
    results = {}
    for model_name, model in models.items():
        results[model_name] = test_model_attention(model, model_name, test_sequence, device)
    
    # Create comparison visualization
    output_dir = Path("model_comparison_results")
    create_comparison_visualization(results, output_dir)
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("COMPARISON SUMMARY")
    logger.info(f"{'='*60}")
    
    for model_name, result in results.items():
        if "error" not in result:
            logger.info(f"\n{model_name}:")
            logger.info(f"  Methionine bias: {result['methionine_bias']:.2f}")
            logger.info(f"  Attention entropy: {result['attention_entropy']:.3f}")
            logger.info(f"  Embedding dimension: {result['attention_shape'][-1]}")
        else:
            logger.info(f"\n{model_name}: ERROR - {result['error']}")
    
    # Recommendations
    logger.info(f"\n{'='*60}")
    logger.info("RECOMMENDATIONS")
    logger.info(f"{'='*60}")
    
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    
    if valid_results:
        # Find model with lowest methionine bias
        best_bias_model = min(valid_results.keys(), key=lambda x: valid_results[x]['methionine_bias'])
        logger.info(f"✓ Lowest methionine bias: {best_bias_model} ({valid_results[best_bias_model]['methionine_bias']:.2f})")
        
        # Find model with highest attention entropy (most spread)
        best_entropy_model = max(valid_results.keys(), key=lambda x: valid_results[x]['attention_entropy'])
        logger.info(f"✓ Most diverse attention: {best_entropy_model} ({valid_results[best_entropy_model]['attention_entropy']:.3f})")
        
        logger.info(f"\nBased on attention analysis, {best_bias_model} shows the most realistic attention patterns.")
    
    logger.info("Model comparison completed!")

if __name__ == "__main__":
    main()
