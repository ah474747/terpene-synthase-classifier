#!/usr/bin/env python3
"""
Simple attention analysis script to debug the SaProt encoder attention extraction.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from Bio import SeqIO

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_attention_extraction():
    """Test attention extraction with a simple sequence"""
    
    # Test sequence
    test_sequence = "MSITFNLKIAPFSGPGIQRSKETFPATEIQITASTKSTMTTKCSFNASTDFMGKLREKVGGKADKPPVVIHPVDISSNLCMIDTLQSLGVDRYFQSEINTLLEHTYRLWKEKKKNIIFKDVSCCAIAFRLLREKGYQVSSDKLAPFADYRIRDVATILELYRASQARLYEDEHTLEKLHDWSSNLLKQHLLNGSIPDHKLHKQVEYFLKNYHGILDRVAVRRSLDLYNINHHHRIPDVADGFPKEDFLEYSMQDFNICQAQQQEELHQLQRWYADCRLDTLNYGRDVVRIANFLTSAIFGEPEFSDARLAFAKHIILVTRIDDFFDHGGSREESYKILDLVQEWKEKPAEEYGSKEVEILFTAVYNTVNDLAEKAHIEQGRCVKPLLIKLWVEILTSFKKELDSWTEETALTLDEYLSSSWVSIGCRICILNSLQYLGIKLSEEMLSSQECTDLCRHVSSVDRLLNDVQTFKKERLENTINSVGLQLAAHKGERAMTEEDAMSKIKEMADYHRRKLMQIVYKEGTVFPRECKDVFLRVCRIGYYLYSSGDEFTSPQQMKEDMKSLVYQPVKIHPLEAINV"
    
    logger.info("Testing attention extraction with SaProt encoder...")
    
    try:
        from models.saprot_encoder import SaProtEncoder
        
        # Initialize encoder
        encoder = SaProtEncoder()
        
        # Test basic encoding
        logger.info("Testing basic sequence encoding...")
        result = encoder.encode_sequence(test_sequence)
        
        if result is None:
            logger.error("Failed to encode sequence")
            return False
            
        logger.info(f"Successfully encoded sequence. Embedding shape: {result.embedding.shape}")
        
        # Test attention extraction
        logger.info("Testing attention weight extraction...")
        attention_result = encoder.encode_sequence(test_sequence, return_attention=True)
        
        if attention_result is None:
            logger.error("Failed to extract attention weights")
            return False
            
        if attention_result.attention_weights is None:
            logger.warning("Attention weights are None - model may not support attention extraction")
            return False
            
        logger.info(f"Successfully extracted attention weights. Shape: {attention_result.attention_weights.shape}")
        
        # Analyze attention patterns
        analyze_attention_patterns(attention_result.attention_weights, test_sequence)
        
        return True
        
    except Exception as e:
        logger.error(f"Error during attention extraction: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_attention_patterns(attention_weights, sequence):
    """Analyze attention patterns"""
    
    logger.info("Analyzing attention patterns...")
    
    # attention_weights shape should be (num_layers, num_heads, seq_len, seq_len)
    if len(attention_weights.shape) != 4:
        logger.error(f"Unexpected attention weights shape: {attention_weights.shape}")
        return
    
    num_layers, num_heads, seq_len, _ = attention_weights.shape
    logger.info(f"Attention weights shape: {attention_weights.shape}")
    logger.info(f"Number of layers: {num_layers}, Number of heads: {num_heads}, Sequence length: {seq_len}")
    
    # Average attention across heads for each layer
    avg_attention = np.mean(attention_weights, axis=1)  # Shape: (num_layers, seq_len, seq_len)
    
    # Analyze attention patterns
    logger.info("\n=== ATTENTION ANALYSIS ===")
    
    # 1. Self-attention (diagonal elements)
    self_attention = np.diagonal(avg_attention, axis1=1, axis2=2)  # Shape: (num_layers, seq_len)
    logger.info(f"Self-attention statistics:")
    logger.info(f"  Mean across layers: {np.mean(self_attention, axis=0)[:10]}...")  # First 10 positions
    logger.info(f"  Std across layers: {np.std(self_attention, axis=0)[:10]}...")
    
    # 2. Attention entropy (measure of attention spread)
    attention_entropy = -np.sum(avg_attention * np.log(avg_attention + 1e-10), axis=2)  # Shape: (num_layers, seq_len)
    logger.info(f"Attention entropy (higher = more spread):")
    logger.info(f"  Mean across layers: {np.mean(attention_entropy, axis=0)[:10]}...")
    
    # 3. Most attended positions
    for layer_idx in range(min(3, num_layers)):  # Analyze first 3 layers
        layer_attention = avg_attention[layer_idx]
        
        # Find positions that receive most attention
        attention_received = np.sum(layer_attention, axis=0)  # Sum over all source positions
        top_attended = np.argsort(attention_received)[-10:]  # Top 10 most attended positions
        
        logger.info(f"\nLayer {layer_idx} - Top 10 most attended positions:")
        for pos in top_attended:
            aa = sequence[pos] if pos < len(sequence) else 'X'
            logger.info(f"  Position {pos}: {aa} (attention: {attention_received[pos]:.4f})")
    
    # 4. Create visualization
    create_attention_visualization(avg_attention, sequence)

def create_attention_visualization(avg_attention, sequence):
    """Create attention visualization"""
    
    logger.info("Creating attention visualization...")
    
    # Create output directory
    output_dir = Path("attention_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Plot attention matrix for first layer
    plt.figure(figsize=(12, 10))
    
    # Show first 100 positions to keep visualization manageable
    max_pos = min(100, len(sequence))
    attention_subset = avg_attention[0, :max_pos, :max_pos]
    
    plt.imshow(attention_subset, cmap='Blues', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.title('Attention Matrix - Layer 0 (First 100 positions)')
    plt.xlabel('Source Position')
    plt.ylabel('Target Position')
    
    # Add amino acid labels
    aa_labels = [sequence[i] for i in range(max_pos)]
    plt.xticks(range(0, max_pos, 10), [aa_labels[i] for i in range(0, max_pos, 10)])
    plt.yticks(range(0, max_pos, 10), [aa_labels[i] for i in range(0, max_pos, 10)])
    
    plt.tight_layout()
    plt.savefig(output_dir / "attention_matrix_layer0.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot attention received by each position
    plt.figure(figsize=(15, 6))
    
    attention_received = np.sum(avg_attention[0], axis=0)
    positions = range(len(attention_received))
    
    plt.plot(positions, attention_received, alpha=0.7)
    plt.title('Attention Received by Each Position - Layer 0')
    plt.xlabel('Position')
    plt.ylabel('Total Attention Received')
    
    # Highlight top 10 positions
    top_10_indices = np.argsort(attention_received)[-10:]
    plt.scatter(top_10_indices, attention_received[top_10_indices], 
               color='red', s=50, zorder=5, label='Top 10')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "attention_received.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot attention patterns across layers
    plt.figure(figsize=(12, 8))
    
    num_layers = avg_attention.shape[0]
    for layer_idx in range(min(6, num_layers)):  # Show first 6 layers
        layer_attention_received = np.sum(avg_attention[layer_idx], axis=0)
        plt.plot(positions, layer_attention_received, alpha=0.7, label=f'Layer {layer_idx}')
    
    plt.title('Attention Patterns Across Layers')
    plt.xlabel('Position')
    plt.ylabel('Total Attention Received')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "attention_across_layers.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualizations saved to {output_dir}/")

def main():
    """Main function"""
    logger.info("Starting attention analysis...")
    
    success = test_attention_extraction()
    
    if success:
        logger.info("Attention analysis completed successfully!")
    else:
        logger.error("Attention analysis failed!")

if __name__ == "__main__":
    main()
