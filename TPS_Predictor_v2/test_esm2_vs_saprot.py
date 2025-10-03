#!/usr/bin/env python3
"""
Test ESM2 model vs SaProt model for attention patterns to see if the methionine bias is real or an artifact.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from transformers import EsmTokenizer, EsmForMaskedLM

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_esm2_attention():
    """Test ESM2 model attention patterns"""
    
    logger.info("Testing ESM2 model attention patterns...")
    
    # Test sequence
    test_sequence = "MSITFNLKIAPFSGPGIQRSKETFPATEIQITASTKSTMTTKCSFNASTDFMGKLREKVGGKADKPPVVIHPVDISSNLCMIDTLQSLGVDRYFQSEINTLLEHTYRLWKEKKKNIIFKDVSCCAIAFRLLREKGYQVSSDKLAPFADYRIRDVATILELYRASQARLYEDEHTLEKLHDWSSNLLKQHLLNGSIPDHKLHKQVEYFLKNYHGILDRVAVRRSLDLYNINHHHRIPDVADGFPKEDFLEYSMQDFNICQAQQQEELHQLQRWYADCRLDTLNYGRDVVRIANFLTSAIFGEPEFSDARLAFAKHIILVTRIDDFFDHGGSREESYKILDLVQEWKEKPAEEYGSKEVEILFTAVYNTVNDLAEKAHIEQGRCVKPLLIKLWVEILTSFKKELDSWTEETALTLDEYLSSSWVSIGCRICILNSLQYLGIKLSEEMLSSQECTDLCRHVSSVDRLLNDVQTFKKERLENTINSVGLQLAAHKGERAMTEEDAMSKIKEMADYHRRKLMQIVYKEGTVFPRECKDVFLRVCRIGYYLYSSGDEFTSPQQMKEDMKSLVYQPVKIHPLEAINV"
    
    logger.info(f"Test sequence length: {len(test_sequence)}")
    
    try:
        # Load ESM2 model
        model_name = "facebook/esm2_t6_8M_UR50D"
        logger.info(f"Loading ESM2 model: {model_name}")
        
        tokenizer = EsmTokenizer.from_pretrained(model_name)
        model = EsmForMaskedLM.from_pretrained(
            model_name,
            attn_implementation="eager"  # Enable attention extraction
        )
        
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        logger.info("ESM2 model loaded successfully")
        
        # Tokenize sequence
        inputs = tokenizer(
            test_sequence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        logger.info(f"Tokenized sequence length: {inputs['input_ids'].shape[1]}")
        
        # Get attention weights
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
            
            if outputs.attentions is not None:
                attention_weights = outputs.attentions[-1].cpu().numpy()  # Last layer
                logger.info(f"ESM2 attention weights shape: {attention_weights.shape}")
                
                # Analyze attention patterns
                analyze_esm2_attention(attention_weights, test_sequence, tokenizer)
            else:
                logger.error("ESM2 model did not return attention weights")
                
    except Exception as e:
        logger.error(f"Error testing ESM2: {e}")
        import traceback
        traceback.print_exc()

def analyze_esm2_attention(attention_weights, sequence, tokenizer):
    """Analyze ESM2 attention patterns"""
    
    logger.info("Analyzing ESM2 attention patterns...")
    
    # attention_weights shape: (batch_size, num_heads, seq_len, seq_len)
    batch_size, num_heads, seq_len, _ = attention_weights.shape
    
    # Average attention across heads
    avg_attention = np.mean(attention_weights[0], axis=0)  # Shape: (seq_len, seq_len)
    
    # Calculate attention received by each position
    attention_received = np.sum(avg_attention, axis=0)  # Total attention received by each position
    
    # Get token strings
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(test_sequence, return_tensors='pt')['input_ids'][0])
    
    logger.info(f"Number of attention heads: {num_heads}")
    logger.info(f"Sequence length: {seq_len}")
    logger.info(f"First 10 tokens: {tokens[:10]}")
    
    # Find most attended positions
    top_attended_indices = np.argsort(attention_received)[-10:]  # Top 10
    top_attended_tokens = [tokens[i] if i < len(tokens) else 'X' for i in top_attended_indices]
    top_attended_scores = attention_received[top_attended_indices]
    
    logger.info("\n=== ESM2 ATTENTION ANALYSIS ===")
    logger.info("Top 10 most attended positions:")
    for i, (pos, token, score) in enumerate(zip(top_attended_indices, top_attended_tokens, top_attended_scores)):
        logger.info(f"  {i+1}. Position {pos}: {token} (attention: {score:.4f})")
    
    # Analyze amino acid attention patterns
    aa_attention = {}
    for i, token in enumerate(tokens):
        if token not in ['<cls>', '<eos>', '<pad>'] and i < len(attention_received):
            if token not in aa_attention:
                aa_attention[token] = []
            aa_attention[token].append(attention_received[i])
    
    # Calculate mean attention per amino acid
    aa_mean_attention = {aa: np.mean(scores) for aa, scores in aa_attention.items()}
    
    logger.info("\nAmino acid attention patterns:")
    sorted_aa = sorted(aa_mean_attention.items(), key=lambda x: x[1], reverse=True)
    for aa, score in sorted_aa[:10]:  # Top 10 amino acids
        logger.info(f"  {aa}: {score:.4f}")
    
    # Compare with SaProt results
    logger.info("\n=== COMPARISON WITH SAPROT ===")
    logger.info("SaProt top attended positions were:")
    logger.info("  1. Position 0 (M): 9.00 attention")
    logger.info("  2. Position 581 (X): 7.70 attention") 
    logger.info("  3. Position 413 (D): 2.34 attention")
    logger.info("  4. Position 402 (D): 2.30 attention")
    
    # Check if ESM2 also shows methionine bias
    methionine_attention = attention_received[0] if len(attention_received) > 0 else 0
    logger.info(f"\nESM2 methionine (position 0) attention: {methionine_attention:.4f}")
    
    if methionine_attention > np.mean(attention_received) * 2:
        logger.warning("ESM2 also shows high methionine attention - this might be a general protein language model behavior")
    else:
        logger.info("ESM2 does not show excessive methionine attention - SaProt behavior might be specific to that model")
    
    # Create visualization
    create_esm2_visualization(attention_received, tokens, avg_attention)

def create_esm2_visualization(attention_received, tokens, attention_matrix):
    """Create ESM2 attention visualization"""
    
    logger.info("Creating ESM2 attention visualization...")
    
    output_dir = Path("esm2_attention_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Plot attention received by each position
    plt.figure(figsize=(15, 6))
    positions = range(len(attention_received))
    
    plt.plot(positions, attention_received, alpha=0.7, linewidth=1)
    plt.title('ESM2: Attention Received by Each Position')
    plt.xlabel('Position')
    plt.ylabel('Total Attention Received')
    
    # Highlight top 10 positions
    top_10_indices = np.argsort(attention_received)[-10:]
    plt.scatter(top_10_indices, attention_received[top_10_indices], 
               color='red', s=50, zorder=5, label='Top 10')
    
    # Highlight methionine
    if len(attention_received) > 0:
        plt.scatter([0], [attention_received[0]], 
                   color='green', s=100, zorder=5, label='Methionine (M)')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "esm2_attention_received.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot attention matrix (first 100 positions)
    plt.figure(figsize=(12, 10))
    max_pos = min(100, len(attention_received))
    attention_subset = attention_matrix[:max_pos, :max_pos]
    
    plt.imshow(attention_subset, cmap='Blues', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.title('ESM2: Attention Matrix - First 100 positions')
    plt.xlabel('Source Position')
    plt.ylabel('Target Position')
    
    # Add token labels
    token_labels = [tokens[i] for i in range(max_pos)]
    plt.xticks(range(0, max_pos, 10), [token_labels[i] for i in range(0, max_pos, 10)])
    plt.yticks(range(0, max_pos, 10), [token_labels[i] for i in range(0, max_pos, 10)])
    
    plt.tight_layout()
    plt.savefig(output_dir / "esm2_attention_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"ESM2 visualizations saved to {output_dir}/")

def test_saprot_vs_esm2_comparison():
    """Compare SaProt and ESM2 attention patterns side by side"""
    
    logger.info("Comparing SaProt vs ESM2 attention patterns...")
    
    test_sequence = "MSITFNLKIAPFSGPGIQRSKETFPATEIQITASTKSTMTTKCSFNASTDFMGKLREKVGGKADKPPVVIHPVDISSNLCMIDTLQSLGVDRYFQSEINTLLEHTYRLWKEKKKNIIFKDVSCCAIAFRLLREKGYQVSSDKLAPFADYRIRDVATILELYRASQARLYEDEHTLEKLHDWSSNLLKQHLLNGSIPDHKLHKQVEYFLKNYHGILDRVAVRRSLDLYNINHHHRIPDVADGFPKEDFLEYSMQDFNICQAQQQEELHQLQRWYADCRLDTLNYGRDVVRIANFLTSAIFGEPEFSDARLAFAKHIILVTRIDDFFDHGGSREESYKILDLVQEWKEKPAEEYGSKEVEILFTAVYNTVNDLAEKAHIEQGRCVKPLLIKLWVEILTSFKKELDSWTEETALTLDEYLSSSWVSIGCRICILNSLQYLGIKLSEEMLSSQECTDLCRHVSSVDRLLNDVQTFKKERLENTINSVGLQLAAHKGERAMTEEDAMSKIKEMADYHRRKLMQIVYKEGTVFPRECKDVFLRVCRIGYYLYSSGDEFTSPQQMKEDMKSLVYQPVKIHPLEAINV"
    
    try:
        # Test SaProt
        logger.info("Testing SaProt model...")
        from models.saprot_encoder import SaProtEncoder
        saprot_encoder = SaProtEncoder()
        saprot_result = saprot_encoder.encode_sequence(test_sequence, return_attention=True)
        
        if saprot_result and saprot_result.attention_weights is not None:
            saprot_attention = np.sum(np.mean(saprot_result.attention_weights[0], axis=1), axis=0)
            logger.info(f"SaProt attention shape: {saprot_result.attention_weights.shape}")
            logger.info(f"SaProt methionine attention: {saprot_attention[0]:.4f}")
            logger.info(f"SaProt mean attention: {np.mean(saprot_attention):.4f}")
            logger.info(f"SaProt methionine/mean ratio: {saprot_attention[0]/np.mean(saprot_attention):.2f}")
        else:
            logger.error("SaProt failed to extract attention")
            return
            
    except Exception as e:
        logger.error(f"Error testing SaProt: {e}")
        return
    
    # Test ESM2
    logger.info("\nTesting ESM2 model...")
    test_esm2_attention()

def main():
    """Main function"""
    test_saprot_vs_esm2_comparison()

if __name__ == "__main__":
    main()
