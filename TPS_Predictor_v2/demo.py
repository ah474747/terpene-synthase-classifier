"""
Demo Script for Terpene Synthase Product Predictor v2

This script demonstrates the complete pipeline with sample data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Import our custom modules
from models.molecular_encoder import TerpeneProductEncoder
from models.saprot_encoder import SaProtEncoder
from models.attention_classifier import TerpenePredictorTrainer, ModelConfig
from training.training_pipeline import TrainingPipeline, TrainingConfig
from evaluation.biological_validator import BiologicalValidator, ValidationConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data():
    """Create sample data for demonstration"""
    
    logger.info("Creating sample data...")
    
    # Sample terpene synthase sequences (simplified)
    sample_sequences = [
        "MSTEQFVLPDLLESCPLKDATNPYYKEAAAESRAWINGYDIFTDRKRAEFIQGQNELLCSHVYWYAGREQLRTTCDFVNLLFVVDEVSDEQNGKGARETGQVFFKAMKYPDWDDGSILAKVTKEFMARFTRLAGPRNTKRFIDLCESYTACVGEEAELRERSELLDLASYIPLRRQNSAVLLCFALVEYILGIDLADEVYEDEMFMKAYWAACDQVCWANDIYSYDMEQSKGLAGNNIVSILMNENGTNLQETADYIGERCGEFVSDYISAKSQISPSLGPEALQFIDFVGYWMIGNIEWCFETPRYFGSRHLEIKETRVVHLRPKEVPEGLSSEDCIESDDE",
        "MALVSIAPLASKSCLHKSLSSSAHELKTICRTIPTLGMSRRGKSATPSMSMSLTTTVSDDGVQRRMGDFHSNLWNDDFIQSLSTSYGEPSYRERAERLIGEVKKMFNSMSSEDGELINPHNDLIQRVWMVDSVERLGIERHFKNEIKSALDYVYSYWSEKGIGCGRESVVADLNSTALGLRTLRLHGYAVSADVLNLFKDQNGQFACSPSQTEEEIGSVLNLYRASLIAFPGEKVMEEAEIFSAKYLEEALQKISVSSLSQEIRDVLEYGWHTYLPRMEARNHIDVFGQDTQNSKSCINTEKLLELAKLEFNIFHSLQKRELEYLVRWWKDSGSPQMTFGRHRHVEYYTLASCIAFEPQHSGFRLGFAKTCHIITILDDMYDTFGTVDELELFTAAMKRWNPSAADCLPEYMKGMYMIVYDTVNEICQEAEKAQGRNTLDYARQAWDEYLDSYMQEAKWIVTGYLPTFAEYYENGKVSSGHRTAALQPILTMDIPFPPHILKEVDFPSKLNDLACAILRLRGDTRCYKADRARGEEASSISCYMKDNPGVTEEDALDHINAMISDVIRGLNWELLNPNSSVPISSKKHVFDISRAFHYGYKYRDGYSVANIETKSLVKRTVIDPVTL",
        "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELGERMACA",
        "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELGERMACD",
        "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELMYRCENE"
    ]
    
    # Sample labels
    sample_labels = ["limonene", "pinene", "limonene", "germacrene_a", "myrcene"]
    
    # Sample organisms
    sample_organisms = ["Citrus limon", "Pinus taeda", "Citrus limon", "Artemisia annua", "Mentha spicata"]
    
    # Generate random protein embeddings (simulating SaProt output)
    protein_embeddings = np.random.randn(len(sample_sequences), 1280)
    
    # Generate random molecular fingerprints (simulating RDKit output)
    molecular_fingerprints = np.random.randn(len(sample_sequences), 2223)
    
    return sample_sequences, sample_labels, sample_organisms, protein_embeddings, molecular_fingerprints

def demo_molecular_encoding():
    """Demonstrate molecular fingerprint encoding"""
    
    logger.info("Demo: Molecular Fingerprint Encoding")
    print("=" * 50)
    
    # Initialize encoder
    encoder = TerpeneProductEncoder()
    
    # Define terpene products
    products = ["limonene", "pinene", "myrcene", "linalool", "germacrene_a"]
    
    # Encode products
    encoded_products = encoder.encode_dataset(products)
    
    # Create fingerprint matrix
    fingerprint_matrix, product_names = encoder.create_fingerprint_matrix(encoded_products)
    
    print(f"Encoded {len(encoded_products)} terpene products")
    print(f"Fingerprint matrix shape: {fingerprint_matrix.shape}")
    print(f"Products: {product_names}")
    
    # Analyze descriptors
    from models.molecular_encoder import FingerprintAnalyzer
    analyzer = FingerprintAnalyzer()
    descriptor_df = analyzer.analyze_descriptors(encoded_products)
    
    print("\nMolecular Descriptors:")
    print(descriptor_df.round(2))
    
    return encoded_products

def demo_protein_encoding():
    """Demonstrate protein sequence encoding"""
    
    logger.info("Demo: Protein Sequence Encoding")
    print("=" * 50)
    
    # Initialize encoder
    encoder = SaProtEncoder()
    
    # Sample sequences
    sequences = [
        "MSTEQFVLPDLLESCPLKDATNPYYKEAAAESRAWINGYDIFTDRKRAEFIQGQNELLCSHVYWYAGREQLRTTCDFVNLLFVVDEVSDEQNGKGARETGQVFFKAMKYPDWDDGSILAKVTKEFMARFTRLAGPRNTKRFIDLCESYTACVGEEAELRERSELLDLASYIPLRRQNSAVLLCFALVEYILGIDLADEVYEDEMFMKAYWAACDQVCWANDIYSYDMEQSKGLAGNNIVSILMNENGTNLQETADYIGERCGEFVSDYISAKSQISPSLGPEALQFIDFVGYWMIGNIEWCFETPRYFGSRHLEIKETRVVHLRPKEVPEGLSSEDCIESDDE",
        "MALVSIAPLASKSCLHKSLSSSAHELKTICRTIPTLGMSRRGKSATPSMSMSLTTTVSDDGVQRRMGDFHSNLWNDDFIQSLSTSYGEPSYRERAERLIGEVKKMFNSMSSEDGELINPHNDLIQRVWMVDSVERLGIERHFKNEIKSALDYVYSYWSEKGIGCGRESVVADLNSTALGLRTLRLHGYAVSADVLNLFKDQNGQFACSPSQTEEEIGSVLNLYRASLIAFPGEKVMEEAEIFSAKYLEEALQKISVSSLSQEIRDVLEYGWHTYLPRMEARNHIDVFGQDTQNSKSCINTEKLLELAKLEFNIFHSLQKRELEYLVRWWKDSGSPQMTFGRHRHVEYYTLASCIAFEPQHSGFRLGFAKTCHIITILDDMYDTFGTVDELELFTAAMKRWNPSAADCLPEYMKGMYMIVYDTVNEICQEAEKAQGRNTLDYARQAWDEYLDSYMQEAKWIVTGYLPTFAEYYENGKVSSGHRTAALQPILTMDIPFPPHILKEVDFPSKLNDLACAILRLRGDTRCYKADRARGEEASSISCYMKDNPGVTEEDALDHINAMISDVIRGLNWELLNPNSSVPISSKKHVFDISRAFHYGYKYRDGYSVANIETKSLVKRTVIDPVTL"
    ]
    
    # Encode sequences
    embeddings = encoder.encode_sequences(sequences)
    
    # Create embedding matrix
    embedding_matrix, _ = encoder.create_embedding_matrix(embeddings)
    
    print(f"Encoded {len(embeddings)} protein sequences")
    print(f"Embedding matrix shape: {embedding_matrix.shape}")
    print(f"Model used: {embeddings[0].model_name}")
    
    # Analyze embeddings
    from models.saprot_encoder import EmbeddingAnalyzer
    analyzer = EmbeddingAnalyzer()
    analysis = analyzer.analyze_embeddings(embeddings)
    
    print(f"\nEmbedding Analysis:")
    print(f"Number of embeddings: {analysis['num_embeddings']}")
    print(f"Embedding dimension: {analysis['embedding_dimension']}")
    print(f"Sequence lengths: {analysis['sequence_lengths']}")
    
    return embeddings

def demo_training_pipeline():
    """Demonstrate training pipeline"""
    
    logger.info("Demo: Training Pipeline")
    print("=" * 50)
    
    # Create sample data
    sequences, labels, organisms, protein_embeddings, molecular_fingerprints = create_sample_data()
    
    # Create training config
    config = TrainingConfig(
        protein_embedding_dim=1280,
        molecular_fingerprint_dim=2223,
        num_classes=len(set(labels)),
        min_samples_per_class=1,  # Lower threshold for demo
        max_samples_per_class=100,
        num_epochs=10,  # Fewer epochs for demo
        early_stopping_patience=5
    )
    
    # Initialize pipeline
    pipeline = TrainingPipeline(config)
    
    # Load data
    protein_embeddings, molecular_fingerprints, labels = pipeline.load_data(
        protein_embeddings, molecular_fingerprints, labels
    )
    
    # Preprocess data
    protein_scaled, molecular_scaled, labels_encoded = pipeline.preprocess_data(
        protein_embeddings, molecular_fingerprints, labels
    )
    
    print(f"Preprocessed data shapes:")
    print(f"  Protein embeddings: {protein_scaled.shape}")
    print(f"  Molecular fingerprints: {molecular_scaled.shape}")
    print(f"  Labels: {len(labels_encoded)}")
    
    # Train model
    trainer = pipeline.train_final_model(
        protein_embeddings, molecular_fingerprints, labels
    )
    
    print(f"Training completed!")
    print(f"Test accuracy: {trainer.results['final_model']['test_accuracy']:.4f}")
    
    return trainer

def demo_biological_validation():
    """Demonstrate biological validation"""
    
    logger.info("Demo: Biological Validation")
    print("=" * 50)
    
    # Create sample data
    sequences, labels, organisms, protein_embeddings, molecular_fingerprints = create_sample_data()
    
    # Create validation config
    config = ValidationConfig(
        holdout_organisms=["Citrus limon"],
        min_accuracy_threshold=0.5,  # Lower threshold for demo
        min_f1_threshold=0.4
    )
    
    # Initialize validator
    validator = BiologicalValidator(config)
    
    # Create mock trainer
    model_config = ModelConfig(
        protein_embedding_dim=1280,
        molecular_fingerprint_dim=2223,
        num_classes=len(set(labels))
    )
    
    trainer = TerpenePredictorTrainer(model_config)
    
    # Run validation
    results = validator.run_comprehensive_validation(
        protein_embeddings, molecular_fingerprints, labels, organisms,
        sequences, [f"SEQ_{i:03d}" for i in range(len(sequences))], trainer
    )
    
    print(f"Validation completed!")
    for test_name, result in results.items():
        print(f"{test_name}: Accuracy={result.accuracy:.4f}, F1={result.f1_score:.4f}")
    
    return results

def main():
    """Main demo function"""
    
    print("Terpene Synthase Product Predictor v2 - Demo")
    print("=" * 60)
    
    try:
        # Demo 1: Molecular encoding
        print("\n1. Molecular Fingerprint Encoding Demo")
        encoded_products = demo_molecular_encoding()
        
        # Demo 2: Protein encoding
        print("\n2. Protein Sequence Encoding Demo")
        embeddings = demo_protein_encoding()
        
        # Demo 3: Training pipeline
        print("\n3. Training Pipeline Demo")
        trainer = demo_training_pipeline()
        
        # Demo 4: Biological validation
        print("\n4. Biological Validation Demo")
        validation_results = demo_biological_validation()
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("All components are working correctly.")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"Demo failed: {e}")
        raise

if __name__ == "__main__":
    main()
