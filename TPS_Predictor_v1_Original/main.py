"""
Main script for terpene synthase product prediction.
This script orchestrates the entire pipeline from data collection to model training.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import argparse
from tqdm import tqdm

# Import our custom modules
from data_collector import TerpeneSynthaseDataCollector
from feature_extractor import ProteinFeatureExtractor
from model_trainer import TerpeneSynthasePredictor


class TerpeneSynthasePipeline:
    """
    Main pipeline class that orchestrates the entire terpene synthase prediction workflow.
    """
    
    def __init__(self, email: str = "your_email@example.com"):
        """
        Initialize the pipeline.
        
        Args:
            email: Email address for NCBI API access
        """
        self.data_collector = TerpeneSynthaseDataCollector(email)
        self.feature_extractor = ProteinFeatureExtractor()
        self.model_trainer = TerpeneSynthasePredictor()
        
    def run_full_pipeline(self, 
                         collect_data: bool = True,
                         extract_features: bool = True,
                         train_models: bool = True,
                         uniprot_limit: int = 500,
                         ncbi_limit: int = 500) -> Dict[str, any]:
        """
        Run the complete pipeline from data collection to model training.
        
        Args:
            collect_data: Whether to collect new data
            extract_features: Whether to extract features
            train_models: Whether to train models
            uniprot_limit: Maximum number of UniProt entries to collect
            ncbi_limit: Maximum number of NCBI entries to collect
            
        Returns:
            Dictionary with pipeline results
        """
        results = {}
        
        # Step 1: Data Collection
        if collect_data:
            print("=" * 50)
            print("STEP 1: DATA COLLECTION")
            print("=" * 50)
            
            # Collect from UniProt
            uniprot_proteins = self.data_collector.search_uniprot_terpene_synthases(limit=uniprot_limit)
            
            # Collect from NCBI
            ncbi_proteins = self.data_collector.search_ncbi_terpene_synthases(limit=ncbi_limit)
            
            # Combine and annotate
            all_proteins = uniprot_proteins + ncbi_proteins
            annotated_proteins = self.data_collector.extract_product_annotations(all_proteins)
            
            # Save data
            self.data_collector.save_data(annotated_proteins, "terpene_synthase_data.json")
            
            results['data_collection'] = {
                'uniprot_count': len(uniprot_proteins),
                'ncbi_count': len(ncbi_proteins),
                'total_count': len(annotated_proteins),
                'annotated_count': len([p for p in annotated_proteins if p.get('products')])
            }
            
            print(f"Data collection complete: {len(annotated_proteins)} proteins collected")
        
        # Step 2: Feature Extraction
        if extract_features:
            print("\n" + "=" * 50)
            print("STEP 2: FEATURE EXTRACTION")
            print("=" * 50)
            
            # Load data
            proteins = self.data_collector.load_data("terpene_synthase_data.json")
            
            if not proteins:
                print("No data found. Please run data collection first.")
                return results
            
            # Prepare sequences and products
            sequences = []
            sequence_ids = []
            products = []
            
            for protein in proteins:
                if protein.get('sequence') and protein.get('products'):
                    sequences.append(protein['sequence'])
                    sequence_ids.append(protein.get('accession', f"seq_{len(sequences)}"))
                    # Use the first product for now (could be extended to handle multiple products)
                    products.append(protein['products'][0])
            
            print(f"Extracting features from {len(sequences)} sequences...")
            
            # Extract features
            features_df = self.feature_extractor.extract_all_features(sequences, sequence_ids)
            
            # Create products DataFrame
            products_df = pd.DataFrame({
                'sequence_id': sequence_ids,
                'product': products
            })
            
            # Save features
            self.feature_extractor.save_features(features_df, "protein_features.csv")
            products_df.to_csv("data/product_annotations.csv", index=False)
            
            results['feature_extraction'] = {
                'sequences_processed': len(sequences),
                'features_extracted': len(features_df.columns) - 1,
                'unique_products': len(products_df['product'].unique())
            }
            
            print(f"Feature extraction complete: {len(features_df.columns)-1} features extracted")
        
        # Step 3: Model Training
        if train_models:
            print("\n" + "=" * 50)
            print("STEP 3: MODEL TRAINING")
            print("=" * 50)
            
            # Load features and products
            features_df = self.feature_extractor.load_features("protein_features.csv")
            products_df = pd.read_csv("data/product_annotations.csv")
            
            if features_df.empty or products_df.empty:
                print("No features or products found. Please run feature extraction first.")
                return results
            
            # Prepare data for training
            X, y = self.model_trainer.prepare_data(features_df, products_df)
            
            if X.shape[0] == 0:
                print("No valid data for training. Please check your data.")
                return results
            
            # Train models
            training_results = self.model_trainer.train_models(X, y)
            
            # Find best model
            best_model = None
            best_accuracy = 0
            
            for model_name, model_info in training_results.items():
                if 'accuracy' in model_info and model_info['accuracy'] > best_accuracy:
                    best_accuracy = model_info['accuracy']
                    best_model = model_name
            
            # Hyperparameter tuning for best model
            if best_model:
                print(f"\nPerforming hyperparameter tuning for best model: {best_model}")
                tuning_results = self.model_trainer.hyperparameter_tuning(X, y, best_model)
                
                if tuning_results:
                    # Save best model
                    self.model_trainer.save_model(best_model)
                    
                    results['model_training'] = {
                        'best_model': best_model,
                        'best_accuracy': best_accuracy,
                        'tuning_results': tuning_results,
                        'all_results': {name: info.get('accuracy', 0) for name, info in training_results.items()}
                    }
            
            print(f"Model training complete. Best model: {best_model} (Accuracy: {best_accuracy:.4f})")
        
        return results
    
    def predict_products(self, sequences: List[str], model_name: str = 'Random Forest') -> List[Dict[str, any]]:
        """
        Predict products for new sequences.
        
        Args:
            sequences: List of protein sequences
            model_name: Name of the model to use for prediction
            
        Returns:
            List of prediction results
        """
        print(f"Predicting products for {len(sequences)} sequences using {model_name}...")
        
        # Extract features
        features_df = self.feature_extractor.extract_all_features(sequences)
        
        # Prepare features (exclude sequence_id)
        feature_cols = [col for col in features_df.columns if col != 'sequence_id']
        X = features_df[feature_cols].values
        
        # Make predictions
        predictions, probabilities = self.model_trainer.predict(X, model_name)
        
        if len(predictions) == 0:
            print("No predictions made. Please ensure the model is trained.")
            return []
        
        # Decode predictions
        if 'product' in self.model_trainer.label_encoders:
            decoded_predictions = self.model_trainer.label_encoders['product'].inverse_transform(predictions)
        else:
            decoded_predictions = predictions
        
        # Prepare results
        results = []
        for i, (pred, prob) in enumerate(zip(decoded_predictions, probabilities)):
            result = {
                'sequence_id': features_df.iloc[i]['sequence_id'],
                'predicted_product': pred,
                'confidence': float(np.max(prob)),
                'all_probabilities': {
                    self.model_trainer.label_encoders['product'].classes_[j]: float(prob[j])
                    for j in range(len(prob))
                }
            }
            results.append(result)
        
        return results
    
    def evaluate_model_performance(self, model_name: str = 'Random Forest'):
        """
        Evaluate model performance and generate visualizations.
        
        Args:
            model_name: Name of the model to evaluate
        """
        print(f"Evaluating {model_name} performance...")
        
        # Load test data
        features_df = self.feature_extractor.load_features("protein_features.csv")
        products_df = pd.read_csv("data/product_annotations.csv")
        
        if features_df.empty or products_df.empty:
            print("No data found for evaluation.")
            return
        
        # Prepare data
        X, y = self.model_trainer.prepare_data(features_df, products_df)
        
        # Split data for evaluation
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Evaluate model
        evaluation_results = self.model_trainer.evaluate_model(model_name, X_test, y_test)
        
        if evaluation_results:
            print(f"Model Accuracy: {evaluation_results['accuracy']:.4f}")
            
            # Plot feature importance
            self.model_trainer.plot_feature_importance(model_name)
            
            # Plot confusion matrix
            self.model_trainer.plot_confusion_matrix(model_name)
            
            return evaluation_results
        
        return None


def main():
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(description='Terpene Synthase Product Prediction Pipeline')
    parser.add_argument('--email', type=str, default='your_email@example.com',
                       help='Email address for NCBI API access')
    parser.add_argument('--skip-data', action='store_true',
                       help='Skip data collection step')
    parser.add_argument('--skip-features', action='store_true',
                       help='Skip feature extraction step')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip model training step')
    parser.add_argument('--uniprot-limit', type=int, default=500,
                       help='Maximum number of UniProt entries to collect')
    parser.add_argument('--ncbi-limit', type=int, default=500,
                       help='Maximum number of NCBI entries to collect')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate model performance after training')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = TerpeneSynthasePipeline(email=args.email)
    
    # Run pipeline
    results = pipeline.run_full_pipeline(
        collect_data=not args.skip_data,
        extract_features=not args.skip_features,
        train_models=not args.skip_training,
        uniprot_limit=args.uniprot_limit,
        ncbi_limit=args.ncbi_limit
    )
    
    # Print summary
    print("\n" + "=" * 50)
    print("PIPELINE SUMMARY")
    print("=" * 50)
    
    for step, step_results in results.items():
        print(f"\n{step.upper()}:")
        for key, value in step_results.items():
            print(f"  {key}: {value}")
    
    # Evaluate model if requested
    if args.evaluate and 'model_training' in results:
        best_model = results['model_training'].get('best_model')
        if best_model:
            pipeline.evaluate_model_performance(best_model)
    
    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()
