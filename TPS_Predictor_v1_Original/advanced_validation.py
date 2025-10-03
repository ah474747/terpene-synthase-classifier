"""
Advanced validation script combining BRENDA curated data with SaProt protein language models.
This addresses both data curation and computational approach issues.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
from brenda_collector import BRENDACollector
from saprot_predictor import SaProtTerpenePredictor
import warnings
warnings.filterwarnings('ignore')


class AdvancedTerpeneSynthaseValidator:
    """
    Advanced validation class combining BRENDA curated data with SaProt protein language models.
    """
    
    def __init__(self):
        """Initialize the advanced validator."""
        self.brenda_collector = BRENDACollector()
        self.saprot_predictor = SaProtTerpenePredictor()
        
    def create_high_quality_dataset(self, limit: int = 100) -> pd.DataFrame:
        """
        Create a high-quality dataset using BRENDA curated data.
        
        Args:
            limit: Maximum number of entries to process
            
        Returns:
            DataFrame with high-quality terpene synthase data
        """
        print("Creating high-quality dataset from BRENDA...")
        
        # Get curated data from BRENDA
        df = self.brenda_collector.create_curated_dataset(limit)
        
        # Filter for high-quality data
        print("Filtering for high-quality data...")
        
        # Remove entries with missing sequences
        df = df.dropna(subset=['sequence'])
        
        # Filter by sequence length (200-1000 amino acids)
        df = df[(df['sequence'].str.len() >= 200) & (df['sequence'].str.len() <= 1000)]
        
        # Remove duplicate sequences
        df = df.drop_duplicates(subset=['sequence'])
        
        # Filter products with sufficient samples
        product_counts = df['product'].value_counts()
        valid_products = product_counts[product_counts >= 3].index
        df = df[df['product'].isin(valid_products)]
        
        print(f"High-quality dataset: {len(df)} sequences, {len(df['product'].unique())} products")
        print(f"Product distribution:")
        print(df['product'].value_counts())
        
        return df
    
    def validate_saprot_approach(self, 
                               validation_df: pd.DataFrame,
                               cv_folds: int = 5) -> Dict[str, any]:
        """
        Validate the SaProt-based approach.
        
        Args:
            validation_df: High-quality validation dataset
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with validation results
        """
        print(f"\nValidating SaProt-based approach...")
        
        # Prepare data
        sequences = validation_df['sequence'].tolist()
        products = validation_df['product'].tolist()
        
        print(f"Processing {len(sequences)} sequences...")
        
        # Extract SaProt embeddings
        try:
            embeddings = self.saprot_predictor.extract_protein_embeddings(sequences)
        except Exception as e:
            print(f"Error extracting SaProt embeddings: {e}")
            print("Falling back to traditional features...")
            return self._validate_traditional_approach(validation_df, cv_folds)
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(products)
        
        print(f"SaProt embeddings shape: {embeddings.shape}")
        print(f"Unique products: {len(np.unique(y_encoded))}")
        
        # Initialize model
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, embeddings, y_encoded, cv=cv_folds, scoring='accuracy')
        
        # Calculate additional metrics
        from sklearn.model_selection import cross_validate
        scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        cv_results = cross_validate(model, embeddings, y_encoded, cv=cv_folds, scoring=scoring)
        
        # Store results
        validation_metrics = {
            'accuracy_mean': cv_scores.mean(),
            'accuracy_std': cv_scores.std(),
            'precision_mean': cv_results['test_precision_macro'].mean(),
            'precision_std': cv_results['test_precision_macro'].std(),
            'recall_mean': cv_results['test_recall_macro'].mean(),
            'recall_std': cv_results['test_recall_macro'].std(),
            'f1_mean': cv_results['test_f1_macro'].mean(),
            'f1_std': cv_results['test_f1_macro'].std(),
            'cv_folds': cv_folds,
            'n_samples': len(embeddings),
            'n_features': embeddings.shape[1],
            'n_classes': len(np.unique(y_encoded))
        }
        
        print(f"SaProt validation results:")
        print(f"  Accuracy: {validation_metrics['accuracy_mean']:.4f} ± {validation_metrics['accuracy_std']:.4f}")
        print(f"  Precision: {validation_metrics['precision_mean']:.4f} ± {validation_metrics['precision_std']:.4f}")
        print(f"  Recall: {validation_metrics['recall_mean']:.4f} ± {validation_metrics['recall_std']:.4f}")
        print(f"  F1-score: {validation_metrics['f1_mean']:.4f} ± {validation_metrics['f1_std']:.4f}")
        
        # Train final model and get feature importance
        model.fit(embeddings, y_encoded)
        
        # For SaProt embeddings, we can't get traditional feature importance
        # Instead, we'll analyze the model's performance
        feature_analysis = {
            'embedding_dimension': embeddings.shape[1],
            'model_type': 'SaProt + Random Forest',
            'note': 'Feature importance not available for protein language model embeddings'
        }
        
        return {
            'validation_metrics': validation_metrics,
            'feature_analysis': feature_analysis,
            'label_encoder': label_encoder,
            'model': model,
            'embeddings': embeddings
        }
    
    def _validate_traditional_approach(self, 
                                     validation_df: pd.DataFrame,
                                     cv_folds: int = 5) -> Dict[str, any]:
        """
        Fallback validation using traditional features.
        
        Args:
            validation_df: Validation dataset
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with validation results
        """
        print("Using traditional feature extraction...")
        
        # Prepare data
        sequences = validation_df['sequence'].tolist()
        products = validation_df['product'].tolist()
        
        # Extract traditional features
        from feature_extractor import ProteinFeatureExtractor
        extractor = ProteinFeatureExtractor()
        features_df = extractor.extract_all_features(sequences)
        
        # Prepare features and labels
        feature_cols = [col for col in features_df.columns if col != 'sequence_id']
        X = features_df[feature_cols].values
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Feature selection
        if X.shape[1] > 100:
            selector = SelectKBest(f_classif, k=100)
            X = selector.fit_transform(X, products)
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(products)
        
        # Initialize model
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y_encoded, cv=cv_folds, scoring='accuracy')
        
        # Calculate additional metrics
        from sklearn.model_selection import cross_validate
        scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        cv_results = cross_validate(model, X, y_encoded, cv=cv_folds, scoring=scoring)
        
        # Store results
        validation_metrics = {
            'accuracy_mean': cv_scores.mean(),
            'accuracy_std': cv_scores.std(),
            'precision_mean': cv_results['test_precision_macro'].mean(),
            'precision_std': cv_results['test_precision_macro'].std(),
            'recall_mean': cv_results['test_recall_macro'].mean(),
            'recall_std': cv_results['test_recall_macro'].std(),
            'f1_mean': cv_results['test_f1_macro'].mean(),
            'f1_std': cv_results['test_f1_macro'].std(),
            'cv_folds': cv_folds,
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y_encoded))
        }
        
        print(f"Traditional validation results:")
        print(f"  Accuracy: {validation_metrics['accuracy_mean']:.4f} ± {validation_metrics['accuracy_std']:.4f}")
        print(f"  Precision: {validation_metrics['precision_mean']:.4f} ± {validation_metrics['precision_std']:.4f}")
        print(f"  Recall: {validation_metrics['recall_mean']:.4f} ± {validation_metrics['recall_std']:.4f}")
        print(f"  F1-score: {validation_metrics['f1_mean']:.4f} ± {validation_metrics['f1_std']:.4f}")
        
        # Train final model and get feature importance
        model.fit(X, y_encoded)
        importance = model.feature_importances_
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_cols[:len(importance)],
            'importance': importance
        }).sort_values('importance', ascending=False).head(20)
        
        print(f"\nTop 20 most important features:")
        print(importance_df)
        
        return {
            'validation_metrics': validation_metrics,
            'feature_importance': importance_df,
            'label_encoder': label_encoder,
            'model': model
        }
    
    def compare_approaches(self, 
                         validation_df: pd.DataFrame) -> Dict[str, any]:
        """
        Compare SaProt vs traditional approaches.
        
        Args:
            validation_df: Validation dataset
            
        Returns:
            Dictionary with comparison results
        """
        print("\n" + "="*60)
        print("COMPARING SAPROT VS TRADITIONAL APPROACHES")
        print("="*60)
        
        # Test SaProt approach
        print("\n1. Testing SaProt approach...")
        saprot_results = self.validate_saprot_approach(validation_df)
        
        # Test traditional approach
        print("\n2. Testing traditional approach...")
        traditional_results = self._validate_traditional_approach(validation_df)
        
        # Compare results
        comparison = {
            'saprot': {
                'accuracy': saprot_results['validation_metrics']['accuracy_mean'],
                'precision': saprot_results['validation_metrics']['precision_mean'],
                'recall': saprot_results['validation_metrics']['recall_mean'],
                'f1_score': saprot_results['validation_metrics']['f1_mean'],
                'n_features': saprot_results['validation_metrics']['n_features']
            },
            'traditional': {
                'accuracy': traditional_results['validation_metrics']['accuracy_mean'],
                'precision': traditional_results['validation_metrics']['precision_mean'],
                'recall': traditional_results['validation_metrics']['recall_mean'],
                'f1_score': traditional_results['validation_metrics']['f1_mean'],
                'n_features': traditional_results['validation_metrics']['n_features']
            }
        }
        
        # Calculate improvements
        improvements = {
            'accuracy_improvement': comparison['saprot']['accuracy'] - comparison['traditional']['accuracy'],
            'precision_improvement': comparison['saprot']['precision'] - comparison['traditional']['precision'],
            'recall_improvement': comparison['saprot']['recall'] - comparison['traditional']['recall'],
            'f1_improvement': comparison['saprot']['f1_score'] - comparison['traditional']['f1_score']
        }
        
        print(f"\nComparison Results:")
        print(f"SaProt Accuracy: {comparison['saprot']['accuracy']:.4f}")
        print(f"Traditional Accuracy: {comparison['traditional']['accuracy']:.4f}")
        print(f"Improvement: {improvements['accuracy_improvement']:+.4f}")
        
        print(f"\nSaProt F1-score: {comparison['saprot']['f1_score']:.4f}")
        print(f"Traditional F1-score: {comparison['traditional']['f1_score']:.4f}")
        print(f"Improvement: {improvements['f1_improvement']:+.4f}")
        
        return {
            'comparison': comparison,
            'improvements': improvements,
            'saprot_results': saprot_results,
            'traditional_results': traditional_results
        }
    
    def run_advanced_validation(self, 
                              limit: int = 100) -> Dict[str, any]:
        """
        Run advanced validation combining BRENDA data with SaProt.
        
        Args:
            limit: Maximum number of entries to process
            
        Returns:
            Dictionary with validation results
        """
        print("="*60)
        print("ADVANCED VALIDATION: BRENDA + SAPROT")
        print("="*60)
        
        # Step 1: Create high-quality dataset
        validation_df = self.create_high_quality_dataset(limit)
        
        if len(validation_df) < 10:
            print("Warning: Dataset too small for reliable validation")
            return {}
        
        # Step 2: Compare approaches
        comparison_results = self.compare_approaches(validation_df)
        
        # Compile results
        validation_summary = {
            'dataset_info': {
                'total_samples': len(validation_df),
                'unique_products': len(validation_df['product'].unique()),
                'product_distribution': validation_df['product'].value_counts().to_dict(),
                'ec_numbers': validation_df['ec_number'].unique().tolist()
            },
            'comparison_results': comparison_results
        }
        
        # Save results
        import json
        with open('data/advanced_validation_results.json', 'w') as f:
            json.dump(validation_summary, f, indent=2, default=str)
        
        print("\n" + "="*60)
        print("ADVANCED VALIDATION COMPLETE")
        print("="*60)
        print(f"Results saved to data/advanced_validation_results.json")
        
        return validation_summary


def main():
    """Main function to run advanced validation."""
    # Initialize validator
    validator = AdvancedTerpeneSynthaseValidator()
    
    # Run advanced validation
    results = validator.run_advanced_validation(limit=50)
    
    if results:
        print("\nAdvanced Validation Summary:")
        print(f"Dataset: {results['dataset_info']['total_samples']} samples, {results['dataset_info']['unique_products']} products")
        
        if 'comparison_results' in results:
            comparison = results['comparison_results']['comparison']
            print(f"SaProt accuracy: {comparison['saprot']['accuracy']:.4f}")
            print(f"Traditional accuracy: {comparison['traditional']['accuracy']:.4f}")
            
            improvements = results['comparison_results']['improvements']
            print(f"Accuracy improvement: {improvements['accuracy_improvement']:+.4f}")


if __name__ == "__main__":
    main()
