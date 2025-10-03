"""
Simple validation script for terpene synthase product prediction.
This script implements comprehensive validation methods without XGBoost/LightGBM dependencies.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from data_collector import TerpeneSynthaseDataCollector
from feature_extractor import ProteinFeatureExtractor
import warnings
warnings.filterwarnings('ignore')


class SimpleTerpeneSynthaseValidator:
    """
    Simple validation class for terpene synthase product prediction.
    """
    
    def __init__(self, email: str = "your_email@example.com"):
        """
        Initialize the validator.
        
        Args:
            email: Email address for NCBI API access
        """
        self.data_collector = TerpeneSynthaseDataCollector(email)
        self.feature_extractor = ProteinFeatureExtractor()
        self.validation_results = {}
        
    def create_validation_dataset(self, 
                                 uniprot_limit: int = 100,
                                 ncbi_limit: int = 100,
                                 min_sequence_length: int = 200,
                                 max_sequence_length: int = 1000) -> pd.DataFrame:
        """
        Create a validation dataset with real terpene synthase data.
        
        Args:
            uniprot_limit: Maximum number of UniProt entries
            ncbi_limit: Maximum number of NCBI entries
            min_sequence_length: Minimum sequence length
            max_sequence_length: Maximum sequence length
            
        Returns:
            DataFrame with validation data
        """
        print("Creating validation dataset...")
        
        # Collect data from both sources
        uniprot_proteins = self.data_collector.search_uniprot_terpene_synthases(limit=uniprot_limit)
        ncbi_proteins = self.data_collector.search_ncbi_terpene_synthases(limit=ncbi_limit)
        
        # Combine and annotate
        all_proteins = uniprot_proteins + ncbi_proteins
        annotated_proteins = self.data_collector.extract_product_annotations(all_proteins)
        
        # Filter by sequence length and product availability
        filtered_proteins = []
        for protein in annotated_proteins:
            sequence = protein.get('sequence', '')
            products = protein.get('products', [])
            
            if (len(sequence) >= min_sequence_length and 
                len(sequence) <= max_sequence_length and 
                len(products) > 0):
                filtered_proteins.append(protein)
        
        print(f"Filtered to {len(filtered_proteins)} proteins with valid sequences and products")
        
        # Create validation dataset
        validation_data = []
        for protein in filtered_proteins:
            for product in protein['products']:
                validation_data.append({
                    'accession': protein.get('accession', ''),
                    'name': protein.get('name', ''),
                    'organism': protein.get('organism', ''),
                    'sequence': protein['sequence'],
                    'product': product,
                    'source': protein.get('source', 'unknown')
                })
        
        validation_df = pd.DataFrame(validation_data)
        
        # Save validation dataset
        os.makedirs("data", exist_ok=True)
        validation_df.to_csv("data/validation_dataset.csv", index=False)
        
        print(f"Created validation dataset with {len(validation_df)} entries")
        print(f"Unique products: {len(validation_df['product'].unique())}")
        print(f"Product distribution:")
        print(validation_df['product'].value_counts().head(10))
        
        return validation_df
    
    def cross_validate_model(self, 
                           validation_df: pd.DataFrame,
                           model_name: str = 'Random Forest',
                           cv_folds: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation on the validation dataset.
        
        Args:
            validation_df: Validation dataset
            model_name: Name of the model to validate
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with validation metrics
        """
        print(f"\nPerforming {cv_folds}-fold cross-validation with {model_name}...")
        
        # Prepare data
        sequences = validation_df['sequence'].tolist()
        products = validation_df['product'].tolist()
        
        # Extract features
        print("Extracting features...")
        features_df = self.feature_extractor.extract_all_features(sequences)
        
        # Prepare features and labels
        feature_cols = [col for col in features_df.columns if col != 'sequence_id']
        X = features_df[feature_cols].values
        y = products
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        print(f"Data shape: X={X.shape}, y={y_encoded.shape}")
        print(f"Unique products: {len(np.unique(y_encoded))}")
        
        # Initialize model
        if model_name == 'Random Forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        elif model_name == 'Gradient Boosting':
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif model_name == 'SVM':
            model = SVC(kernel='rbf', random_state=42, probability=True)
        elif model_name == 'Logistic Regression':
            model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_name == 'Neural Network':
            model = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
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
        
        print(f"Cross-validation results:")
        print(f"  Accuracy: {validation_metrics['accuracy_mean']:.4f} ± {validation_metrics['accuracy_std']:.4f}")
        print(f"  Precision: {validation_metrics['precision_mean']:.4f} ± {validation_metrics['precision_std']:.4f}")
        print(f"  Recall: {validation_metrics['recall_mean']:.4f} ± {validation_metrics['recall_std']:.4f}")
        print(f"  F1-score: {validation_metrics['f1_mean']:.4f} ± {validation_metrics['f1_std']:.4f}")
        
        return validation_metrics
    
    def test_with_known_products(self, 
                                validation_df: pd.DataFrame,
                                model_name: str = 'Random Forest') -> Dict[str, any]:
        """
        Test the model with known products and compare predictions.
        
        Args:
            validation_df: Validation dataset with known products
            model_name: Name of the model to test
            
        Returns:
            Dictionary with test results
        """
        print(f"\nTesting {model_name} with known products...")
        
        # Prepare data
        sequences = validation_df['sequence'].tolist()
        known_products = validation_df['product'].tolist()
        
        # Extract features
        features_df = self.feature_extractor.extract_all_features(sequences)
        
        # Prepare features and labels
        feature_cols = [col for col in features_df.columns if col != 'sequence_id']
        X = features_df[feature_cols].values
        y = known_products
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )
        
        # Train model
        if model_name == 'Random Forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        elif model_name == 'Gradient Boosting':
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif model_name == 'SVM':
            model = SVC(kernel='rbf', random_state=42, probability=True)
        elif model_name == 'Logistic Regression':
            model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_name == 'Neural Network':
            model = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        
        # Create detailed results
        test_results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'y_test': y_test,
            'label_encoder': label_encoder
        }
        
        print(f"Test results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-score: {f1:.4f}")
        
        # Show classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
        
        return test_results
    
    def analyze_feature_importance(self, 
                                 validation_df: pd.DataFrame,
                                 model_name: str = 'Random Forest',
                                 top_n: int = 20) -> pd.DataFrame:
        """
        Analyze feature importance for biological relevance.
        
        Args:
            validation_df: Validation dataset
            model_name: Name of the model to analyze
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        print(f"\nAnalyzing feature importance with {model_name}...")
        
        # Prepare data
        sequences = validation_df['sequence'].tolist()
        products = validation_df['product'].tolist()
        
        # Extract features
        features_df = self.feature_extractor.extract_all_features(sequences)
        
        # Prepare features and labels
        feature_cols = [col for col in features_df.columns if col != 'sequence_id']
        X = features_df[feature_cols].values
        y = products
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Train model
        if model_name == 'Random Forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            print(f"Feature importance not available for {model_name}")
            return pd.DataFrame()
        
        model.fit(X, y_encoded)
        
        # Get feature importance
        importance = model.feature_importances_
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        print(f"Top {top_n} most important features:")
        print(importance_df)
        
        # Analyze biological relevance
        print("\nBiological relevance analysis:")
        motif_features = importance_df[importance_df['feature'].str.contains('motif')]
        aa_features = importance_df[importance_df['feature'].str.contains('_percent')]
        kmer_features = importance_df[importance_df['feature'].str.contains('kmer')]
        
        print(f"Motif features in top {top_n}: {len(motif_features)}")
        print(f"Amino acid features in top {top_n}: {len(aa_features)}")
        print(f"K-mer features in top {top_n}: {len(kmer_features)}")
        
        return importance_df
    
    def test_sequence_length_robustness(self, 
                                      validation_df: pd.DataFrame,
                                      model_name: str = 'Random Forest') -> Dict[str, float]:
        """
        Test model robustness across different sequence lengths.
        
        Args:
            validation_df: Validation dataset
            model_name: Name of the model to test
            
        Returns:
            Dictionary with robustness results
        """
        print(f"\nTesting sequence length robustness with {model_name}...")
        
        # Group sequences by length
        validation_df['sequence_length'] = validation_df['sequence'].str.len()
        
        # Define length bins
        length_bins = [0, 300, 500, 700, 1000, float('inf')]
        length_labels = ['<300', '300-500', '500-700', '700-1000', '>1000']
        validation_df['length_bin'] = pd.cut(validation_df['sequence_length'], 
                                           bins=length_bins, labels=length_labels)
        
        robustness_results = {}
        
        for bin_label in length_labels:
            bin_data = validation_df[validation_df['length_bin'] == bin_label]
            
            if len(bin_data) < 10:  # Skip bins with too few samples
                continue
            
            print(f"\nTesting {bin_label} amino acids ({len(bin_data)} sequences)...")
            
            # Prepare data for this bin
            sequences = bin_data['sequence'].tolist()
            products = bin_data['product'].tolist()
            
            # Extract features
            features_df = self.feature_extractor.extract_all_features(sequences)
            
            # Prepare features and labels
            feature_cols = [col for col in features_df.columns if col != 'sequence_id']
            X = features_df[feature_cols].values
            y = products
            
            # Encode labels
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            
            # Cross-validation for this bin
            if model_name == 'Random Forest':
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            elif model_name == 'SVM':
                model = SVC(kernel='rbf', random_state=42, probability=True)
            
            cv_scores = cross_val_score(model, X, y_encoded, cv=3, scoring='accuracy')
            
            robustness_results[bin_label] = {
                'accuracy_mean': cv_scores.mean(),
                'accuracy_std': cv_scores.std(),
                'n_samples': len(X)
            }
            
            print(f"  Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return robustness_results
    
    def run_comprehensive_validation(self, 
                                   uniprot_limit: int = 100,
                                   ncbi_limit: int = 100) -> Dict[str, any]:
        """
        Run comprehensive validation of the terpene synthase predictor.
        
        Args:
            uniprot_limit: Maximum number of UniProt entries
            ncbi_limit: Maximum number of NCBI entries
            
        Returns:
            Dictionary with all validation results
        """
        print("="*60)
        print("COMPREHENSIVE VALIDATION OF TERPENE SYNTHASE PREDICTOR")
        print("="*60)
        
        # Step 1: Create validation dataset
        validation_df = self.create_validation_dataset(uniprot_limit, ncbi_limit)
        
        if len(validation_df) < 20:
            print("Warning: Validation dataset is too small for reliable validation")
            return {}
        
        # Step 2: Cross-validation
        cv_results = self.cross_validate_model(validation_df, 'Random Forest')
        
        # Step 3: Test with known products
        test_results = self.test_with_known_products(validation_df, 'Random Forest')
        
        # Step 4: Feature importance analysis
        importance_df = self.analyze_feature_importance(validation_df, 'Random Forest')
        
        # Step 5: Sequence length robustness
        robustness_results = self.test_sequence_length_robustness(validation_df, 'Random Forest')
        
        # Compile all results
        validation_summary = {
            'dataset_info': {
                'total_samples': len(validation_df),
                'unique_products': len(validation_df['product'].unique()),
                'product_distribution': validation_df['product'].value_counts().to_dict()
            },
            'cross_validation': cv_results,
            'test_results': test_results,
            'feature_importance': importance_df.to_dict() if not importance_df.empty else {},
            'robustness': robustness_results
        }
        
        # Save results
        import json
        with open('data/validation_results.json', 'w') as f:
            json.dump(validation_summary, f, indent=2, default=str)
        
        print("\n" + "="*60)
        print("VALIDATION COMPLETE")
        print("="*60)
        print(f"Results saved to data/validation_results.json")
        
        return validation_summary


def main():
    """Main function to run validation."""
    # Initialize validator
    validator = SimpleTerpeneSynthaseValidator(email="your_email@example.com")
    
    # Run comprehensive validation
    results = validator.run_comprehensive_validation(uniprot_limit=50, ncbi_limit=50)
    
    if results:
        print("\nValidation Summary:")
        print(f"Dataset: {results['dataset_info']['total_samples']} samples, {results['dataset_info']['unique_products']} products")
        print(f"Cross-validation accuracy: {results['cross_validation']['accuracy_mean']:.4f} ± {results['cross_validation']['accuracy_std']:.4f}")
        print(f"Test accuracy: {results['test_results']['accuracy']:.4f}")


if __name__ == "__main__":
    main()
