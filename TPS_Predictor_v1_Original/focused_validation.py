"""
Focused validation script for terpene synthase product prediction.
This script addresses the issues found in the initial validation.
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
from data_collector import TerpeneSynthaseDataCollector
from feature_extractor import ProteinFeatureExtractor
import warnings
warnings.filterwarnings('ignore')


class FocusedTerpeneSynthaseValidator:
    """
    Focused validation class that addresses the issues found in initial validation.
    """
    
    def __init__(self, email: str = "your_email@example.com"):
        """Initialize the validator."""
        self.data_collector = TerpeneSynthaseDataCollector(email)
        self.feature_extractor = ProteinFeatureExtractor()
        
    def clean_product_names(self, products: List[str]) -> List[str]:
        """
        Clean and standardize product names.
        
        Args:
            products: List of product names
            
        Returns:
            List of cleaned product names
        """
        cleaned_products = []
        
        for product in products:
            # Convert to lowercase
            product = product.lower().strip()
            
            # Skip empty or very short products
            if len(product) < 3:
                cleaned_products.append('unknown')
                continue
            
            # Standardize common names
            if 'limonene' in product or 'limone' in product:
                product = 'limonene'
            elif 'pinene' in product:
                product = 'pinene'
            elif 'myrcene' in product:
                product = 'myrcene'
            elif 'linalool' in product:
                product = 'linalool'
            elif 'geraniol' in product:
                product = 'geraniol'
            elif 'caryophyllene' in product:
                product = 'caryophyllene'
            elif 'humulene' in product:
                product = 'humulene'
            elif 'farnesene' in product:
                product = 'farnesene'
            elif 'bisabolene' in product:
                product = 'bisabolene'
            elif 'squalene' in product:
                product = 'squalene'
            elif 'sabinene' in product:
                product = 'sabinene'
            elif 'terpinolene' in product:
                product = 'terpinolene'
            elif 'terpineol' in product:
                product = 'terpineol'
            elif 'germacrene' in product:
                product = 'germacrene'
            elif 'thujene' in product:
                product = 'thujene'
            elif 'ocimene' in product:
                product = 'ocimene'
            elif 'cadinene' in product:
                product = 'cadinene'
            elif 'phellandrene' in product:
                product = 'phellandrene'
            elif 'copaene' in product:
                product = 'copaene'
            elif 'camphene' in product:
                product = 'camphene'
            elif 'selinene' in product:
                product = 'selinene'
            elif 'carvacrol' in product:
                product = 'carvacrol'
            elif 'thymol' in product:
                product = 'thymol'
            else:
                # Keep original if no match
                product = product
            
            cleaned_products.append(product)
        
        return cleaned_products
    
    def create_focused_validation_dataset(self, 
                                        uniprot_limit: int = 100,
                                        ncbi_limit: int = 100,
                                        min_sequence_length: int = 200,
                                        max_sequence_length: int = 1000,
                                        min_samples_per_class: int = 5) -> pd.DataFrame:
        """
        Create a focused validation dataset with cleaned data.
        
        Args:
            uniprot_limit: Maximum number of UniProt entries
            ncbi_limit: Maximum number of NCBI entries
            min_sequence_length: Minimum sequence length
            max_sequence_length: Maximum sequence length
            min_samples_per_class: Minimum samples per product class
            
        Returns:
            DataFrame with focused validation data
        """
        print("Creating focused validation dataset...")
        
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
        
        # Clean product names
        print("Cleaning product names...")
        validation_df['cleaned_product'] = self.clean_product_names(validation_df['product'].tolist())
        
        # Filter by minimum samples per class
        product_counts = validation_df['cleaned_product'].value_counts()
        valid_products = product_counts[product_counts >= min_samples_per_class].index
        
        print(f"Products with at least {min_samples_per_class} samples:")
        for product in valid_products:
            print(f"  {product}: {product_counts[product]} samples")
        
        # Filter to valid products only
        focused_df = validation_df[validation_df['cleaned_product'].isin(valid_products)].copy()
        
        print(f"Focused dataset: {len(focused_df)} samples, {len(valid_products)} products")
        
        # Save focused dataset
        os.makedirs("data", exist_ok=True)
        focused_df.to_csv("data/focused_validation_dataset.csv", index=False)
        
        return focused_df
    
    def extract_simple_features(self, sequences: List[str]) -> pd.DataFrame:
        """
        Extract simple, focused features from protein sequences.
        
        Args:
            sequences: List of protein sequences
            
        Returns:
            DataFrame with simple features
        """
        print("Extracting simple features...")
        
        features_list = []
        
        for i, sequence in enumerate(sequences):
            # Basic features
            feature_dict = {
                'sequence_id': f'seq_{i}',
                'length': len(sequence),
                'A_count': sequence.count('A'),
                'C_count': sequence.count('C'),
                'D_count': sequence.count('D'),
                'E_count': sequence.count('E'),
                'F_count': sequence.count('F'),
                'G_count': sequence.count('G'),
                'H_count': sequence.count('H'),
                'I_count': sequence.count('I'),
                'K_count': sequence.count('K'),
                'L_count': sequence.count('L'),
                'M_count': sequence.count('M'),
                'N_count': sequence.count('N'),
                'P_count': sequence.count('P'),
                'Q_count': sequence.count('Q'),
                'R_count': sequence.count('R'),
                'S_count': sequence.count('S'),
                'T_count': sequence.count('T'),
                'V_count': sequence.count('V'),
                'W_count': sequence.count('W'),
                'Y_count': sequence.count('Y'),
            }
            
            # Calculate percentages
            total = len(sequence)
            for aa in 'ACDEFGHIKLMNPQRSTVWY':
                feature_dict[f'{aa}_percent'] = feature_dict[f'{aa}_count'] / total if total > 0 else 0
            
            # Motif features
            feature_dict['DDXXD_motif'] = 1 if 'DDDAD' in sequence else 0
            feature_dict['NSE_motif'] = 1 if 'NSE' in sequence else 0
            feature_dict['RR_motif'] = 1 if 'RRRRR' in sequence else 0
            feature_dict['GG_motif'] = 1 if 'GGGGG' in sequence else 0
            feature_dict['HH_motif'] = 1 if 'HHHHH' in sequence else 0
            
            # Simple k-mer features (2-mers only)
            kmers_2 = [sequence[i:i+2] for i in range(len(sequence) - 1)]
            kmer_counts = pd.Series(kmers_2).value_counts()
            total_kmers = len(kmers_2)
            
            # Add top 20 most common 2-mers
            top_kmers = kmer_counts.head(20)
            for kmer, count in top_kmers.items():
                feature_dict[f'kmer_2_{kmer}'] = count / total_kmers if total_kmers > 0 else 0
            
            features_list.append(feature_dict)
        
        return pd.DataFrame(features_list)
    
    def validate_focused_model(self, 
                             validation_df: pd.DataFrame,
                             model_name: str = 'Random Forest',
                             cv_folds: int = 5,
                             max_features: int = 100) -> Dict[str, any]:
        """
        Validate the model with focused features and data.
        
        Args:
            validation_df: Focused validation dataset
            model_name: Name of the model to validate
            cv_folds: Number of cross-validation folds
            max_features: Maximum number of features to use
            
        Returns:
            Dictionary with validation results
        """
        print(f"\nValidating {model_name} with focused approach...")
        
        # Prepare data
        sequences = validation_df['sequence'].tolist()
        products = validation_df['cleaned_product'].tolist()
        
        # Extract simple features
        features_df = self.extract_simple_features(sequences)
        
        # Prepare features and labels
        feature_cols = [col for col in features_df.columns if col != 'sequence_id']
        X = features_df[feature_cols].values
        y = products
        
        print(f"Initial features: {X.shape[1]}")
        
        # Handle NaN values
        print("Handling NaN values...")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Feature selection
        if X.shape[1] > max_features:
            print(f"Selecting top {max_features} features...")
            selector = SelectKBest(f_classif, k=max_features)
            X_selected = selector.fit_transform(X, y)
            selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
            X = X_selected
            print(f"Selected features: {X.shape[1]}")
        else:
            selected_features = feature_cols
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        print(f"Data shape: X={X.shape}, y={y_encoded.shape}")
        print(f"Unique products: {len(np.unique(y_encoded))}")
        print(f"Product distribution:")
        for i, product in enumerate(label_encoder.classes_):
            count = np.sum(y_encoded == i)
            print(f"  {product}: {count} samples")
        
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
        
        print(f"Cross-validation results:")
        print(f"  Accuracy: {validation_metrics['accuracy_mean']:.4f} ± {validation_metrics['accuracy_std']:.4f}")
        print(f"  Precision: {validation_metrics['precision_mean']:.4f} ± {validation_metrics['precision_std']:.4f}")
        print(f"  Recall: {validation_metrics['recall_mean']:.4f} ± {validation_metrics['recall_std']:.4f}")
        print(f"  F1-score: {validation_metrics['f1_mean']:.4f} ± {validation_metrics['f1_std']:.4f}")
        
        # Train final model and get feature importance
        model.fit(X, y_encoded)
        importance = model.feature_importances_
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': selected_features,
            'importance': importance
        }).sort_values('importance', ascending=False).head(20)
        
        print(f"\nTop 20 most important features:")
        print(importance_df)
        
        return {
            'validation_metrics': validation_metrics,
            'feature_importance': importance_df,
            'selected_features': selected_features,
            'label_encoder': label_encoder
        }
    
    def run_focused_validation(self, 
                             uniprot_limit: int = 100,
                             ncbi_limit: int = 100) -> Dict[str, any]:
        """
        Run focused validation of the terpene synthase predictor.
        
        Args:
            uniprot_limit: Maximum number of UniProt entries
            ncbi_limit: Maximum number of NCBI entries
            
        Returns:
            Dictionary with validation results
        """
        print("="*60)
        print("FOCUSED VALIDATION OF TERPENE SYNTHASE PREDICTOR")
        print("="*60)
        
        # Step 1: Create focused validation dataset
        validation_df = self.create_focused_validation_dataset(uniprot_limit, ncbi_limit)
        
        if len(validation_df) < 20:
            print("Warning: Validation dataset is too small for reliable validation")
            return {}
        
        # Step 2: Validate with focused approach
        results = self.validate_focused_model(validation_df, 'Random Forest')
        
        # Compile results
        validation_summary = {
            'dataset_info': {
                'total_samples': len(validation_df),
                'unique_products': len(validation_df['cleaned_product'].unique()),
                'product_distribution': validation_df['cleaned_product'].value_counts().to_dict()
            },
            'validation_results': results
        }
        
        # Save results
        import json
        with open('data/focused_validation_results.json', 'w') as f:
            json.dump(validation_summary, f, indent=2, default=str)
        
        print("\n" + "="*60)
        print("FOCUSED VALIDATION COMPLETE")
        print("="*60)
        print(f"Results saved to data/focused_validation_results.json")
        
        return validation_summary


def main():
    """Main function to run focused validation."""
    # Initialize validator
    validator = FocusedTerpeneSynthaseValidator(email="your_email@example.com")
    
    # Run focused validation
    results = validator.run_focused_validation(uniprot_limit=100, ncbi_limit=100)
    
    if results:
        print("\nFocused Validation Summary:")
        print(f"Dataset: {results['dataset_info']['total_samples']} samples, {results['dataset_info']['unique_products']} products")
        print(f"Cross-validation accuracy: {results['validation_results']['validation_metrics']['accuracy_mean']:.4f} ± {results['validation_results']['validation_metrics']['accuracy_std']:.4f}")


if __name__ == "__main__":
    main()
