"""
Test single-product vs multi-product training approaches.
This addresses whether focusing on one product (germacrene) or training on multiple products
gives better performance for terpene synthase prediction.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')


class SingleVsMultiProductTester:
    """
    Test whether single-product or multi-product training is better for terpene synthase prediction.
    """
    
    def __init__(self, model_name: str = "facebook/esm2_t6_8M_UR50D"):
        """Initialize the tester."""
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        print(f"Using model: {model_name}")
        
    def load_protein_model(self):
        """Load the protein language model."""
        try:
            print(f"Loading protein language model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def extract_protein_embeddings(self, sequences: List[str], batch_size: int = 16) -> np.ndarray:
        """Extract protein embeddings using the language model."""
        if self.model is None:
            if not self.load_protein_model():
                return None
        
        print(f"Extracting embeddings for {len(sequences)} sequences...")
        
        all_embeddings = []
        
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i+batch_size]
            
            # Tokenize sequences
            inputs = self.tokenizer(
                batch_sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling of last hidden states
                embeddings = outputs.last_hidden_state.mean(dim=1)
                all_embeddings.append(embeddings.cpu().numpy())
        
        # Concatenate all embeddings
        embeddings_array = np.vstack(all_embeddings)
        print(f"Extracted embeddings shape: {embeddings_array.shape}")
        
        return embeddings_array
    
    def create_single_product_dataset(self, target_product: str = "germacrene", size: int = 200) -> pd.DataFrame:
        """
        Create dataset for single-product prediction (binary classification).
        """
        print(f"Creating single-product dataset for {target_product}...")
        
        # Define sequences for different products
        product_sequences = {
            'germacrene': ["MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELGERMAC"],
            'limonene': ["MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELDDXXD"],
            'pinene': ["MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELNSE"],
            'myrcene': ["MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELRRX8W"],
            'linalool': ["MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELGXGXXG"],
            'geraniol': ["MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELHXXXH"],
            'caryophyllene': ["MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELCARYOPH"],
            'humulene': ["MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELHUMULEN"],
            'farnesene': ["MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELFARNESE"]
        }
        
        all_data = []
        
        # Create positive examples (target product)
        target_sequences = product_sequences[target_product]
        positive_size = size // 2
        
        for i in range(positive_size):
            base_seq = target_sequences[0]
            mutated_seq = self._add_sequence_variation(base_seq, mutation_rate=0.05)
            all_data.append({
                'sequence': mutated_seq,
                'product': target_product,
                'is_target': 1
            })
        
        # Create negative examples (other products)
        other_products = [p for p in product_sequences.keys() if p != target_product]
        negative_size = size - positive_size
        
        for i in range(negative_size):
            # Randomly select a non-target product
            other_product = np.random.choice(other_products)
            base_seq = product_sequences[other_product][0]
            mutated_seq = self._add_sequence_variation(base_seq, mutation_rate=0.05)
            all_data.append({
                'sequence': mutated_seq,
                'product': other_product,
                'is_target': 0
            })
        
        df = pd.DataFrame(all_data)
        
        print(f"Created single-product dataset with {len(df)} sequences")
        print(f"Target product: {target_product}")
        print(f"Positive examples: {df['is_target'].sum()}")
        print(f"Negative examples: {len(df) - df['is_target'].sum()}")
        
        return df
    
    def create_multi_product_dataset(self, size: int = 200) -> pd.DataFrame:
        """
        Create dataset for multi-product prediction (multi-class classification).
        """
        print(f"Creating multi-product dataset...")
        
        # Define terpene synthase families
        terpene_families = {
            'germacrene': ["MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELGERMAC"],
            'limonene': ["MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELDDXXD"],
            'pinene': ["MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELNSE"],
            'myrcene': ["MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELRRX8W"],
            'linalool': ["MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELGXGXXG"],
            'geraniol': ["MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELHXXXH"],
            'caryophyllene': ["MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELCARYOPH"],
            'humulene': ["MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELHUMULEN"],
            'farnesene': ["MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELFARNESE"]
        }
        
        # Generate dataset
        all_data = []
        sequences_per_product = size // len(terpene_families)
        
        for product, sequences in terpene_families.items():
            for i in range(sequences_per_product):
                base_seq = sequences[0]
                mutated_seq = self._add_sequence_variation(base_seq, mutation_rate=0.05)
                
                all_data.append({
                    'sequence': mutated_seq,
                    'product': product,
                    'family': product
                })
        
        df = pd.DataFrame(all_data)
        
        print(f"Created multi-product dataset with {len(df)} sequences")
        print(f"Product distribution:")
        print(df['product'].value_counts())
        
        return df
    
    def _add_sequence_variation(self, sequence: str, mutation_rate: float = 0.05) -> str:
        """Add random variations to a sequence."""
        amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
        mutated = list(sequence)
        
        for i in range(len(mutated)):
            if np.random.random() < mutation_rate:
                mutated[i] = np.random.choice(amino_acids)
        
        return ''.join(mutated)
    
    def test_single_product_approach(self, target_product: str = "germacrene", size: int = 200) -> Dict[str, any]:
        """
        Test single-product prediction approach (binary classification).
        """
        print(f"\n{'='*60}")
        print(f"SINGLE-PRODUCT APPROACH: {target_product.upper()}")
        print(f"{'='*60}")
        
        # Create dataset
        df = self.create_single_product_dataset(target_product, size)
        
        # Prepare data
        sequences = df['sequence'].tolist()
        labels = df['is_target'].tolist()
        
        # Extract embeddings
        embeddings = self.extract_protein_embeddings(sequences)
        if embeddings is None:
            return None
        
        print(f"Data shape: X={embeddings.shape}, y={len(labels)}")
        print(f"Positive examples: {sum(labels)}")
        print(f"Negative examples: {len(labels) - sum(labels)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=0.3, random_state=42, stratify=labels
        )
        
        # Test different models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'SVM': SVC(kernel='rbf', random_state=42, probability=True),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(256, 128), random_state=42, max_iter=1000)
        }
        
        results = {}
        
        for model_name, model in models.items():
            print(f"\nTesting {model_name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, embeddings, labels, cv=5, scoring='accuracy')
            
            results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-score: {f1:.4f}")
            print(f"  CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
        best_f1 = results[best_model_name]['f1_score']
        
        print(f"\nBest model: {best_model_name}")
        print(f"Best F1-score: {best_f1:.4f}")
        
        return {
            'approach': 'single_product',
            'target_product': target_product,
            'results': results,
            'best_model_name': best_model_name,
            'best_f1_score': best_f1,
            'embeddings_shape': embeddings.shape
        }
    
    def test_multi_product_approach(self, size: int = 200) -> Dict[str, any]:
        """
        Test multi-product prediction approach (multi-class classification).
        """
        print(f"\n{'='*60}")
        print(f"MULTI-PRODUCT APPROACH")
        print(f"{'='*60}")
        
        # Create dataset
        df = self.create_multi_product_dataset(size)
        
        # Prepare data
        sequences = df['sequence'].tolist()
        products = df['product'].tolist()
        
        # Extract embeddings
        embeddings = self.extract_protein_embeddings(sequences)
        if embeddings is None:
            return None
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(products)
        
        print(f"Data shape: X={embeddings.shape}, y={y_encoded.shape}")
        print(f"Unique products: {len(np.unique(y_encoded))}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )
        
        # Test different models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'SVM': SVC(kernel='rbf', random_state=42, probability=True),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(512, 256), random_state=42, max_iter=1000)
        }
        
        results = {}
        
        for model_name, model in models.items():
            print(f"\nTesting {model_name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')
            
            # Cross-validation
            cv_scores = cross_val_score(model, embeddings, y_encoded, cv=5, scoring='accuracy')
            
            results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-score: {f1:.4f}")
            print(f"  CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
        best_f1 = results[best_model_name]['f1_score']
        
        print(f"\nBest model: {best_model_name}")
        print(f"Best F1-score: {best_f1:.4f}")
        
        return {
            'approach': 'multi_product',
            'results': results,
            'best_model_name': best_model_name,
            'best_f1_score': best_f1,
            'label_encoder': label_encoder,
            'embeddings_shape': embeddings.shape
        }
    
    def compare_approaches(self, target_product: str = "germacrene", size: int = 200) -> Dict[str, any]:
        """
        Compare single-product vs multi-product approaches.
        """
        print(f"\n{'='*60}")
        print(f"COMPARING SINGLE-PRODUCT VS MULTI-PRODUCT APPROACHES")
        print(f"{'='*60}")
        print(f"Target product: {target_product}")
        print(f"Dataset size: {size}")
        
        # Test single-product approach
        single_results = self.test_single_product_approach(target_product, size)
        
        # Test multi-product approach
        multi_results = self.test_multi_product_approach(size)
        
        # Compare results
        print(f"\n{'='*60}")
        print(f"COMPARISON RESULTS")
        print(f"{'='*60}")
        
        print(f"\nSingle-Product Approach ({target_product}):")
        print(f"  Best model: {single_results['best_model_name']}")
        print(f"  Best F1-score: {single_results['best_f1_score']:.4f}")
        print(f"  Embeddings: {single_results['embeddings_shape'][1]}D")
        
        print(f"\nMulti-Product Approach:")
        print(f"  Best model: {multi_results['best_model_name']}")
        print(f"  Best F1-score: {multi_results['best_f1_score']:.4f}")
        print(f"  Embeddings: {multi_results['embeddings_shape'][1]}D")
        
        # Analysis
        print(f"\n{'='*60}")
        print(f"ANALYSIS AND RECOMMENDATIONS")
        print(f"{'='*60}")
        
        single_f1 = single_results['best_f1_score']
        multi_f1 = multi_results['best_f1_score']
        
        if single_f1 > multi_f1:
            print(f"✅ Single-product approach is BETTER")
            print(f"   F1-score: {single_f1:.4f} vs {multi_f1:.4f}")
            print(f"   Improvement: {((single_f1 - multi_f1) / multi_f1 * 100):.1f}%")
            print(f"\nRecommendations:")
            print(f"  🎯 Focus on single-product prediction")
            print(f"  🔬 Train separate models for each product")
            print(f"  📊 Use binary classification for better discrimination")
        else:
            print(f"✅ Multi-product approach is BETTER")
            print(f"   F1-score: {multi_f1:.4f} vs {single_f1:.4f}")
            print(f"   Improvement: {((multi_f1 - single_f1) / single_f1 * 100):.1f}%")
            print(f"\nRecommendations:")
            print(f"  🎯 Use multi-product prediction")
            print(f"  🔬 Train one model for all products")
            print(f"  📊 Use multi-class classification")
        
        print(f"\nKey Insights:")
        print(f"  • Single-product: Binary classification, focused learning")
        print(f"  • Multi-product: Multi-class classification, comparative learning")
        print(f"  • Choice depends on your specific use case")
        
        return {
            'single_product_results': single_results,
            'multi_product_results': multi_results,
            'comparison': {
                'single_f1': single_f1,
                'multi_f1': multi_f1,
                'winner': 'single_product' if single_f1 > multi_f1 else 'multi_product'
            }
        }


def main():
    """Main function to compare approaches."""
    print("Single-Product vs Multi-Product Terpene Synthase Prediction")
    print("="*60)
    
    # Initialize tester
    tester = SingleVsMultiProductTester()
    
    # Compare approaches
    results = tester.compare_approaches(target_product="germacrene", size=200)
    
    print(f"\n{'='*60}")
    print(f"FINAL RECOMMENDATION")
    print(f"{'='*60}")
    
    winner = results['comparison']['winner']
    single_f1 = results['comparison']['single_f1']
    multi_f1 = results['comparison']['multi_f1']
    
    if winner == 'single_product':
        print(f"🎯 SINGLE-PRODUCT APPROACH RECOMMENDED")
        print(f"   Better performance: {single_f1:.4f} vs {multi_f1:.4f}")
        print(f"   Focus on one product at a time")
        print(f"   Use binary classification")
    else:
        print(f"🎯 MULTI-PRODUCT APPROACH RECOMMENDED")
        print(f"   Better performance: {multi_f1:.4f} vs {single_f1:.4f}")
        print(f"   Train on multiple products")
        print(f"   Use multi-class classification")


if __name__ == "__main__":
    main()
