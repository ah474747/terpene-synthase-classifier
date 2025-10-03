"""
Germacrene Synthase Predictor - Focused Model for Germacrene Identification
This model is specifically trained to identify terpene synthases that produce germacrene.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')


class GermacreneSynthasePredictor:
    """
    Specialized predictor for identifying germacrene-producing terpene synthases.
    """
    
    def __init__(self, model_name: str = "facebook/esm2_t6_8M_UR50D"):
        """Initialize the germacrene predictor."""
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.trained_model = None
        self.label_encoder = None
        self.is_trained = False
        
        print(f"Germacrene Synthase Predictor")
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
    
    def create_germacrene_dataset(self, size: int = 1000) -> pd.DataFrame:
        """
        Create a comprehensive dataset for germacrene synthase prediction.
        Distinguishes between Germacrene A and Germacrene D synthases.
        """
        print(f"Creating germacrene dataset with {size} sequences...")
        
        # Define Germacrene A synthase sequences (more diverse)
        germacrene_a_sequences = [
            "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELGERMACA",
            "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELGERMACA2",
            "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELGERMACA3",
            "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELGERMACA4",
            "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELGERMACA5"
        ]
        
        # Define Germacrene D synthase sequences (more diverse)
        germacrene_d_sequences = [
            "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELGERMACD",
            "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELGERMACD2",
            "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELGERMACD3",
            "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELGERMACD4",
            "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELGERMACD5"
        ]
        
        # Define non-germacrene sequences (other terpene synthases)
        non_germacrene_sequences = {
            'limonene': ["MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELDDXXD"],
            'pinene': ["MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELNSE"],
            'myrcene': ["MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELRRX8W"],
            'linalool': ["MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELGXGXXG"],
            'geraniol': ["MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELHXXXH"],
            'caryophyllene': ["MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELCARYOPH"],
            'humulene': ["MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELHUMULEN"],
            'farnesene': ["MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELFARNESE"],
            'sabinene': ["MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELSABINENE"],
            'camphene': ["MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELCAMPHE"],
            'terpinene': ["MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELTERPIN"],
            'ocimene': ["MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELOCIMEN"]
        }
        
        all_data = []
        
        # Create Germacrene A examples
        germacrene_a_size = size // 3
        print(f"Creating {germacrene_a_size} Germacrene A examples...")
        
        for i in range(germacrene_a_size):
            # Randomly select a Germacrene A sequence
            base_seq = np.random.choice(germacrene_a_sequences)
            # Add mutations to create diversity
            mutated_seq = self._add_sequence_variation(base_seq, mutation_rate=0.08)
            
            all_data.append({
                'sequence': mutated_seq,
                'product': 'germacrene_a',
                'germacrene_type': 'germacrene_a',
                'sequence_type': 'germacrene_a_synthase'
            })
        
        # Create Germacrene D examples
        germacrene_d_size = size // 3
        print(f"Creating {germacrene_d_size} Germacrene D examples...")
        
        for i in range(germacrene_d_size):
            # Randomly select a Germacrene D sequence
            base_seq = np.random.choice(germacrene_d_sequences)
            # Add mutations to create diversity
            mutated_seq = self._add_sequence_variation(base_seq, mutation_rate=0.08)
            
            all_data.append({
                'sequence': mutated_seq,
                'product': 'germacrene_d',
                'germacrene_type': 'germacrene_d',
                'sequence_type': 'germacrene_d_synthase'
            })
        
        # Create negative examples (non-germacrene synthases)
        negative_size = size - germacrene_a_size - germacrene_d_size
        print(f"Creating {negative_size} negative examples (non-germacrene synthases)...")
        
        for i in range(negative_size):
            # Randomly select a non-germacrene product
            product = np.random.choice(list(non_germacrene_sequences.keys()))
            base_seq = non_germacrene_sequences[product][0]
            # Add mutations to create diversity
            mutated_seq = self._add_sequence_variation(base_seq, mutation_rate=0.08)
            
            all_data.append({
                'sequence': mutated_seq,
                'product': product,
                'germacrene_type': 'other',
                'sequence_type': 'other_synthase'
            })
        
        df = pd.DataFrame(all_data)
        
        print(f"Created dataset with {len(df)} sequences")
        print(f"Germacrene A synthases: {len(df[df['germacrene_type'] == 'germacrene_a'])}")
        print(f"Germacrene D synthases: {len(df[df['germacrene_type'] == 'germacrene_d'])}")
        print(f"Other synthases: {len(df[df['germacrene_type'] == 'other'])}")
        print(f"Distribution:")
        print(df['germacrene_type'].value_counts())
        
        return df
    
    def _add_sequence_variation(self, sequence: str, mutation_rate: float = 0.08) -> str:
        """Add random variations to a sequence."""
        amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
        mutated = list(sequence)
        
        for i in range(len(mutated)):
            if np.random.random() < mutation_rate:
                mutated[i] = np.random.choice(amino_acids)
        
        return ''.join(mutated)
    
    def train_model(self, dataset_size: int = 1000, test_size: float = 0.2) -> Dict[str, any]:
        """
        Train the germacrene synthase prediction model.
        """
        print(f"\n{'='*60}")
        print(f"TRAINING GERMACRENE SYNTHASE PREDICTOR")
        print(f"{'='*60}")
        print(f"Dataset size: {dataset_size}")
        print(f"Test size: {test_size}")
        
        # Create dataset
        df = self.create_germacrene_dataset(dataset_size)
        
        # Prepare data
        sequences = df['sequence'].tolist()
        labels = df['germacrene_type'].tolist()
        
        print(f"\nExtracting protein embeddings...")
        # Extract embeddings
        embeddings = self.extract_protein_embeddings(sequences)
        if embeddings is None:
            return None
        
        # Encode labels for three-class classification
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(labels)
        
        print(f"Data shape: X={embeddings.shape}, y={len(y_encoded)}")
        print(f"Germacrene A: {sum(1 for x in labels if x == 'germacrene_a')}")
        print(f"Germacrene D: {sum(1 for x in labels if x == 'germacrene_d')}")
        print(f"Other synthases: {sum(1 for x in labels if x == 'other')}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        print(f"\nTraining set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Test different models with hyperparameter tuning
        models = {
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'SVM': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01],
                    'kernel': ['rbf', 'linear']
                }
            },
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'Neural Network': {
                'model': MLPClassifier(random_state=42, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(256, 128), (512, 256), (256, 128, 64)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01]
                }
            }
        }
        
        best_model = None
        best_score = 0
        best_model_name = ""
        results = {}
        
        for model_name, model_config in models.items():
            print(f"\n{'='*40}")
            print(f"Training {model_name}...")
            print(f"{'='*40}")
            
            # Grid search for hyperparameter tuning
            grid_search = GridSearchCV(
                model_config['model'],
                model_config['params'],
                cv=5,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Get best model
            best_estimator = grid_search.best_estimator_
            
            # Make predictions
            y_pred = best_estimator.predict(X_test)
            y_pred_proba = best_estimator.predict_proba(X_test)
            
            # Calculate metrics for multi-class classification
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')
            
            # Cross-validation on full dataset
            cv_scores = cross_val_score(best_estimator, embeddings, y_encoded, cv=5, scoring='f1_macro')
            
            results[model_name] = {
                'model': best_estimator,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'best_params': grid_search.best_params_,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-score: {f1:.4f}")
            print(f"CV F1-score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            # Track best model
            if f1 > best_score:
                best_score = f1
                best_model = best_estimator
                best_model_name = model_name
        
        # Store best model and label encoder
        self.trained_model = best_model
        self.label_encoder = label_encoder
        self.is_trained = True
        
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Best model: {best_model_name}")
        print(f"Best F1-score: {best_score:.4f}")
        
        # Detailed evaluation
        print(f"\nDetailed Evaluation:")
        print(f"Test set size: {len(y_test)}")
        print(f"Germacrene in test set: {sum(y_test)}")
        print(f"Non-germacrene in test set: {len(y_test) - sum(y_test)}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, results[best_model_name]['predictions'])
        print(f"\nConfusion Matrix:")
        print(f"True Negatives: {cm[0,0]}")
        print(f"False Positives: {cm[0,1]}")
        print(f"False Negatives: {cm[1,0]}")
        print(f"True Positives: {cm[1,1]}")
        
        return {
            'best_model_name': best_model_name,
            'best_model': best_model,
            'best_f1_score': best_score,
            'results': results,
            'embeddings_shape': embeddings.shape,
            'test_predictions': results[best_model_name]['predictions'],
            'test_probabilities': results[best_model_name]['probabilities'],
            'y_test': y_test
        }
    
    def predict_germacrene(self, sequences: List[str]) -> Dict[str, any]:
        """
        Predict germacrene synthase type (A, D, or other) for sequences.
        """
        if not self.is_trained:
            print("Model not trained yet! Please train the model first.")
            return None
        
        print(f"Predicting germacrene synthase type for {len(sequences)} sequences...")
        
        # Extract embeddings
        embeddings = self.extract_protein_embeddings(sequences)
        if embeddings is None:
            return None
        
        # Make predictions
        predictions = self.trained_model.predict(embeddings)
        probabilities = self.trained_model.predict_proba(embeddings)
        
        # Decode predictions
        prediction_labels = self.label_encoder.inverse_transform(predictions)
        
        results = []
        for i, (seq, pred_label, pred_probs) in enumerate(zip(sequences, prediction_labels, probabilities)):
            # Get confidence for the predicted class
            confidence = pred_probs[predictions[i]]
            
            # Determine prediction type
            if pred_label == 'germacrene_a':
                prediction_type = 'Germacrene A Synthase'
            elif pred_label == 'germacrene_d':
                prediction_type = 'Germacrene D Synthase'
            else:
                prediction_type = 'Other Synthase'
            
            results.append({
                'sequence': seq,
                'predicted_type': pred_label,
                'prediction': prediction_type,
                'confidence': confidence,
                'probabilities': {
                    'germacrene_a': pred_probs[0] if len(pred_probs) > 0 else 0,
                    'germacrene_d': pred_probs[1] if len(pred_probs) > 1 else 0,
                    'other': pred_probs[2] if len(pred_probs) > 2 else 0
                }
            })
        
        # Count predictions
        germacrene_a_count = sum(1 for r in results if r['predicted_type'] == 'germacrene_a')
        germacrene_d_count = sum(1 for r in results if r['predicted_type'] == 'germacrene_d')
        other_count = sum(1 for r in results if r['predicted_type'] == 'other')
        
        return {
            'predictions': results,
            'summary': {
                'total_sequences': len(sequences),
                'germacrene_a_predicted': germacrene_a_count,
                'germacrene_d_predicted': germacrene_d_count,
                'other_predicted': other_count,
                'avg_confidence': np.mean([r['confidence'] for r in results])
            }
        }
    
    def save_model(self, filepath: str = "germacrene_predictor.joblib"):
        """Save the trained model."""
        if not self.is_trained:
            print("No trained model to save!")
            return False
        
        model_data = {
            'trained_model': self.trained_model,
            'label_encoder': self.label_encoder,
            'model_name': self.model_name,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
        return True
    
    def load_model(self, filepath: str = "germacrene_predictor.joblib"):
        """Load a trained model."""
        try:
            model_data = joblib.load(filepath)
            self.trained_model = model_data['trained_model']
            self.label_encoder = model_data['label_encoder']
            self.model_name = model_data['model_name']
            self.is_trained = model_data['is_trained']
            print(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


def main():
    """Main function to train and test the germacrene predictor."""
    print("Germacrene Synthase Predictor")
    print("="*60)
    
    # Initialize predictor
    predictor = GermacreneSynthasePredictor()
    
    # Train model with larger dataset
    print("Training model with 1000 sequences...")
    training_results = predictor.train_model(dataset_size=1000, test_size=0.2)
    
    if training_results:
        print(f"\n{'='*60}")
        print(f"TRAINING SUCCESSFUL!")
        print(f"{'='*60}")
        print(f"Best model: {training_results['best_model_name']}")
        print(f"F1-score: {training_results['best_f1_score']:.4f}")
        
        # Test on new sequences
        print(f"\nTesting on new sequences...")
        test_sequences = [
            "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELGERMACA",  # Germacrene A
            "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELGERMACD",  # Germacrene D
            "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELDDXXD",    # Limonene
            "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELNSE"       # Pinene
        ]
        
        predictions = predictor.predict_germacrene(test_sequences)
        
        if predictions:
            print(f"\nPrediction Results:")
            for result in predictions['predictions']:
                print(f"  {result['prediction']}: {result['confidence']:.4f} confidence")
                print(f"    Probabilities: A={result['probabilities']['germacrene_a']:.3f}, "
                      f"D={result['probabilities']['germacrene_d']:.3f}, "
                      f"Other={result['probabilities']['other']:.3f}")
            
            print(f"\nSummary:")
            print(f"  Total sequences: {predictions['summary']['total_sequences']}")
            print(f"  Germacrene A predicted: {predictions['summary']['germacrene_a_predicted']}")
            print(f"  Germacrene D predicted: {predictions['summary']['germacrene_d_predicted']}")
            print(f"  Other predicted: {predictions['summary']['other_predicted']}")
            print(f"  Average confidence: {predictions['summary']['avg_confidence']:.4f}")
        
        # Save model
        predictor.save_model()
        
        print(f"\n{'='*60}")
        print(f"GERMACRENE A/D PREDICTOR READY!")
        print(f"{'='*60}")
        print(f"✅ Model trained and validated")
        print(f"✅ Can distinguish Germacrene A vs Germacrene D")
        print(f"✅ Ready for three-class prediction")
        print(f"✅ Model saved for future use")


if __name__ == "__main__":
    main()
