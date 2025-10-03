"""
Model training module for terpene synthase product prediction.
This module implements various machine learning models for predicting enzyme products.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.multioutput import MultiOutputClassifier
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class TerpeneSynthasePredictor:
    """
    Main class for training and evaluating terpene synthase product prediction models.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the predictor.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
        
    def prepare_data(self, features_df: pd.DataFrame, products_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training.
        
        Args:
            features_df: DataFrame with protein features
            products_df: DataFrame with product annotations
            
        Returns:
            Tuple of (X, y) arrays
        """
        print("Preparing data for training...")
        
        # Merge features and products
        merged_df = pd.merge(features_df, products_df, on='sequence_id', how='inner')
        
        # Separate features and labels
        feature_cols = [col for col in merged_df.columns if col not in ['sequence_id', 'product']]
        X = merged_df[feature_cols].values
        y = merged_df['product'].values
        
        # Store feature names
        self.feature_names = feature_cols
        
        # Encode labels
        if 'product' not in self.label_encoders:
            self.label_encoders['product'] = LabelEncoder()
            y_encoded = self.label_encoders['product'].fit_transform(y)
        else:
            y_encoded = self.label_encoders['product'].transform(y)
        
        print(f"Data shape: X={X.shape}, y={y_encoded.shape}")
        print(f"Number of unique products: {len(np.unique(y_encoded))}")
        
        return X, y_encoded
    
    def train_models(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train multiple models and compare their performance.
        
        Args:
            X: Feature matrix
            y: Target labels
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with model performance results
        """
        print("Training multiple models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models to train
        models_to_train = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            ),
            'SVM': SVC(
                kernel='rbf',
                random_state=self.random_state,
                probability=True
            ),
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                random_state=self.random_state,
                max_iter=500
            )
        }
        
        # Train and evaluate models
        results = {}
        
        for name, model in tqdm(models_to_train.items(), desc="Training models"):
            print(f"\nTraining {name}...")
            
            try:
                # Train model
                if name in ['SVM', 'Logistic Regression', 'Neural Network']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                
                # Cross-validation score
                if name in ['SVM', 'Logistic Regression', 'Neural Network']:
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                else:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                
                # Store results
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'y_test': y_test
                }
                
                print(f"{name} - Accuracy: {accuracy:.4f}, CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                
            except Exception as e:
                print(f"Error training {name}: {e}")
                results[name] = {'error': str(e)}
        
        # Store trained models
        self.models = results
        self.is_trained = True
        
        return results
    
    def hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray, model_name: str = 'Random Forest') -> Dict[str, Any]:
        """
        Perform hyperparameter tuning for a specific model.
        
        Args:
            X: Feature matrix
            y: Target labels
            model_name: Name of the model to tune
            
        Returns:
            Dictionary with tuning results
        """
        print(f"Performing hyperparameter tuning for {model_name}...")
        
        # Define parameter grids
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            },
            'LightGBM': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 50, 100]
            }
        }
        
        if model_name not in param_grids:
            print(f"Hyperparameter tuning not implemented for {model_name}")
            return {}
        
        # Initialize model
        if model_name == 'Random Forest':
            model = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        elif model_name == 'XGBoost':
            model = xgb.XGBClassifier(random_state=self.random_state, n_jobs=-1)
        elif model_name == 'LightGBM':
            model = lgb.LGBMClassifier(random_state=self.random_state, n_jobs=-1, verbose=-1)
        
        # Perform grid search
        grid_search = GridSearchCV(
            model, 
            param_grids[model_name], 
            cv=5, 
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_:.4f}")
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_model': grid_search.best_estimator_
        }
    
    def evaluate_model(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate a specific model.
        
        Args:
            model_name: Name of the model to evaluate
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation results
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return {}
        
        model_info = self.models[model_name]
        if 'error' in model_info:
            print(f"Model {model_name} has errors: {model_info['error']}")
            return {}
        
        model = model_info['model']
        
        # Make predictions
        if model_name in ['SVM', 'Logistic Regression', 'Neural Network']:
            X_test_scaled = self.scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
        else:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def get_feature_importance(self, model_name: str, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance for tree-based models.
        
        Args:
            model_name: Name of the model
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return pd.DataFrame()
        
        model_info = self.models[model_name]
        if 'error' in model_info:
            print(f"Model {model_name} has errors: {model_info['error']}")
            return pd.DataFrame()
        
        model = model_info['model']
        
        # Check if model has feature_importances_
        if not hasattr(model, 'feature_importances_'):
            print(f"Model {model_name} does not support feature importance")
            return pd.DataFrame()
        
        # Get feature importance
        importance = model.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df
    
    def plot_feature_importance(self, model_name: str, top_n: int = 20, figsize: Tuple[int, int] = (10, 8)):
        """
        Plot feature importance for a model.
        
        Args:
            model_name: Name of the model
            top_n: Number of top features to plot
            figsize: Figure size
        """
        importance_df = self.get_feature_importance(model_name, top_n)
        
        if importance_df.empty:
            return
        
        plt.figure(figsize=figsize)
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title(f'Feature Importance - {model_name}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, model_name: str, figsize: Tuple[int, int] = (8, 6)):
        """
        Plot confusion matrix for a model.
        
        Args:
            model_name: Name of the model
            figsize: Figure size
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return
        
        model_info = self.models[model_name]
        if 'error' in model_info:
            print(f"Model {model_name} has errors: {model_info['error']}")
            return
        
        y_test = model_info['y_test']
        y_pred = model_info['predictions']
        
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
    
    def save_model(self, model_name: str, filename: Optional[str] = None):
        """
        Save a trained model to disk.
        
        Args:
            model_name: Name of the model to save
            filename: Optional custom filename
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return
        
        model_info = self.models[model_name]
        if 'error' in model_info:
            print(f"Model {model_name} has errors: {model_info['error']}")
            return
        
        if filename is None:
            filename = f"{model_name.lower().replace(' ', '_')}_model.pkl"
        
        os.makedirs("models", exist_ok=True)
        filepath = os.path.join("models", filename)
        
        # Save model and related objects
        model_data = {
            'model': model_info['model'],
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, model_name: str, filename: Optional[str] = None):
        """
        Load a trained model from disk.
        
        Args:
            model_name: Name of the model to load
            filename: Optional custom filename
        """
        if filename is None:
            filename = f"{model_name.lower().replace(' ', '_')}_model.pkl"
        
        filepath = os.path.join("models", filename)
        
        if not os.path.exists(filepath):
            print(f"Model file {filepath} not found")
            return
        
        # Load model data
        model_data = joblib.load(filepath)
        
        # Restore model and related objects
        self.models[model_name] = {'model': model_data['model']}
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")
    
    def predict(self, X: np.ndarray, model_name: str = 'Random Forest') -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using a trained model.
        
        Args:
            X: Feature matrix
            model_name: Name of the model to use
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if not self.is_trained or model_name not in self.models:
            print(f"Model {model_name} not trained or not found")
            return np.array([]), np.array([])
        
        model_info = self.models[model_name]
        if 'error' in model_info:
            print(f"Model {model_name} has errors: {model_info['error']}")
            return np.array([]), np.array([])
        
        model = model_info['model']
        
        # Make predictions
        if model_name in ['SVM', 'Logistic Regression', 'Neural Network']:
            X_scaled = self.scaler.transform(X)
            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)
        else:
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)
        
        return predictions, probabilities


def main():
    """Main function to demonstrate model training."""
    # This would typically load real data
    print("Terpene Synthase Predictor - Model Training Module")
    print("This module provides functionality to train and evaluate models for predicting terpene synthase products.")
    print("Use the data_collector.py and feature_extractor.py modules to prepare your data first.")


if __name__ == "__main__":
    main()
