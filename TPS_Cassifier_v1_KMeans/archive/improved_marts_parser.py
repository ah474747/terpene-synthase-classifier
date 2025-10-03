#!/usr/bin/env python3
"""
Improved MARTS-DB parser that preserves rich product annotation
"""

import pandas as pd
import re
from typing import Dict, List, Tuple


class ImprovedMARTSDBParser:
    """
    Enhanced parser for MARTS-DB data that preserves rich product annotation
    """
    
    def __init__(self):
        # Comprehensive germacrene-related keywords
        self.germacrene_keywords = [
            'germacrene', 'germacrene-a', 'germacrene-b', 'germacrene-c', 'germacrene-d',
            'germacrene-e', 'germacrene-f', 'germacrene-g', 'germacrene-h',
            'bicyclogermacrene', 'germacra', 'germacradiene', 'germacratriene',
            'germacrone', 'germacrol', 'germacrene-oxide'
        ]
        
        # All terpene product categories for classification
        self.product_categories = {
            'monoterpenes': ['limonene', 'pinene', 'myrcene', 'ocimene', 'sabinene', 'thujene', 'terpinene'],
            'sesquiterpenes': ['caryophyllene', 'humulene', 'farnesene', 'bisabolene', 'selinene', 'eudesmol'],
            'germacrenes': self.germacrene_keywords,
            'diterpenes': ['kaurene', 'abietadiene', 'taxadiene'],
            'triterpenes': ['squalene', 'lupeol'],
            'other': ['linalool', 'nerol', 'geraniol']
        }
    
    def parse_marts_csv(self, csv_file: str) -> pd.DataFrame:
        """
        Parse MARTS-DB CSV file and preserve all annotation
        
        Args:
            csv_file: Path to MARTS-DB CSV file
            
        Returns:
            Enhanced DataFrame with comprehensive annotation
        """
        print(f"Parsing MARTS-DB CSV: {csv_file}")
        
        # Read CSV file
        df = pd.read_csv(csv_file)
        
        print(f"Loaded {len(df)} records")
        print(f"Columns: {list(df.columns)}")
        
        # Clean and validate data
        df = self._clean_data(df)
        
        # Add enhanced product annotation
        df = self._add_product_annotation(df)
        
        # Add classification labels
        df = self._add_classification_labels(df)
        
        # Add sequence statistics
        df = self._add_sequence_stats(df)
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the data"""
        print("Cleaning data...")
        
        # Remove rows with missing sequences
        initial_count = len(df)
        df = df.dropna(subset=['Aminoacid_sequence'])
        df = df[df['Aminoacid_sequence'].str.strip() != '']
        
        # Remove duplicates (same enzyme + sequence combination)
        df = df.drop_duplicates(subset=['Enzyme_marts_ID', 'Aminoacid_sequence'])
        
        print(f"After cleaning: {len(df)} sequences (removed {initial_count - len(df)})")
        return df
    
    def _add_product_annotation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive product annotation"""
        print("Adding product annotation...")
        
        # Create detailed product classification
        df['product_category'] = df['Product_name'].apply(self._classify_product_category)
        df['is_germacrene_family'] = df['Product_name'].apply(self._is_germacrene_family)
        df['germacrene_type'] = df['Product_name'].apply(self._get_germacrene_type)
        
        # Add enzyme classification
        df['enzyme_class'] = df['Class'].fillna('unknown')
        df['enzyme_type'] = df['Type'].fillna('unknown')
        
        # Add species information
        df['species'] = df['Species'].fillna('unknown')
        df['kingdom'] = df['Kingdom'].fillna('unknown')
        
        return df
    
    def _classify_product_category(self, product_name: str) -> str:
        """Classify product into categories"""
        if pd.isna(product_name):
            return 'unknown'
        
        product_lower = product_name.lower()
        
        for category, keywords in self.product_categories.items():
            for keyword in keywords:
                if keyword in product_lower:
                    return category
        
        return 'other'
    
    def _is_germacrene_family(self, product_name: str) -> bool:
        """Check if product is in germacrene family"""
        if pd.isna(product_name):
            return False
        
        product_lower = product_name.lower()
        return any(keyword in product_lower for keyword in self.germacrene_keywords)
    
    def _get_germacrene_type(self, product_name: str) -> str:
        """Get specific germacrene type"""
        if pd.isna(product_name):
            return 'none'
        
        product_lower = product_name.lower()
        
        for keyword in self.germacrene_keywords:
            if keyword in product_lower:
                return keyword
        
        return 'none'
    
    def _add_classification_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add binary classification labels for different tasks"""
        print("Adding classification labels...")
        
        # Binary germacrene classification (main task)
        df['is_germacrene'] = df['is_germacrene_family'].astype(int)
        
        # Multi-class classification (optional)
        df['product_class'] = df['product_category']
        
        # Enzyme type classification
        df['is_sesquiterpene_synthase'] = (df['enzyme_type'] == 'sesq').astype(int)
        df['is_monoterpene_synthase'] = (df['enzyme_type'] == 'mono').astype(int)
        
        return df
    
    def _add_sequence_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add sequence statistics"""
        print("Adding sequence statistics...")
        
        df['sequence_length'] = df['Aminoacid_sequence'].str.len()
        df['has_start_codon'] = df['Aminoacid_sequence'].str.startswith('M').astype(int)
        
        return df
    
    def create_training_dataset(self, df: pd.DataFrame, 
                               task: str = 'germacrene') -> Tuple[pd.DataFrame, Dict]:
        """
        Create training dataset for specific task
        
        Args:
            df: Parsed MARTS-DB DataFrame
            task: Training task ('germacrene', 'product_category', 'enzyme_type')
            
        Returns:
            Tuple of (training_data, statistics)
        """
        print(f"Creating training dataset for task: {task}")
        
        # Select relevant columns
        if task == 'germacrene':
            target_col = 'is_germacrene'
            task_name = 'Germacrene Synthase Classification'
        elif task == 'product_category':
            target_col = 'product_category'
            task_name = 'Product Category Classification'
        elif task == 'enzyme_type':
            target_col = 'enzyme_type'
            task_name = 'Enzyme Type Classification'
        else:
            raise ValueError(f"Unknown task: {task}")
        
        # Create training dataset
        training_data = df[['Enzyme_marts_ID', 'Aminoacid_sequence', target_col]].copy()
        training_data.columns = ['id', 'sequence', 'target']
        
        # Calculate statistics
        stats = {
            'task': task_name,
            'total_sequences': len(training_data),
            'positive_samples': (training_data['target'] == 1).sum() if task == 'germacrene' else training_data['target'].nunique(),
            'negative_samples': (training_data['target'] == 0).sum() if task == 'germacrene' else 0,
            'class_distribution': training_data['target'].value_counts().to_dict() if task != 'germacrene' else None,
            'sequence_length_stats': {
                'mean': training_data['sequence'].str.len().mean(),
                'std': training_data['sequence'].str.len().std(),
                'min': training_data['sequence'].str.len().min(),
                'max': training_data['sequence'].str.len().max()
            }
        }
        
        print(f"Training dataset created:")
        print(f"  - Total sequences: {stats['total_sequences']}")
        if task == 'germacrene':
            print(f"  - Germacrene synthases: {stats['positive_samples']}")
            print(f"  - Other synthases: {stats['negative_samples']}")
            print(f"  - Class balance: {stats['positive_samples']/stats['total_sequences']:.2%}")
        else:
            print(f"  - Unique classes: {stats['positive_samples']}")
            print(f"  - Class distribution: {stats['class_distribution']}")
        
        return training_data, stats
    
    def save_enhanced_data(self, df: pd.DataFrame, output_file: str):
        """Save enhanced dataset with all annotation"""
        print(f"Saving enhanced dataset to {output_file}")
        
        # Save full dataset
        df.to_csv(output_file, index=False)
        
        # Create summary report
        summary_file = output_file.replace('.csv', '_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("MARTS-DB Enhanced Dataset Summary\n")
            f.write("=" * 40 + "\n\n")
            
            f.write(f"Total sequences: {len(df)}\n")
            f.write(f"Germacrene synthases: {df['is_germacrene'].sum()}\n")
            f.write(f"Other synthases: {(~df['is_germacrene'].astype(bool)).sum()}\n\n")
            
            f.write("Product Categories:\n")
            for category, count in df['product_category'].value_counts().items():
                f.write(f"  {category}: {count}\n")
            
            f.write(f"\nGermacrene Types:\n")
            germacrene_types = df[df['is_germacrene_family']]['germacrene_type'].value_counts()
            for gtype, count in germacrene_types.items():
                f.write(f"  {gtype}: {count}\n")
            
            f.write(f"\nEnzyme Types:\n")
            for etype, count in df['enzyme_type'].value_counts().items():
                f.write(f"  {etype}: {count}\n")
        
        print(f"✓ Enhanced dataset and summary saved")
        print(f"  - Dataset: {output_file}")
        print(f"  - Summary: {summary_file}")


def main():
    """Main function to demonstrate the improved parser"""
    parser = ImprovedMARTSDBParser()
    
    # Parse the MARTS-DB data
    df = parser.parse_marts_csv("data/marts_db.csv")
    
    # Save enhanced dataset
    parser.save_enhanced_data(df, "data/marts_db_enhanced.csv")
    
    # Create training datasets for different tasks
    germacrene_data, germacrene_stats = parser.create_training_dataset(df, 'germacrene')
    germacrene_data.to_csv("data/germacrene_training_data.csv", index=False)
    
    print("\n" + "="*50)
    print("ENHANCED MARTS-DB PARSING COMPLETE")
    print("="*50)
    print(f"✓ Preserved all original annotation")
    print(f"✓ Added comprehensive product classification")
    print(f"✓ Created germacrene training dataset")
    print(f"✓ Generated detailed statistics")


if __name__ == "__main__":
    main()

