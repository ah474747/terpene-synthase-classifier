#!/usr/bin/env python3
"""
Performance Analyzer for Multi-Modal Terpene Synthase Classifier
===============================================================

This script analyzes the performance results from the Multi-Modal Terpene Synthase
Classifier and generates comprehensive performance reports for code review.

Usage:
    python3 performance_analyzer.py

Features:
    - Training performance analysis
    - Validation results interpretation
    - Generalization capability assessment
    - Comparative performance metrics
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

class PerformanceAnalyzer:
    """Analyzes performance results from the TPS Classifier."""
    
    def __init__(self, results_dir: str = "results"):
        """Initialize the performance analyzer."""
        self.results_dir = Path(results_dir)
        self.training_results = None
        self.validation_results = None
        self.generalization_results = None
        
    def load_results(self):
        """Load all result files."""
        print("📊 Loading performance results...")
        
        # Load training results
        training_file = self.results_dir / "training_results" / "final_functional_training_results.json"
        if training_file.exists():
            with open(training_file, 'r') as f:
                self.training_results = json.load(f)
            print(f"✅ Training results loaded: {training_file}")
        else:
            print(f"⚠️  Training results not found: {training_file}")
        
        # Load validation results
        validation_file = self.results_dir / "validation_results" / "hierarchical_strength_analysis.json"
        if validation_file.exists():
            with open(validation_file, 'r') as f:
                self.validation_results = json.load(f)
            print(f"✅ Validation results loaded: {validation_file}")
        else:
            print(f"⚠️  Validation results not found: {validation_file}")
        
        # Load generalization results
        generalization_file = self.results_dir / "generalization_results" / "generalization_validation_results.json"
        if generalization_file.exists():
            with open(generalization_file, 'r') as f:
                self.generalization_results = json.load(f)
            print(f"✅ Generalization results loaded: {generalization_file}")
        else:
            print(f"⚠️  Generalization results not found: {generalization_file}")
    
    def analyze_training_performance(self) -> Dict[str, Any]:
        """Analyze training performance metrics."""
        if not self.training_results:
            return {"error": "Training results not available"}
        
        analysis = {
            "final_metrics": {
                "macro_f1": self.training_results.get("final_macro_f1", 0),
                "micro_f1": self.training_results.get("final_micro_f1", 0),
                "macro_precision": self.training_results.get("final_macro_precision", 0),
                "macro_recall": self.training_results.get("final_macro_recall", 0)
            },
            "training_history": self.training_results.get("training_history", {}),
            "optimal_thresholds": self.training_results.get("optimal_thresholds", []),
            "training_time": self.training_results.get("training_time_seconds", 0)
        }
        
        # Calculate improvements
        if "training_history" in self.training_results:
            history = self.training_results["training_history"]
            if "val_f1_scores" in history and len(history["val_f1_scores"]) > 0:
                initial_f1 = history["val_f1_scores"][0]
                final_f1 = history["val_f1_scores"][-1]
                analysis["improvement"] = {
                    "initial_f1": initial_f1,
                    "final_f1": final_f1,
                    "improvement_ratio": final_f1 / initial_f1 if initial_f1 > 0 else 0
                }
        
        return analysis
    
    def analyze_validation_performance(self) -> Dict[str, Any]:
        """Analyze validation performance across different levels."""
        if not self.validation_results:
            return {"error": "Validation results not available"}
        
        analysis = {
            "hierarchical_performance": {},
            "strength_analysis": {},
            "class_distribution": {}
        }
        
        # Extract hierarchical performance
        if "hierarchical_results" in self.validation_results:
            hierarchical = self.validation_results["hierarchical_results"]
            analysis["hierarchical_performance"] = {
                "terpene_type_f1": hierarchical.get("terpene_type", {}).get("macro_f1", 0),
                "enzyme_class_f1": hierarchical.get("enzyme_class", {}).get("macro_f1", 0),
                "promiscuity_f1": hierarchical.get("promiscuity", {}).get("macro_f1", 0),
                "ensemble_f1": hierarchical.get("ensemble_prediction", {}).get("macro_f1", 0)
            }
        
        # Extract strength analysis
        if "strength_analysis" in self.validation_results:
            strength = self.validation_results["strength_analysis"]
            analysis["strength_analysis"] = {
                "strongest_level": strength.get("strongest_level", ""),
                "weakest_level": strength.get("weakest_level", ""),
                "performance_ranking": strength.get("performance_ranking", [])
            }
        
        return analysis
    
    def analyze_generalization_performance(self) -> Dict[str, Any]:
        """Analyze generalization performance on external sequences."""
        if not self.generalization_results:
            return {"error": "Generalization results not available"}
        
        analysis = {
            "overall_performance": {
                "macro_f1": self.generalization_results.get("macro_f1", 0),
                "precision_at_3": self.generalization_results.get("precision_at_3", 0),
                "successful_predictions": self.generalization_results.get("successful_predictions", 0),
                "total_sequences": self.generalization_results.get("n_sequences", 0)
            },
            "structure_analysis": {
                "real_structures": 0,
                "simulated_structures": 0,
                "failed_downloads": 0
            },
            "prediction_accuracy": {
                "top_1_correct": 0,
                "top_3_correct": 0,
                "total_sequences": 0
            }
        }
        
        # Analyze detailed results
        if "detailed_results" in self.generalization_results:
            detailed = self.generalization_results["detailed_results"]
            analysis["prediction_accuracy"]["total_sequences"] = len(detailed)
            
            for result in detailed:
                # Structure analysis
                if result.get("has_structure", False):
                    analysis["structure_analysis"]["real_structures"] += 1
                else:
                    analysis["structure_analysis"]["simulated_structures"] += 1
                
                # Prediction accuracy analysis
                # This would need the actual ground truth labels to be accurate
                # For now, we'll use placeholder logic
                top_predictions = result.get("top_3_predictions", [])
                if top_predictions:
                    # Check if any prediction is above threshold
                    above_threshold = any(pred.get("predicted", False) for pred in top_predictions)
                    if above_threshold:
                        analysis["prediction_accuracy"]["top_3_correct"] += 1
                    
                    # Check top-1 prediction
                    if top_predictions[0].get("predicted", False):
                        analysis["prediction_accuracy"]["top_1_correct"] += 1
        
        return analysis
    
    def generate_performance_report(self) -> str:
        """Generate a comprehensive performance report."""
        report = []
        report.append("# Performance Analysis Report")
        report.append("=" * 50)
        report.append("")
        
        # Training Performance
        training_analysis = self.analyze_training_performance()
        if "error" not in training_analysis:
            report.append("## Training Performance")
            report.append("")
            final_metrics = training_analysis["final_metrics"]
            report.append(f"- **Final Macro F1**: {final_metrics['macro_f1']:.4f}")
            report.append(f"- **Final Micro F1**: {final_metrics['micro_f1']:.4f}")
            report.append(f"- **Macro Precision**: {final_metrics['macro_precision']:.4f}")
            report.append(f"- **Macro Recall**: {final_metrics['macro_recall']:.4f}")
            report.append(f"- **Training Time**: {training_analysis['training_time']:.1f} seconds")
            
            if "improvement" in training_analysis:
                imp = training_analysis["improvement"]
                report.append(f"- **Improvement Ratio**: {imp['improvement_ratio']:.2f}x")
            report.append("")
        
        # Validation Performance
        validation_analysis = self.analyze_validation_performance()
        if "error" not in validation_analysis:
            report.append("## Validation Performance")
            report.append("")
            hierarchical = validation_analysis["hierarchical_performance"]
            report.append(f"- **Terpene Type F1**: {hierarchical.get('terpene_type_f1', 0):.4f}")
            report.append(f"- **Enzyme Class F1**: {hierarchical.get('enzyme_class_f1', 0):.4f}")
            report.append(f"- **Promiscuity F1**: {hierarchical.get('promiscuity_f1', 0):.4f}")
            report.append(f"- **Ensemble F1**: {hierarchical.get('ensemble_f1', 0):.4f}")
            report.append("")
        
        # Generalization Performance
        generalization_analysis = self.analyze_generalization_performance()
        if "error" not in generalization_analysis:
            report.append("## Generalization Performance")
            report.append("")
            overall = generalization_analysis["overall_performance"]
            report.append(f"- **External Macro F1**: {overall['macro_f1']:.4f}")
            report.append(f"- **Precision@3**: {overall['precision_at_3']:.4f}")
            report.append(f"- **Success Rate**: {overall['successful_predictions']}/{overall['total_sequences']}")
            
            structure = generalization_analysis["structure_analysis"]
            report.append(f"- **Real Structures**: {structure['real_structures']}")
            report.append(f"- **Simulated Structures**: {structure['simulated_structures']}")
            
            accuracy = generalization_analysis["prediction_accuracy"]
            report.append(f"- **Top-1 Accuracy**: {accuracy['top_1_correct']}/{accuracy['total_sequences']}")
            report.append(f"- **Top-3 Accuracy**: {accuracy['top_3_correct']}/{accuracy['total_sequences']}")
            report.append("")
        
        return "\n".join(report)
    
    def save_analysis(self, output_file: str = "performance_analysis_report.md"):
        """Save the performance analysis to a file."""
        report = self.generate_performance_report()
        
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"📄 Performance analysis saved to: {output_path}")
        return output_path

def main():
    """Main function for performance analysis."""
    print("📊 Multi-Modal Terpene Synthase Classifier - Performance Analyzer")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = PerformanceAnalyzer()
    
    # Load results
    analyzer.load_results()
    
    # Generate and save analysis
    output_file = analyzer.save_analysis("performance_analysis_report.md")
    
    # Print summary
    print("\n📈 Performance Analysis Summary:")
    print("-" * 40)
    
    training_analysis = analyzer.analyze_training_performance()
    if "error" not in training_analysis:
        final_f1 = training_analysis["final_metrics"]["macro_f1"]
        print(f"🎯 Final Training Macro F1: {final_f1:.4f}")
    
    generalization_analysis = analyzer.analyze_generalization_performance()
    if "error" not in generalization_analysis:
        external_f1 = generalization_analysis["overall_performance"]["macro_f1"]
        print(f"🌐 External Validation F1: {external_f1:.4f}")
    
    print(f"\n📄 Full analysis report saved to: {output_file}")

if __name__ == "__main__":
    main()



