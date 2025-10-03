#!/usr/bin/env python3
"""
Acceptance Criteria Checker
===========================

Checks if the stabilization meets all acceptance criteria.
"""

import argparse
import json
import logging
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from tps.utils import setup_logging

class AcceptanceCriteriaChecker:
    """Checks acceptance criteria for TPS classifier stabilization."""
    
    def __init__(self, results_dir: Path):
        """Initialize checker."""
        self.results_dir = Path(results_dir)
        self.criteria_results = {}
    
    def check_determinism(self) -> bool:
        """Check determinism criteria."""
        logger = logging.getLogger(__name__)
        
        try:
            # Check if determinism tests passed
            determinism_test_file = self.results_dir / "determinism_test_results.json"
            if determinism_test_file.exists():
                with open(determinism_test_file, 'r') as f:
                    results = json.load(f)
                
                determinism_passed = results.get('determinism_passed', False)
                self.criteria_results['determinism'] = determinism_passed
                
                if determinism_passed:
                    logger.info("✅ Determinism: Byte-identical outputs across runs")
                    return True
                else:
                    logger.error("❌ Determinism: Outputs not byte-identical")
                    return False
            else:
                logger.warning("⚠️  Determinism test results not found")
                self.criteria_results['determinism'] = False
                return False
                
        except Exception as e:
            logger.error(f"❌ Determinism check failed: {e}")
            self.criteria_results['determinism'] = False
            return False
    
    def check_artifact_discipline(self) -> bool:
        """Check artifact discipline criteria."""
        logger = logging.getLogger(__name__)
        
        try:
            # Check if artifact tests passed
            artifact_test_file = self.results_dir / "artifact_test_results.json"
            if artifact_test_file.exists():
                with open(artifact_test_file, 'r') as f:
                    results = json.load(f)
                
                artifact_passed = results.get('artifact_passed', False)
                self.criteria_results['artifact_discipline'] = artifact_passed
                
                if artifact_passed:
                    logger.info("✅ Artifact Discipline: Clear errors for missing artifacts")
                    return True
                else:
                    logger.error("❌ Artifact Discipline: Missing artifact handling failed")
                    return False
            else:
                logger.warning("⚠️  Artifact test results not found")
                self.criteria_results['artifact_discipline'] = False
                return False
                
        except Exception as e:
            logger.error(f"❌ Artifact discipline check failed: {e}")
            self.criteria_results['artifact_discipline'] = False
            return False
    
    def check_label_mapping_lock(self) -> bool:
        """Check label mapping lock criteria."""
        logger = logging.getLogger(__name__)
        
        try:
            # Check if label mapping tests passed
            label_test_file = self.results_dir / "label_mapping_test_results.json"
            if label_test_file.exists():
                with open(label_test_file, 'r') as f:
                    results = json.load(f)
                
                label_mapping_passed = results.get('label_mapping_passed', False)
                self.criteria_results['label_mapping_lock'] = label_mapping_passed
                
                if label_mapping_passed:
                    logger.info("✅ Label Mapping Lock: Dimensions and names consistent")
                    return True
                else:
                    logger.error("❌ Label Mapping Lock: Inconsistent dimensions or names")
                    return False
            else:
                logger.warning("⚠️  Label mapping test results not found")
                self.criteria_results['label_mapping_lock'] = False
                return False
                
        except Exception as e:
            logger.error(f"❌ Label mapping lock check failed: {e}")
            self.criteria_results['label_mapping_lock'] = False
            return False
    
    def check_no_leakage(self) -> bool:
        """Check no data leakage criteria."""
        logger = logging.getLogger(__name__)
        
        try:
            # Check if no leakage tests passed
            leakage_test_file = self.results_dir / "no_leakage_test_results.json"
            if leakage_test_file.exists():
                with open(leakage_test_file, 'r') as f:
                    results = json.load(f)
                
                no_leakage_passed = results.get('no_leakage_passed', False)
                self.criteria_results['no_leakage'] = no_leakage_passed
                
                if no_leakage_passed:
                    logger.info("✅ No Leakage: kNN index built only from training data")
                    return True
                else:
                    logger.error("❌ No Leakage: Data leakage detected")
                    return False
            else:
                logger.warning("⚠️  No leakage test results not found")
                self.criteria_results['no_leakage'] = False
                return False
                
        except Exception as e:
            logger.error(f"❌ No leakage check failed: {e}")
            self.criteria_results['no_leakage'] = False
            return False
    
    def check_esm_parity(self) -> bool:
        """Check ESM parity criteria."""
        logger = logging.getLogger(__name__)
        
        try:
            # Check if ESM parity tests passed
            esm_test_file = self.results_dir / "esm_parity_test_results.json"
            if esm_test_file.exists():
                with open(esm_test_file, 'r') as f:
                    results = json.load(f)
                
                esm_parity_passed = results.get('esm_parity_passed', False)
                self.criteria_results['esm_parity'] = esm_parity_passed
                
                if esm_parity_passed:
                    logger.info("✅ ESM Parity: Tokenizer/model ID and pooling match training")
                    return True
                else:
                    logger.error("❌ ESM Parity: Tokenizer/model ID or pooling mismatch")
                    return False
            else:
                logger.warning("⚠️  ESM parity test results not found")
                self.criteria_results['esm_parity'] = False
                return False
                
        except Exception as e:
            logger.error(f"❌ ESM parity check failed: {e}")
            self.criteria_results['esm_parity'] = False
            return False
    
    def check_identity_aware_performance(self) -> bool:
        """Check identity-aware performance criteria."""
        logger = logging.getLogger(__name__)
        
        try:
            # Load evaluation results
            eval_results_file = self.results_dir / "reports" / "val_compare.json"
            if not eval_results_file.exists():
                logger.warning("⚠️  Identity-aware evaluation results not found")
                self.criteria_results['identity_aware_performance'] = False
                return False
            
            with open(eval_results_file, 'r') as f:
                results = json.load(f)
            
            # Check for baseline and final results
            if 'val_base' not in results or 'val_final' not in results:
                logger.error("❌ Identity-aware Performance: Missing baseline or final results")
                self.criteria_results['identity_aware_performance'] = False
                return False
            
            baseline_metrics = results['val_base']['metrics']
            final_metrics = results['val_final']['metrics']
            
            # Check macro-F1 improvement
            baseline_macro_f1 = baseline_metrics.get('macro_f1', 0)
            final_macro_f1 = final_metrics.get('macro_f1', 0)
            
            improvement = final_macro_f1 - baseline_macro_f1
            
            # Check bootstrap CI
            baseline_ci = results['val_base']['bootstrap_ci']['macro_f1']
            final_ci = results['val_final']['bootstrap_ci']['macro_f1']
            
            # CI should not overlap 0 for improvement
            improvement_significant = (final_ci[1] - baseline_ci[2]) > 0
            
            criteria_met = (
                improvement >= 0.05 and  # At least 5 point improvement
                improvement_significant  # 95% CI excludes 0
            )
            
            self.criteria_results['identity_aware_performance'] = criteria_met
            
            if criteria_met:
                logger.info(f"✅ Identity-aware Performance: Macro-F1 improved by {improvement:.3f}")
                logger.info(f"   Baseline: {baseline_macro_f1:.3f}, Final: {final_macro_f1:.3f}")
                logger.info(f"   Improvement significant: {improvement_significant}")
                return True
            else:
                logger.error(f"❌ Identity-aware Performance: Insufficient improvement")
                logger.error(f"   Baseline: {baseline_macro_f1:.3f}, Final: {final_macro_f1:.3f}")
                logger.error(f"   Improvement: {improvement:.3f} (need ≥0.05)")
                logger.error(f"   Improvement significant: {improvement_significant}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Identity-aware performance check failed: {e}")
            self.criteria_results['identity_aware_performance'] = False
            return False
    
    def check_external_validation(self) -> bool:
        """Check external validation criteria."""
        logger = logging.getLogger(__name__)
        
        try:
            # Load external validation results
            ext_results_file = self.results_dir / "reports" / "ext30.json"
            if not ext_results_file.exists():
                logger.warning("⚠️  External validation results not found")
                self.criteria_results['external_validation'] = False
                return False
            
            with open(ext_results_file, 'r') as f:
                results = json.load(f)
            
            ext30_results = results.get('ext30_final', {})
            metrics = ext30_results.get('metrics', {})
            
            # Check macro-F1 improvement over baseline
            macro_f1 = metrics.get('macro_f1', 0)
            historical_baseline = 0.0569
            
            improvement = macro_f1 - historical_baseline
            
            # Check Top-3 accuracy
            top3_accuracy = metrics.get('top_3_accuracy', 0)
            
            # Check ECE reduction
            ece = metrics.get('ece', 1.0)  # Higher is worse
            ece_reduction = 1.0 - ece  # Reduction from perfect calibration
            
            criteria_met = (
                improvement > 0 and  # Material improvement over historical baseline
                top3_accuracy > 0.1 and  # Reasonable top-3 accuracy
                ece_reduction >= 0.3  # At least 30% ECE reduction
            )
            
            self.criteria_results['external_validation'] = criteria_met
            
            if criteria_met:
                logger.info(f"✅ External Validation: Material improvement over baseline")
                logger.info(f"   Macro-F1: {macro_f1:.4f} (baseline: {historical_baseline:.4f})")
                logger.info(f"   Top-3 Accuracy: {top3_accuracy:.3f}")
                logger.info(f"   ECE Reduction: {ece_reduction:.1%}")
                return True
            else:
                logger.error(f"❌ External Validation: Insufficient improvement")
                logger.error(f"   Macro-F1: {macro_f1:.4f} (baseline: {historical_baseline:.4f})")
                logger.error(f"   Top-3 Accuracy: {top3_accuracy:.3f}")
                logger.error(f"   ECE Reduction: {ece_reduction:.1%}")
                return False
                
        except Exception as e:
            logger.error(f"❌ External validation check failed: {e}")
            self.criteria_results['external_validation'] = False
            return False
    
    def check_all_criteria(self) -> dict:
        """Check all acceptance criteria."""
        logger = logging.getLogger(__name__)
        
        logger.info("🔍 Checking Acceptance Criteria")
        logger.info("=" * 50)
        
        criteria_checks = [
            ("Determinism", self.check_determinism),
            ("Artifact Discipline", self.check_artifact_discipline),
            ("Label Mapping Lock", self.check_label_mapping_lock),
            ("No Leakage", self.check_no_leakage),
            ("ESM Parity", self.check_esm_parity),
            ("Identity-aware Performance", self.check_identity_aware_performance),
            ("External Validation", self.check_external_validation),
        ]
        
        all_passed = True
        
        for criteria_name, check_func in criteria_checks:
            logger.info(f"\n📋 Checking {criteria_name}...")
            passed = check_func()
            if not passed:
                all_passed = False
        
        # Summary
        logger.info("\n" + "=" * 50)
        logger.info("ACCEPTANCE CRITERIA SUMMARY")
        logger.info("=" * 50)
        
        for criteria_name, passed in self.criteria_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"{criteria_name}: {status}")
        
        overall_status = "✅ ALL CRITERIA MET" if all_passed else "❌ CRITERIA NOT MET"
        logger.info(f"\nOVERALL RESULT: {overall_status}")
        
        # Save results
        results_file = self.results_dir / "acceptance_criteria_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'criteria_results': self.criteria_results,
                'all_criteria_met': all_passed,
                'timestamp': str(Path.cwd())
            }, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
        
        return {
            'all_criteria_met': all_passed,
            'criteria_results': self.criteria_results
        }

def main():
    parser = argparse.ArgumentParser(description="Check TPS classifier acceptance criteria")
    parser.add_argument("--results_dir", required=True, help="Directory containing test results")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Initialize checker
    checker = AcceptanceCriteriaChecker(results_dir=args.results_dir)
    
    # Check all criteria
    results = checker.check_all_criteria()
    
    # Exit with appropriate code
    if results['all_criteria_met']:
        logger.info("🎉 All acceptance criteria met!")
        sys.exit(0)
    else:
        logger.error("❌ Some acceptance criteria not met")
        sys.exit(1)

if __name__ == "__main__":
    main()



