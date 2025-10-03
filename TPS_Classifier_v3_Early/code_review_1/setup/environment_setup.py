#!/usr/bin/env python3
"""
Environment Setup Script for Multi-Modal Terpene Synthase Classifier
====================================================================

This script sets up the Python environment and installs all required dependencies
for the Multi-Modal Terpene Synthase Classifier code review.

Usage:
    python3 environment_setup.py

Requirements:
    - Python 3.8 or higher
    - pip package manager
    - Internet connection for downloading packages
"""

import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    else:
        print(f"✅ Python version: {sys.version.split()[0]}")
        return True

def install_requirements():
    """Install required packages from requirements.txt."""
    requirements_file = Path(__file__).parent.parent.parent / "code" / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ Error: requirements.txt not found")
        return False
    
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✅ All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def verify_installation():
    """Verify that key packages are installed correctly."""
    key_packages = [
        "torch",
        "torch_geometric", 
        "transformers",
        "biopython",
        "numpy",
        "pandas",
        "scikit-learn"
    ]
    
    print("🔍 Verifying package installation...")
    failed_packages = []
    
    for package in key_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n❌ Failed to import: {', '.join(failed_packages)}")
        return False
    else:
        print("\n✅ All key packages verified successfully")
        return True

def create_directories():
    """Create necessary directories for the project."""
    directories = [
        "alphafold_structures/pdb",
        "models_final_functional",
        "results",
        "logs"
    ]
    
    print("📁 Creating project directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ {directory}")

def main():
    """Main setup function."""
    print("🧬 Multi-Modal Terpene Synthase Classifier - Environment Setup")
    print("=" * 70)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("\n⚠️  Some packages failed to install. Please check the error messages above.")
        sys.exit(1)
    
    print("\n🎉 Environment setup completed successfully!")
    print("\nNext steps:")
    print("1. Review the documentation in the documentation/ directory")
    print("2. Examine the code in the code/ directory")
    print("3. Run the deployment pipeline: python3 code/deployment/TPS_Predictor.py")
    print("4. Analyze the results in the results/ directory")

if __name__ == "__main__":
    main()



