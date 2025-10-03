#!/usr/bin/env python3
"""
Google Colab Setup Script for TS-GSD Pipeline

This script sets up the environment for running the Terpene Synthase
classification project in Google Colab.
"""

import subprocess
import sys
import os

def install_packages():
    """Install required packages for the TS-GSD pipeline"""
    print("🚀 Setting up TS-GSD Pipeline environment...")
    
    packages = [
        "pandas>=1.5.0",
        "numpy>=1.21.0", 
        "requests>=2.28.0",
        "tqdm>=4.64.0",
        "biopython>=1.79",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "jupyter>=1.0.0",
        "ipykernel>=6.0.0"
    ]
    
    print("📦 Installing core packages...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
    
    # Install PyTorch for GPU support in Colab
    print("🔥 Installing PyTorch with GPU support...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ])
        print("✅ PyTorch with CUDA support installed")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install PyTorch: {e}")
    
    # Install Transformers
    print("🤗 Installing Transformers...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers>=4.21.0"])
        print("✅ Transformers installed")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Transformers: {e}")

def create_directories():
    """Create necessary directories"""
    print("📁 Creating project directories...")
    
    directories = [
        "data",
        "models", 
        "results",
        "notebooks"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def test_installation():
    """Test that all packages are properly installed"""
    print("🧪 Testing installation...")
    
    try:
        import pandas as pd
        import numpy as np
        import requests
        import tqdm
        from Bio import SeqIO
        import torch
        import transformers
        print("✅ All packages imported successfully!")
        
        # Test GPU availability
        if torch.cuda.is_available():
            print(f"🚀 GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  GPU not available - using CPU")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    return True

def main():
    """Main setup function"""
    print("=" * 60)
    print("🧬 TS-GSD Pipeline - Google Colab Setup")
    print("=" * 60)
    
    # Install packages
    install_packages()
    
    # Create directories
    create_directories()
    
    # Test installation
    if test_installation():
        print("\n🎉 Setup completed successfully!")
        print("You can now run the TS-GSD pipeline:")
        print("python ts_gsd_pipeline.py")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()



