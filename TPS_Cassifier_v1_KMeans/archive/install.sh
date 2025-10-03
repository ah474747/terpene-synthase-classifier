#!/bin/bash

# Germacrene Synthase Classifier Installation Script
# ================================================

echo "Germacrene Synthase Classifier Installation"
echo "=========================================="

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.8 or higher is required. Found Python $python_version"
    exit 1
fi

echo "✓ Python $python_version detected"

# Create virtual environment (optional)
read -p "Create a virtual environment? (y/n): " create_venv
if [ "$create_venv" = "y" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✓ Virtual environment created and activated"
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "✗ Failed to install dependencies"
    exit 1
fi

# Create directories
echo "Creating project directories..."
mkdir -p data models results

# Run setup script
echo "Running setup script..."
python3 setup.py

if [ $? -eq 0 ]; then
    echo "✓ Setup completed successfully"
else
    echo "⚠ Setup completed with warnings"
fi

# Run tests
echo "Running tests..."
python3 test_classifier.py

if [ $? -eq 0 ]; then
    echo "🎉 Installation completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Add your FASTA files to the data/ directory"
    echo "2. Run: python3 terpene_classifier.py"
    echo "3. Use: python3 predict.py --help for prediction options"
else
    echo "⚠ Some tests failed. Please check the output above."
fi

