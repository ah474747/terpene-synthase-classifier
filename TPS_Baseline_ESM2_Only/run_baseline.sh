#!/usr/bin/env bash
# Quick-start script for ESM2-only baseline training

set -euo pipefail

echo "=========================================="
echo "  ESM2-Only Baseline Training"
echo "=========================================="
echo ""

# Check if fair-esm is installed
if ! python3 -c "import esm" 2>/dev/null; then
    echo "⚠️  fair-esm not found. Installing dependencies..."
    pip3 install -r requirements.txt
else
    echo "✅ Dependencies already installed"
fi

echo ""
echo "🚀 Starting baseline training..."
echo "   This will take ~30-60 minutes depending on your hardware"
echo ""

python3 train_baseline.py

echo ""
echo "=========================================="
echo "  ✅ Baseline training complete!"
echo "=========================================="
echo ""
echo "📊 Results saved to: results/baseline_cv_results.json"
echo "💾 Embeddings saved to: data/esm2_embeddings.npy"
echo ""

