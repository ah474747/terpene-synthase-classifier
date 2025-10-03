# Training Cell Fix - Use Simple ESM Function

## Original Training Cell (REPLACE THIS):

```python
# Load classes
with open("data/classes.txt") as f:
    classes = [line.strip() for line in f.readlines()]
print(f"Classes: {classes}")

# Create embedder
embedder = ESMEmbedder()

# Create datasets
train_dataset = TerpeneDataset("data/train.fasta", "data/train_labels.csv", embedder)
val_dataset = TerpeneDataset("data/val.fasta", "data/val_labels_binary.csv", embedder)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Embedding dimension: {train_dataset.embeddings.shape[1]}")
```

## NEW Training Cell (COPY THIS):

```python
import pandas as pd

# Load classes
with open("data/classes.txt") as f:
    classes = [line.strip() for line in f.readlines()]
print(f"Classes: {classes}")

# Load sequences and labels MANUALLY (no complex dataset class)
def load_sequences_and_labels(fasta_path, csv_path):
    # Load sequences
    sequences = []
    with open(fasta_path) as f:
        for line in f:
            if не line.startswith(">"):
                sequences.append(line.strip())
    
    # Load labels  
    labels = pd.read_csv(csv_path).values
    
    return sequences, labels

# Load training data
train_sequences, train_labels = load_sequences_and_labels("data/train.fasta", "data/train_labels.csv")
val_sequences, val_labels = load_sequences_and_labels("data/val.fasta", "data/val_labels_binary.csv")

print(f"Training sequences: {len(train_sequences)}")
print(f"Validation sequences: {len(val_sequences)}")

# Get embeddings using our simple function
print("Computing training embeddings...")
train_embeddings = simple_embed(train_sequences)
print(f"Training embeddings shape: {train_embeddings.shape}")

print("Computing validation embeddings...")
val_embeddings = simple_embed(val_sequences)
print(f"Validation embeddings shape: {val_embeddings.shape}")

# Create simple PyTorch datasets
import torch
from torch.utils.data import TensorDataset, DataLoader

train_tensor_x = torch.tensor(train_embeddings, dtype=torch.float32)
train_tensor_y = torch.tensor(train_labels, dtype=torch.float32)

val_tensor_x = torch.tensor(val_embeddings, dtype=torch.float32)  
val_tensor_y = torch.tensor(val_labels, dtype=torch.float32)

# Create datasets
train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
val_dataset = TensorDataset(val_tensor_x, val_tensor_y)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Embedding dimension: {train_tensor_x.shape[1]}")
```

## What Changed:

1. **Removed** the custom `TerpeneDataset` class
2. **Added** simple `load_sequences_and_labels()` function  
3. **Used** our `simple_embed()` function directly
4. **Created** standard PyTorch `TensorDataset` objects
5. **Much simpler** - no complex class inheritance

## Updated Model Setup Cell:

```python
from torch.utils.data import DataLoader
import torch.optim as optim

# Setup  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plm_dim = train_tensor_x.shape[1]  # Use the actual embedding dimension
n_classes = len(classes)

# Create model
model = TerpeneClassifier(plm_dim=plm_dim, n_classes=n_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

print(f"Model: {plm_dim} -> 6 classes")
print(f"Training batches: {len(train_loader)}")  
print(f"Validation batches: {len(val_loader)}")
```

This approach is much more straightforward and should work without any of the embedding issues we were having!
