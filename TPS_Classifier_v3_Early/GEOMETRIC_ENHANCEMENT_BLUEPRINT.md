
# Geometric Feature Maximization Blueprint

## 🎯 Objective
Create the definitive "functional" graph features by modeling the true geometric constraints of the active site with cofactors and substrates.

## 📋 Required Components

### Input Data
- **1,222 high-confidence PDB files** (from Module 4.5)
- **Cofactor templates**: Mg²⁺ ion coordinates (standard geometry)
- **Substrate templates**: Prenyl diphosphate structures (FPP, GPP, DMAPP)

### Computational Tools
1. **Ligand Docking Software**
   - AutoDock Vina (recommended)
   - OpenEye OMEGA (alternative)
   - Schrödinger Glide (commercial option)

2. **Structure Preparation**
   - PyMOL (visualization and preparation)
   - ChimeraX (structure analysis)
   - BioPython (automated processing)

## 🔧 Implementation Steps

### Step 1: Active Site Identification
```python
# Automated active site detection
def identify_active_site(pdb_file):
    # 1. Find conserved motifs (DDxxD, NSE/DTE)
    # 2. Locate metal-binding residues
    # 3. Define binding pocket (10-15 Å radius)
    # 4. Extract coordinates for docking
```

### Step 2: Cofactor Placement
```python
# Mg²⁺ ion placement
def place_mg2_ions(pdb_file, active_site_coords):
    # 1. Place 3 Mg²⁺ ions in standard geometry
    # 2. Optimize positions based on conserved residues
    # 3. Ensure proper coordination distances (2.0-2.5 Å)
    # 4. Generate Mg²⁺-bound PDB structure
```

### Step 3: Substrate Docking
```python
# Prenyl diphosphate docking
def dock_substrate(pdb_file, mg2_bound_structure):
    # 1. Prepare substrate structure (FPP/GPP/DMAPP)
    # 2. Define docking grid around active site
    # 3. Run AutoDock Vina with constraints
    # 4. Select best pose based on binding energy
    # 5. Generate final ligand-bound complex
```

### Step 4: GCN Recalculation
```python
# Enhanced graph creation with ligands
def create_ligand_aware_graphs(ligand_bound_pdbs):
    # 1. Parse ligand-bound PDB structures
    # 2. Extract protein + ligand coordinates
    # 3. Create enhanced node features (protein + ligand)
    # 4. Calculate ligand-protein contacts
    # 5. Generate functional graph representation
```

## 📊 Expected Outcomes

### Enhanced Node Features (30D)
- **Protein nodes**: 25D (20D one-hot + 5D physicochemical)
- **Ligand nodes**: 5D (substrate type, binding energy, coordination)
- **Contact edges**: Protein-ligand interactions

### Performance Improvements
- **Expected F1 increase**: 5-15% improvement
- **Better rare class performance**: Ligand constraints improve specificity
- **Enhanced generalization**: Functional constraints improve predictions

## 🚀 Implementation Timeline

### Phase 1: Preparation (1-2 days)
- Set up docking software
- Prepare cofactor/substrate templates
- Develop active site detection pipeline

### Phase 2: Docking (3-5 days)
- Automated docking for 1,222 structures
- Quality control and validation
- Generate ligand-bound PDB library

### Phase 3: Graph Enhancement (2-3 days)
- Recalculate GCN features with ligands
- Update model architecture for 30D nodes
- Retrain with enhanced features

### Phase 4: Validation (1 day)
- Performance comparison
- Final F1 score assessment
- Production deployment

## 🎯 Success Metrics
- **F1 Score Target**: 0.40-0.45 (40-45% macro F1)
- **Rare Class Improvement**: 10-20% better performance on sparse classes
- **Generalization**: Maintained performance on external sequences

## 💡 Key Advantages
1. **Functional Accuracy**: True active site geometry
2. **Substrate Specificity**: Different substrates for different TPS types
3. **Mechanistic Insights**: Ligand binding constraints
4. **Enhanced Performance**: Superior predictive power
