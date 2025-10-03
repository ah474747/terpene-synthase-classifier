"""
Molecular Fingerprint Encoder for Terpene Products

This module handles encoding of terpene products into molecular fingerprints
using RDKit for chemical structure representation.

Based on research best practices for molecular fingerprint integration.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import logging
from pathlib import Path
import pickle
from dataclasses import dataclass

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning("RDKit not available. Install with: pip install rdkit")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MolecularFingerprint:
    """Container for molecular fingerprint data"""
    smiles: str
    morgan_fingerprint: np.ndarray
    maccs_keys: np.ndarray
    rdkit_descriptors: np.ndarray
    molecular_weight: float
    logp: float
    tpsa: float
    num_rotatable_bonds: int
    num_aromatic_rings: int

class TerpeneProductEncoder:
    """Encodes terpene products into molecular fingerprints"""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for molecular fingerprinting")
        
        # Terpene product SMILES database
        self.terpene_smiles = {
            "limonene": "CC1=CCC(CC1)C(=C)C",
            "pinene": "CC1=CCC2CC1C2(C)C",
            "myrcene": "CC(=CCCC(=C)C)C",
            "linalool": "CC(C)=CCCC(C)(C=C)O",
            "germacrene_a": "CC1=CCCC(=C)C2CC1C2(C)C",
            "germacrene_d": "CC1=CCCC(=C)C2CC1C2(C)C",
            "caryophyllene": "CC1=CCCC(=C)C2CC1C2(C)C",
            "humulene": "CC1=CCCC(=C)C2CC1C2(C)C",
            "farnesene": "CC(=CCCC(=CCCC(=C)C)C)C",
            "bisabolene": "CC1=CCCC(=C)C2CC1C2(C)C",
        }
        
        # Fingerprint parameters
        self.morgan_radius = 2
        self.morgan_nbits = 2048
        self.maccs_nbits = 167
        
        # RDKit descriptor names
        self.descriptor_names = [
            'MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 'TPSA',
            'NumRotatableBonds', 'NumAromaticRings', 'NumSaturatedRings',
            'NumAliphaticRings', 'NumHeteroatoms', 'FractionCsp3',
            'HeavyAtomCount', 'NumRadicalElectrons', 'NumValenceElectrons'
        ]

    def encode_product(self, product_name: str, smiles: Optional[str] = None) -> Optional[MolecularFingerprint]:
        """Encode a terpene product into molecular fingerprint"""
        try:
            # Get SMILES
            if smiles is None:
                smiles = self.terpene_smiles.get(product_name.lower())
                if smiles is None:
                    logger.error(f"No SMILES found for product: {product_name}")
                    return None
            
            # Create molecule from SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.error(f"Invalid SMILES: {smiles}")
                return None
            
            # Generate fingerprints
            morgan_fp = self._generate_morgan_fingerprint(mol)
            maccs_fp = self._generate_maccs_fingerprint(mol)
            rdkit_desc = self._generate_rdkit_descriptors(mol)
            
            # Extract key properties
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            num_rot_bonds = Descriptors.NumRotatableBonds(mol)
            num_arom_rings = Descriptors.NumAromaticRings(mol)
            
            return MolecularFingerprint(
                smiles=smiles,
                morgan_fingerprint=morgan_fp,
                maccs_keys=maccs_fp,
                rdkit_descriptors=rdkit_desc,
                molecular_weight=mw,
                logp=logp,
                tpsa=tpsa,
                num_rotatable_bonds=num_rot_bonds,
                num_aromatic_rings=num_arom_rings
            )
            
        except Exception as e:
            logger.error(f"Error encoding product {product_name}: {e}")
            return None

    def _generate_morgan_fingerprint(self, mol) -> np.ndarray:
        """Generate Morgan fingerprint"""
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, self.morgan_radius, nBits=self.morgan_nbits
        )
        return np.array(fp)

    def _generate_maccs_fingerprint(self, mol) -> np.ndarray:
        """Generate MACCS keys fingerprint"""
        fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
        return np.array(fp)

    def _generate_rdkit_descriptors(self, mol) -> np.ndarray:
        """Generate RDKit molecular descriptors"""
        descriptors = []
        for desc_name in self.descriptor_names:
            try:
                desc_value = getattr(Descriptors, desc_name)(mol)
                descriptors.append(desc_value)
            except:
                descriptors.append(0.0)
        return np.array(descriptors)

    def encode_dataset(self, products: List[str]) -> Dict[str, MolecularFingerprint]:
        """Encode a dataset of terpene products"""
        logger.info(f"Encoding {len(products)} terpene products...")
        
        encoded_products = {}
        
        for product in products:
            fingerprint = self.encode_product(product)
            if fingerprint:
                encoded_products[product] = fingerprint
            else:
                logger.warning(f"Failed to encode product: {product}")
        
        logger.info(f"Successfully encoded {len(encoded_products)} products")
        return encoded_products

    def create_fingerprint_matrix(self, encoded_products: Dict[str, MolecularFingerprint]) -> Tuple[np.ndarray, List[str]]:
        """Create fingerprint matrix for machine learning"""
        products = list(encoded_products.keys())
        
        # Combine all fingerprint types
        fingerprint_vectors = []
        
        for product in products:
            fp = encoded_products[product]
            # Concatenate all fingerprint types
            combined_fp = np.concatenate([
                fp.morgan_fingerprint,
                fp.maccs_keys,
                fp.rdkit_descriptors
            ])
            fingerprint_vectors.append(combined_fp)
        
        fingerprint_matrix = np.array(fingerprint_vectors)
        
        logger.info(f"Created fingerprint matrix: {fingerprint_matrix.shape}")
        return fingerprint_matrix, products

    def save_encoded_products(self, encoded_products: Dict[str, MolecularFingerprint], filename: str = "terpene_fingerprints.pkl"):
        """Save encoded products to file"""
        output_path = self.cache_dir / filename
        
        # Convert to serializable format
        serializable_data = {}
        for product, fp in encoded_products.items():
            serializable_data[product] = {
                'smiles': fp.smiles,
                'morgan_fingerprint': fp.morgan_fingerprint,
                'maccs_keys': fp.maccs_keys,
                'rdkit_descriptors': fp.rdkit_descriptors,
                'molecular_weight': fp.molecular_weight,
                'logp': fp.logp,
                'tpsa': fp.tpsa,
                'num_rotatable_bonds': fp.num_rotatable_bonds,
                'num_aromatic_rings': fp.num_aromatic_rings
            }
        
        with open(output_path, 'wb') as f:
            pickle.dump(serializable_data, f)
        
        logger.info(f"Saved encoded products to {output_path}")

    def load_encoded_products(self, filename: str = "terpene_fingerprints.pkl") -> Dict[str, MolecularFingerprint]:
        """Load encoded products from file"""
        input_path = self.cache_dir / filename
        
        if not input_path.exists():
            logger.error(f"File not found: {input_path}")
            return {}
        
        with open(input_path, 'rb') as f:
            serializable_data = pickle.load(f)
        
        # Convert back to MolecularFingerprint objects
        encoded_products = {}
        for product, data in serializable_data.items():
            encoded_products[product] = MolecularFingerprint(
                smiles=data['smiles'],
                morgan_fingerprint=data['morgan_fingerprint'],
                maccs_keys=data['maccs_keys'],
                rdkit_descriptors=data['rdkit_descriptors'],
                molecular_weight=data['molecular_weight'],
                logp=data['logp'],
                tpsa=data['tpsa'],
                num_rotatable_bonds=data['num_rotatable_bonds'],
                num_aromatic_rings=data['num_aromatic_rings']
            )
        
        logger.info(f"Loaded {len(encoded_products)} encoded products")
        return encoded_products

class FingerprintAnalyzer:
    """Analyzes molecular fingerprints for insights"""
    
    def __init__(self):
        pass
    
    def analyze_similarity(self, encoded_products: Dict[str, MolecularFingerprint]) -> pd.DataFrame:
        """Analyze similarity between terpene products"""
        products = list(encoded_products.keys())
        similarity_matrix = np.zeros((len(products), len(products)))
        
        for i, prod1 in enumerate(products):
            for j, prod2 in enumerate(products):
                fp1 = encoded_products[prod1].morgan_fingerprint
                fp2 = encoded_products[prod2].morgan_fingerprint
                
                # Calculate Tanimoto similarity
                similarity = self._tanimoto_similarity(fp1, fp2)
                similarity_matrix[i, j] = similarity
        
        # Create DataFrame
        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=products,
            columns=products
        )
        
        return similarity_df
    
    def _tanimoto_similarity(self, fp1: np.ndarray, fp2: np.ndarray) -> float:
        """Calculate Tanimoto similarity between fingerprints"""
        intersection = np.sum(fp1 & fp2)
        union = np.sum(fp1 | fp2)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def analyze_descriptors(self, encoded_products: Dict[str, MolecularFingerprint]) -> pd.DataFrame:
        """Analyze molecular descriptors"""
        data = []
        
        for product, fp in encoded_products.items():
            data.append({
                'product': product,
                'molecular_weight': fp.molecular_weight,
                'logp': fp.logp,
                'tpsa': fp.tpsa,
                'num_rotatable_bonds': fp.num_rotatable_bonds,
                'num_aromatic_rings': fp.num_aromatic_rings
            })
        
        return pd.DataFrame(data)

def main():
    """Main function to demonstrate molecular fingerprint encoding"""
    logger.info("Starting molecular fingerprint encoding...")
    
    # Initialize encoder
    encoder = TerpeneProductEncoder()
    
    # Define terpene products
    products = [
        "limonene", "pinene", "myrcene", "linalool",
        "germacrene_a", "germacrene_d", "caryophyllene",
        "humulene", "farnesene", "bisabolene"
    ]
    
    # Encode products
    encoded_products = encoder.encode_dataset(products)
    
    # Create fingerprint matrix
    fingerprint_matrix, product_names = encoder.create_fingerprint_matrix(encoded_products)
    
    # Save results
    encoder.save_encoded_products(encoded_products)
    
    # Analyze results
    analyzer = FingerprintAnalyzer()
    similarity_df = analyzer.analyze_similarity(encoded_products)
    descriptor_df = analyzer.analyze_descriptors(encoded_products)
    
    # Print summary
    print(f"\nMolecular Fingerprint Summary:")
    print(f"Products encoded: {len(encoded_products)}")
    print(f"Fingerprint matrix shape: {fingerprint_matrix.shape}")
    print(f"Morgan fingerprint bits: {encoder.morgan_nbits}")
    print(f"MACCS keys bits: {encoder.maccs_nbits}")
    print(f"RDKit descriptors: {len(encoder.descriptor_names)}")
    
    print(f"\nMolecular Descriptors:")
    print(descriptor_df.round(2))
    
    print(f"\nSimilarity Matrix (top 5x5):")
    print(similarity_df.iloc[:5, :5].round(3))

if __name__ == "__main__":
    main()
