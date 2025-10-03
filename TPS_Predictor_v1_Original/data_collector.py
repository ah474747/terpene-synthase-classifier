"""
Data collection module for terpene synthase protein sequences and product annotations.
This module handles downloading and processing data from UniProt and other databases.
"""

import requests
import pandas as pd
import time
from typing import List, Dict, Tuple
from Bio import Entrez, SeqIO
from Bio.SeqRecord import SeqRecord
import xml.etree.ElementTree as ET
import json
import os
from tqdm import tqdm


class TerpeneSynthaseDataCollector:
    """
    Collects terpene synthase protein sequences and product annotations from various databases.
    """
    
    def __init__(self, email: str = "your_email@example.com"):
        """
        Initialize the data collector.
        
        Args:
            email: Email address for NCBI API access (required by NCBI)
        """
        self.email = email
        Entrez.email = email
        self.uniprot_base_url = "https://rest.uniprot.org"
        self.ncbi_base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        
    def search_uniprot_terpene_synthases(self, limit: int = 1000) -> List[Dict]:
        """
        Search UniProt for terpene synthase proteins.
        
        Args:
            limit: Maximum number of results to retrieve
            
        Returns:
            List of protein entries with metadata
        """
        print("Searching UniProt for terpene synthase proteins...")
        
        # Search query for terpene synthases
        query = "terpene synthase AND reviewed:true"
        
        # Search UniProt
        search_url = f"{self.uniprot_base_url}/uniprotkb/search"
        params = {
            "query": query,
            "format": "json",
            "size": limit,
            "fields": "accession,id,protein_name,organism_name,sequence,cc_function,cc_catalytic_activity,cc_cofactor,ft_region"
        }
        
        try:
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            proteins = []
            for entry in data.get("results", []):
                protein_data = {
                    "accession": entry.get("primaryAccession", ""),
                    "id": entry.get("uniProtkbId", ""),
                    "name": entry.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", ""),
                    "organism": entry.get("organism", {}).get("scientificName", ""),
                    "sequence": entry.get("sequence", {}).get("value", ""),
                    "function": self._extract_function(entry),
                    "catalytic_activity": self._extract_catalytic_activity(entry),
                    "cofactor": self._extract_cofactor(entry),
                    "domains": self._extract_domains(entry)
                }
                proteins.append(protein_data)
                
            print(f"Found {len(proteins)} terpene synthase proteins")
            return proteins
            
        except requests.RequestException as e:
            print(f"Error searching UniProt: {e}")
            return []
    
    def _extract_function(self, entry: Dict) -> str:
        """Extract function description from UniProt entry."""
        comments = entry.get("comments", [])
        for comment in comments:
            if comment.get("commentType") == "FUNCTION":
                return comment.get("texts", [{}])[0].get("value", "")
        return ""
    
    def _extract_catalytic_activity(self, entry: Dict) -> str:
        """Extract catalytic activity from UniProt entry."""
        comments = entry.get("comments", [])
        for comment in comments:
            if comment.get("commentType") == "CATALYTIC_ACTIVITY":
                return comment.get("reaction", {}).get("name", "")
        return ""
    
    def _extract_cofactor(self, entry: Dict) -> str:
        """Extract cofactor information from UniProt entry."""
        comments = entry.get("comments", [])
        for comment in comments:
            if comment.get("commentType") == "COFACTOR":
                return comment.get("texts", [{}])[0].get("value", "")
        return ""
    
    def _extract_domains(self, entry: Dict) -> List[str]:
        """Extract protein domains from UniProt entry."""
        features = entry.get("features", [])
        domains = []
        for feature in features:
            if feature.get("type") == "Domain":
                domains.append(feature.get("description", ""))
        return domains
    
    def search_ncbi_terpene_synthases(self, limit: int = 1000) -> List[Dict]:
        """
        Search NCBI for terpene synthase proteins.
        
        Args:
            limit: Maximum number of results to retrieve
            
        Returns:
            List of protein entries with metadata
        """
        print("Searching NCBI for terpene synthase proteins...")
        
        # Search for terpene synthase proteins
        search_term = "terpene synthase[Protein Name]"
        
        try:
            # Search for protein IDs
            search_url = f"{self.ncbi_base_url}/esearch.fcgi"
            search_params = {
                "db": "protein",
                "term": search_term,
                "retmax": limit,
                "retmode": "json"
            }
            
            response = requests.get(search_url, params=search_params)
            response.raise_for_status()
            search_data = response.json()
            
            protein_ids = search_data.get("esearchresult", {}).get("idlist", [])
            print(f"Found {len(protein_ids)} protein IDs")
            
            if not protein_ids:
                return []
            
            # Fetch detailed information for each protein
            proteins = []
            batch_size = 100
            
            for i in tqdm(range(0, len(protein_ids), batch_size)):
                batch_ids = protein_ids[i:i+batch_size]
                
                # Fetch protein details
                fetch_url = f"{self.ncbi_base_url}/efetch.fcgi"
                fetch_params = {
                    "db": "protein",
                    "id": ",".join(batch_ids),
                    "rettype": "gb",
                    "retmode": "xml"
                }
                
                response = requests.get(fetch_url, params=fetch_params)
                response.raise_for_status()
                
                # Parse XML response
                root = ET.fromstring(response.content)
                
                for protein in root.findall(".//GBSeq"):
                    protein_data = self._parse_ncbi_protein(protein)
                    if protein_data:
                        proteins.append(protein_data)
                
                # Rate limiting
                time.sleep(0.1)
            
            print(f"Retrieved {len(proteins)} terpene synthase proteins from NCBI")
            return proteins
            
        except requests.RequestException as e:
            print(f"Error searching NCBI: {e}")
            return []
    
    def _parse_ncbi_protein(self, protein_xml) -> Dict:
        """Parse NCBI protein XML entry."""
        try:
            accession = protein_xml.find("GBSeq_accession-version").text if protein_xml.find("GBSeq_accession-version") is not None else ""
            definition = protein_xml.find("GBSeq_definition").text if protein_xml.find("GBSeq_definition") is not None else ""
            organism = protein_xml.find("GBSeq_organism").text if protein_xml.find("GBSeq_organism") is not None else ""
            sequence = protein_xml.find("GBSeq_sequence").text if protein_xml.find("GBSeq_sequence") is not None else ""
            
            # Extract features
            features = []
            for feature in protein_xml.findall(".//GBFeature"):
                feature_type = feature.find("GBFeature_key").text if feature.find("GBFeature_key") is not None else ""
                qualifiers = {}
                for qualifier in feature.findall(".//GBQualifier"):
                    name = qualifier.find("GBQualifier_name").text if qualifier.find("GBQualifier_name") is not None else ""
                    value = qualifier.find("GBQualifier_value").text if qualifier.find("GBQualifier_value") is not None else ""
                    qualifiers[name] = value
                features.append({"type": feature_type, "qualifiers": qualifiers})
            
            return {
                "accession": accession,
                "definition": definition,
                "organism": organism,
                "sequence": sequence,
                "features": features,
                "source": "NCBI"
            }
            
        except Exception as e:
            print(f"Error parsing NCBI protein: {e}")
            return None
    
    def extract_product_annotations(self, proteins: List[Dict]) -> List[Dict]:
        """
        Extract product annotations from protein entries.
        
        Args:
            proteins: List of protein entries
            
        Returns:
            List of proteins with extracted product annotations
        """
        print("Extracting product annotations...")
        
        annotated_proteins = []
        
        for protein in tqdm(proteins):
            # Extract products from various fields
            products = []
            
            # From function description
            if protein.get("function"):
                products.extend(self._extract_products_from_text(protein["function"]))
            
            # From catalytic activity
            if protein.get("catalytic_activity"):
                products.extend(self._extract_products_from_text(protein["catalytic_activity"]))
            
            # From definition (NCBI)
            if protein.get("definition"):
                products.extend(self._extract_products_from_text(protein["definition"]))
            
            # From features (NCBI)
            if protein.get("features"):
                for feature in protein["features"]:
                    if feature.get("type") == "product":
                        product_name = feature.get("qualifiers", {}).get("product", "")
                        if product_name:
                            products.append(product_name)
            
            # Remove duplicates and clean up
            products = list(set(products))
            products = [p.strip() for p in products if p.strip()]
            
            protein["products"] = products
            annotated_proteins.append(protein)
        
        print(f"Extracted products for {len(annotated_proteins)} proteins")
        return annotated_proteins
    
    def _extract_products_from_text(self, text: str) -> List[str]:
        """Extract product names from text using simple pattern matching."""
        if not text:
            return []
        
        # Common terpene product patterns
        terpene_patterns = [
            r"(\w+ene)",  # Words ending in "ene"
            r"(\w+ol)",   # Words ending in "ol"
            r"(\w+al)",   # Words ending in "al"
            r"(\w+one)",  # Words ending in "one"
            r"(\w+ate)",  # Words ending in "ate"
        ]
        
        import re
        products = []
        
        for pattern in terpene_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            products.extend(matches)
        
        return products
    
    def save_data(self, proteins: List[Dict], filename: str = "terpene_synthase_data.json"):
        """
        Save collected data to JSON file.
        
        Args:
            proteins: List of protein entries
            filename: Output filename
        """
        print(f"Saving data to {filename}...")
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        filepath = os.path.join("data", filename)
        with open(filepath, 'w') as f:
            json.dump(proteins, f, indent=2)
        
        print(f"Saved {len(proteins)} proteins to {filepath}")
    
    def load_data(self, filename: str = "terpene_synthase_data.json") -> List[Dict]:
        """
        Load data from JSON file.
        
        Args:
            filename: Input filename
            
        Returns:
            List of protein entries
        """
        filepath = os.path.join("data", filename)
        
        if not os.path.exists(filepath):
            print(f"File {filepath} not found")
            return []
        
        with open(filepath, 'r') as f:
            proteins = json.load(f)
        
        print(f"Loaded {len(proteins)} proteins from {filepath}")
        return proteins


def main():
    """Main function to demonstrate data collection."""
    # Initialize collector
    collector = TerpeneSynthaseDataCollector(email="your_email@example.com")
    
    # Collect data from UniProt
    uniprot_proteins = collector.search_uniprot_terpene_synthases(limit=500)
    
    # Collect data from NCBI
    ncbi_proteins = collector.search_ncbi_terpene_synthases(limit=500)
    
    # Combine data
    all_proteins = uniprot_proteins + ncbi_proteins
    
    # Extract product annotations
    annotated_proteins = collector.extract_product_annotations(all_proteins)
    
    # Save data
    collector.save_data(annotated_proteins)
    
    # Print summary
    print(f"\nData Collection Summary:")
    print(f"Total proteins collected: {len(annotated_proteins)}")
    print(f"Proteins with products: {len([p for p in annotated_proteins if p.get('products')])}")
    
    # Show sample entries
    print(f"\nSample entries:")
    for i, protein in enumerate(annotated_proteins[:3]):
        print(f"\n{i+1}. {protein.get('name', 'Unknown')}")
        print(f"   Organism: {protein.get('organism', 'Unknown')}")
        print(f"   Products: {protein.get('products', [])}")
        print(f"   Sequence length: {len(protein.get('sequence', ''))}")


if __name__ == "__main__":
    main()
