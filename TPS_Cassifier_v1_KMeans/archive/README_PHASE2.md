# Phase 2: Large-Scale Semi-Supervised Learning

## Overview

Phase 2 involves collecting and processing large-scale terpene synthase sequences from NCBI and UniProt for semi-supervised learning to improve our germacrene classifier.

## Data Collection Strategy

### Manual Download Instructions

**NCBI (https://www.ncbi.nlm.nih.gov/protein/):**
1. Search for: `"terpene synthase"[Title] OR "terpene synthase"[Abstract]`
2. Download results as FASTA format
3. Save as: `ncbi_terpene_synthase.fasta`
4. Repeat for additional terms:
   - `"terpenoid synthase"`
   - `"isoprenoid synthase"`
   - `"sesquiterpene synthase"`
   - `"monoterpene synthase"`
   - `"diterpene synthase"`
   - `"triterpene synthase"`

**UniProt (https://www.uniprot.org/):**
1. Search for: `name:terpene synthase OR description:terpene synthase`
2. Download as FASTA format
3. Save as: `uniprot_terpene_synthase.fasta`
4. Repeat for additional terms:
   - `terpenoid synthase`
   - `isoprenoid synthase`
   - `sesquiterpene synthase`

### File Organization

Place all downloaded FASTA files in: `data/phase2/downloaded/`

```
data/phase2/downloaded/
├── ncbi_terpene_synthase.fasta
├── ncbi_terpenoid_synthase.fasta
├── ncbi_sesquiterpene_synthase.fasta
├── uniprot_terpene_synthase.fasta
├── uniprot_terpenoid_synthase.fasta
└── ...
```

## Processing Pipeline

### 1. Process Downloaded Sequences

```bash
python3 process_downloaded_sequences.py --input-dir data/phase2/downloaded --output-dir data/phase2/processed
```

**What it does:**
- Loads all FASTA files from the input directory
- Applies quality filters (length, GC content, valid amino acids)
- Removes duplicate sequences using MD5 hashing
- Parses terpene product information from descriptions
- Filters out "putative" sequences
- Creates summary statistics
- Saves processed data in multiple formats

**Output files:**
- `processed_sequences.csv` - All sequences with metadata
- `processed_sequences.fasta` - Clean FASTA format
- `germacrene_sequences.fasta` - Germacrene-specific sequences
- `processing_summary.json` - Detailed statistics

### 2. Semi-Supervised Learning

```bash
python3 phase2_semi_supervised.py --unlabeled-data data/phase2/processed/processed_sequences.csv
```

**What it does:**
- Loads the Phase 1 trained model
- Generates embeddings for unlabeled sequences
- Applies pseudo-labeling with confidence thresholds
- Combines labeled and pseudo-labeled data
- Retrains the model on the expanded dataset
- Evaluates performance improvements

## Quality Filters

The processing pipeline applies several quality filters:

1. **Length Filter**: 200-1000 amino acids
2. **GC Content Filter**: 20-80% GC content
3. **Amino Acid Filter**: Only standard 20 amino acids
4. **Stop Codon Filter**: No 'X' characters
5. **Duplicate Filter**: MD5 hash-based deduplication
6. **Putative Filter**: Excludes sequences marked as "putative"

## Terpene Product Recognition

The processor recognizes 50+ terpene products including:

**Monoterpenes**: limonene, pinene, myrcene, linalool, camphor, etc.
**Sesquiterpenes**: germacrene, caryophyllene, humulene, farnesene, etc.
**Diterpenes**: taxadiene, kaurene, abietadiene, etc.
**Triterpenes**: squalene, lupeol, beta-amyrin, etc.

## Expected Results

After processing, you should see:
- Thousands of terpene synthase sequences
- Hundreds of germacrene-specific sequences
- Clean, deduplicated dataset ready for training
- Detailed statistics on sequence types and products

## Next Steps

1. Download FASTA files from NCBI and UniProt
2. Run the processing script
3. Review the summary statistics
4. Run the semi-supervised learning pipeline
5. Evaluate model improvements

## Troubleshooting

**No sequences found**: Check that FASTA files are in the correct directory and format
**Memory issues**: Process files in smaller batches
**API errors**: Use manual download instead of automated collection
**Quality issues**: Adjust filter parameters in the script
