"""
Preprocessing module for TrOCR training.
Handles text normalization and image preprocessing.
"""

import re
from typing import Dict, Any, List, Union
from PIL import Image
from transformers import TrOCRProcessor
from datasets import Dataset, DatasetDict


class TextNormalizer:
    """Text normalization utilities for medical prescriptions."""
    
    def __init__(self):
        # Common medical abbreviations and their expansions
        self.medical_abbrevs = {
            'BID': 'twice daily',
            'TID': 'three times daily', 
            'QID': 'four times daily',
            'QDS': 'four times daily',
            'OD': 'once daily',
            'BD': 'twice daily',
            'PRN': 'as needed',
            'SOS': 'as needed',
            'mg': 'milligrams',
            'mL': 'milliliters',
            'ml': 'milliliters',
            'mg/ml': 'milligrams per milliliter',
            'q8h': 'every 8 hours',
            'q12h': 'every 12 hours',
            'q24h': 'every 24 hours',
            'q4h': 'every 4 hours',
            'q6h': 'every 6 hours',
            'ql2h': 'every 12 hours',
            'SC': 'subcutaneous',
            'IV': 'intravenous',
        }
    
    def light_normalize(self, text: str) -> str:
        """
        Apply light normalization suitable for OCR training.
        Preserves medical terminology while cleaning up formatting.
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Normalize common punctuation
        text = re.sub(r'\s*-\s*', ' - ', text)  # Standardize dash spacing
        text = re.sub(r'\s*:\s*', ': ', text)  # Standardize colon spacing
        text = re.sub(r'\s*,\s*', ', ', text)  # Standardize comma spacing
        
        # Fix common OCR-prone patterns
        text = re.sub(r'(\d+)\s*mg\s*', r'\1mg ', text)  # Normalize mg spacing
        text = re.sub(r'(\d+)\s*ml\s*', r'\1ml ', text)  # Normalize ml spacing
        text = re.sub(r'(\d+)\s*mL\s*', r'\1mL ', text)  # Normalize mL spacing
        
        # Normalize multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def expand_abbreviations(self, text: str) -> str:
        """Expand medical abbreviations for better understanding."""
        for abbrev, expansion in self.medical_abbrevs.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
        return text
    
    def normalize_for_training(self, text: str) -> str:
        """Full normalization pipeline for training data."""
        text = self.light_normalize(text)
        return text
    
    def normalize_for_inference(self, text: str) -> str:
        """Normalization for inference/comparison."""
        text = self.light_normalize(text)
        text = self.expand_abbreviations(text)
        return text


def preprocess_dataset(
    dataset_dict: Union[Dict[str, Dataset], Dataset],
    model_name: str = "microsoft/trocr-large-handwritten",
    max_target_length: int = 128
) -> Union[Dict[str, Dataset], Dataset]:
    """
    Lightweight preprocessing that keeps memory usage low:
    - No pixel tensors or label tensors are materialized.
    - Adds a 'normalized_text' column only.
    Images remain as lazy-loaded datasets.Image so they decode on demand.
    """
    text_normalizer = TextNormalizer()

    def add_normalized_text(batch):
        texts = batch["text"] if isinstance(batch["text"], list) else [batch["text"]]
        normalized = [text_normalizer.normalize_for_training(t) for t in texts]
        return {"normalized_text": normalized if len(normalized) > 1 else normalized[0]}

    if isinstance(dataset_dict, Dataset):
        print("Adding normalized_text to dataset (no tensor materialization)...")
        ds = dataset_dict.map(add_normalized_text, batched=True, desc="Normalizing text")
        return ds
    else:
        processed = {}
        for split_name, ds in dataset_dict.items():
            print(f"Adding normalized_text to {split_name} set...")
            processed[split_name] = ds.map(add_normalized_text, batched=True, desc=f"Normalizing {split_name}")
        print("Preprocessing complete (texts normalized).")
        return processed


def create_data_collator(processor: TrOCRProcessor):
    """Create a data collator for TrOCR training."""
    
    def collate_fn(batch):
        """Custom collate function for TrOCR training."""
        # Stack pixel values
        pixel_values = torch.stack([item['pixel_values'].squeeze() for item in batch])
        
        # Stack labels
        labels = torch.stack([item['labels'].squeeze() for item in batch])
        
        return {
            'pixel_values': pixel_values,
            'labels': labels
        }
    
    return collate_fn


def build_vocabulary(dataset_dict: Dict[str, Dataset]) -> List[str]:
    """
    Build vocabulary from training labels for OCR correction.
    
    Args:
        dataset_dict: Dataset dictionary
        
    Returns:
        List of unique words from training data
    """
    print("Building vocabulary from training data...")
    
    vocab = set()
    normalizer = TextNormalizer()
    
    # Extract words from training set
    for example in dataset_dict['train']:
        text = example['text']
        normalized_text = normalizer.normalize_for_training(text)
        
        # Split into words and add to vocabulary
        words = normalized_text.split()
        for word in words:
            # Clean word (remove punctuation for vocabulary)
            clean_word = re.sub(r'[^\w\s]', '', word).lower()
            if clean_word and len(clean_word) > 1:  # Only words with 2+ characters
                vocab.add(clean_word)
    
    vocab_list = sorted(list(vocab))
    print(f"Built vocabulary with {len(vocab_list)} unique words")
    
    return vocab_list


if __name__ == "__main__":
    # Test the preprocessing
    import torch
    from dataset_loader import load_dataset
    
    print("Testing preprocessing module...")
    
    # Test text normalization
    normalizer = TextNormalizer()
    
    test_texts = [
        "Alprazolam - 2 drops - every 6 hours",
        "Augmentin - 250mg - TDS",
        "Ciprofloxacin 200mg TID #30 tablets, 2 refills",
        "Losartan  50mg  SC  QID  x  until  symptoms  resolve"
    ]
    
    print("\nText Normalization Test:")
    print("-" * 50)
    for text in test_texts:
        normalized = normalizer.normalize_for_training(text)
        expanded = normalizer.normalize_for_inference(text)
        print(f"Original:  {text}")
        print(f"Normalized: {normalized}")
        print(f"Expanded:   {expanded}")
        print()
    
    # Test vocabulary building
    print("Loading small dataset for vocabulary test...")
    dataset_dict = load_dataset()
    
    # Build vocabulary from training data
    vocab = build_vocabulary(dataset_dict)
    
    print(f"\nVocabulary sample (first 20 words):")
    print(vocab[:20])
    
    print("\nPreprocessing test complete!")