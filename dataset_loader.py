"""
Dataset loader module for handwritten medical prescription OCR.
Handles loading, splitting, and creating Hugging Face Dataset objects.
"""

import os
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from PIL import Image
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""
    dataset_dir: str = "handwritten_output"
    labels_file: str = "handwritten_output/labels.txt"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    max_samples: Optional[int] = None
    seed: int = 42


def load_dataset(
    config: DatasetConfig = None,
    images_dir: str = "handwritten_output",
    labels_file: str = "handwritten_output/labels.txt",
    test_size: float = 0.2,
    val_size: float = 0.125,  # 0.125 of 80% = 10% of total
    seed: int = 42
) -> DatasetDict:
    """
    Load handwritten medical prescription dataset and create train/val/test splits.
    
    Args:
        config: DatasetConfig object (takes precedence if provided)
        images_dir: Directory containing the image files
        labels_file: Path to the labels text file
        test_size: Proportion for test set (default 0.2 = 20%)
        val_size: Proportion of remaining data for validation (0.125 of 80% = 10% total)
        seed: Random seed for reproducibility
        
    Returns:
        DatasetDict with train/validation/test splits
    """
    # Use config if provided, otherwise use individual parameters
    if config is not None:
        images_dir = config.dataset_dir
        labels_file = config.labels_file
        test_size = config.test_split
        val_size = config.val_split / (1 - config.test_split)  # Adjust val_size relative to remaining data
        seed = config.seed
        max_samples = config.max_samples
    else:
        max_samples = None
    
    # Set random seeds
    random.seed(seed)
    
    # Read labels file
    image_paths = []
    texts = []
    
    print(f"Loading labels from {labels_file}...")
    with open(labels_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Split filename and text (format: "filename.png text content")
            parts = line.split(' ', 1)
            if len(parts) != 2:
                continue
                
            filename, text = parts
            image_path = os.path.join(images_dir, filename)
            
            # Verify image file exists
            if os.path.exists(image_path):
                image_paths.append(image_path)
                texts.append(text.strip())
            else:
                print(f"Warning: Image file not found: {image_path}")
    
    print(f"Loaded {len(image_paths)} image-text pairs")
    
    # Apply max_samples limit if specified
    if max_samples is not None and max_samples < len(image_paths):
        print(f"Limiting dataset to {max_samples} samples (from {len(image_paths)})")
        # Shuffle and take first max_samples
        combined = list(zip(image_paths, texts))
        random.shuffle(combined)
        image_paths, texts = zip(*combined[:max_samples])
        image_paths, texts = list(image_paths), list(texts)
        print(f"Using {len(image_paths)} samples")
    
    # Create initial train/test split (80%/20%)
    train_images, test_images, train_texts, test_texts = train_test_split(
        image_paths, texts, test_size=test_size, random_state=seed, shuffle=True
    )
    
    # Create train/validation split from train set (90%/10% of train = 72%/8% of total)
    train_images, val_images, train_texts, val_texts = train_test_split(
        train_images, train_texts, test_size=val_size, random_state=seed, shuffle=True
    )
    
    print(f"Split sizes:")
    print(f"  Train: {len(train_images)} ({len(train_images)/len(image_paths)*100:.1f}%)")
    print(f"  Validation: {len(val_images)} ({len(val_images)/len(image_paths)*100:.1f}%)")
    print(f"  Test: {len(test_images)} ({len(test_images)/len(image_paths)*100:.1f}%)")
    
    # Create datasets
    def create_dataset(image_paths: List[str], texts: List[str]) -> Dataset:
        """Create a Hugging Face Dataset from image paths and texts."""
        def load_image(path: str) -> Image.Image:
            return Image.open(path).convert('RGB')
        
        # Load all images
        images = [load_image(path) for path in image_paths]
        
        return Dataset.from_dict({
            'image': images,
            'text': texts,
            'image_path': image_paths
        })
    
    print("Creating Hugging Face datasets...")
    train_dataset = create_dataset(train_images, train_texts)
    val_dataset = create_dataset(val_images, val_texts)
    test_dataset = create_dataset(test_images, test_texts)
    
    # Create DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    
    print("Dataset loading complete!")
    return dataset_dict


def get_dataset_stats(dataset_dict: DatasetDict) -> Dict:
    """Get statistics about the dataset."""
    stats = {}
    
    for split_name, dataset in dataset_dict.items():
        texts = dataset['text']
        text_lengths = [len(text) for text in texts]
        word_counts = [len(text.split()) for text in texts]
        
        stats[split_name] = {
            'num_samples': len(dataset),
            'avg_text_length': sum(text_lengths) / len(text_lengths),
            'max_text_length': max(text_lengths),
            'min_text_length': min(text_lengths),
            'avg_word_count': sum(word_counts) / len(word_counts),
            'max_word_count': max(word_counts),
            'min_word_count': min(word_counts)
        }
    
    return stats


def print_dataset_info(dataset_dict: DatasetDict):
    """Print detailed information about the dataset."""
    print("\n" + "="*50)
    print("DATASET INFORMATION")
    print("="*50)
    
    stats = get_dataset_stats(dataset_dict)
    
    for split_name, split_stats in stats.items():
        print(f"\n{split_name.upper()} SET:")
        print(f"  Samples: {split_stats['num_samples']:,}")
        print(f"  Avg text length: {split_stats['avg_text_length']:.1f} chars")
        print(f"  Text length range: {split_stats['min_text_length']}-{split_stats['max_text_length']} chars")
        print(f"  Avg word count: {split_stats['avg_word_count']:.1f} words")
        print(f"  Word count range: {split_stats['min_word_count']}-{split_stats['max_word_count']} words")
    
    # Show sample data
    print(f"\nSAMPLE DATA (from train set):")
    print("-" * 50)
    for i in range(min(3, len(dataset_dict['train']))):
        sample = dataset_dict['train'][i]
        print(f"Image: {sample['image_path']}")
        print(f"Text: {sample['text']}")
        print(f"Image size: {sample['image'].size}")
        print("-" * 50)


if __name__ == "__main__":
    # Test the dataset loader
    dataset_dict = load_dataset()
    print_dataset_info(dataset_dict)