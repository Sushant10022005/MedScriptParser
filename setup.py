"""
Setup script for OCR Medical Prescription Parser.
Run this script to verify installation and prepare the environment.
"""

import os
import sys
import subprocess
import importlib
from typing import List, Tuple


def check_python_version() -> bool:
    """Check if Python version is compatible."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    else:
        print("✅ Python version is compatible")
        return True


def check_gpu_availability() -> bool:
    """Check if GPU is available."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            print(f"✅ GPU available: {device_name}")
            print(f"   GPU memory: {memory:.1f} GB")
            print(f"   GPU count: {device_count}")
            return True
        else:
            print("⚠️  No GPU available - will use CPU (training will be slow)")
            return False
            
    except ImportError:
        print("❌ PyTorch not installed")
        return False


def check_required_packages() -> List[Tuple[str, bool]]:
    """Check if required packages are installed."""
    required_packages = [
        'torch',
        'transformers', 
        'datasets',
        'accelerate',
        'evaluate',
        'seqeval',
        'huggingface_hub',
        'sentencepiece',
        'PIL',
        'rapidfuzz',
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'tqdm'
    ]
    
    results = []
    
    print("Checking required packages:")
    for package in required_packages:
        try:
            if package == 'PIL':
                importlib.import_module('PIL')
            else:
                importlib.import_module(package)
            print(f"  ✅ {package}")
            results.append((package, True))
        except ImportError:
            print(f"  ❌ {package}")
            results.append((package, False))
    
    return results


def check_dataset() -> bool:
    """Check if dataset is available."""
    dataset_dir = "handwritten_output"
    labels_file = os.path.join(dataset_dir, "labels.txt")
    
    print("Checking dataset:")
    
    if not os.path.exists(dataset_dir):
        print(f"  ❌ Dataset directory not found: {dataset_dir}")
        return False
    
    if not os.path.exists(labels_file):
        print(f"  ❌ Labels file not found: {labels_file}")
        return False
    
    # Count images
    image_files = [f for f in os.listdir(dataset_dir) if f.endswith('.png')]
    num_images = len(image_files)
    
    # Count labels
    with open(labels_file, 'r', encoding='utf-8') as f:
        num_labels = sum(1 for line in f if line.strip())
    
    print(f"  ✅ Found {num_images} images")
    print(f"  ✅ Found {num_labels} labels")
    
    if num_images != num_labels:
        print(f"  ⚠️  Warning: Number of images ({num_images}) != number of labels ({num_labels})")
    
    return True


def install_missing_packages(missing_packages: List[str]) -> bool:
    """Install missing packages."""
    if not missing_packages:
        return True
    
    print(f"\nInstalling missing packages: {', '.join(missing_packages)}")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install"
        ] + missing_packages)
        print("✅ Packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False


def create_directories() -> None:
    """Create necessary directories."""
    directories = [
        "ocr_project",
        "results", 
        "plots"
    ]
    
    print("Creating directories:")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✅ {directory}/")


def download_base_models() -> bool:
    """Download base models to cache."""
    try:
        print("Pre-downloading base models (this may take a few minutes)...")
        
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        
        # Download TrOCR model
        print("  Downloading TrOCR model...")
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        print("  ✅ TrOCR model downloaded")
        
        # Try to download medical NER model (might not be available)
        try:
            print("  Downloading medical NER model...")
            from transformers import AutoTokenizer, AutoModelForTokenClassification
            tokenizer = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all")
            ner_model = AutoModelForTokenClassification.from_pretrained("d4data/biomedical-ner-all")
            print("  ✅ Medical NER model downloaded")
        except Exception:
            print("  ⚠️  Medical NER model not available (will use rule-based NER)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to download models: {e}")
        return False


def run_quick_test() -> bool:
    """Run a quick test of the pipeline."""
    try:
        print("Running quick test...")
        
        # Test dataset loading
        from dataset_loader import load_dataset, DatasetConfig
        config = DatasetConfig(max_samples=10)
        datasets = load_dataset(config)
        print(f"  ✅ Dataset loading works ({len(datasets['train'])} samples)")
        
        # Test preprocessing
        from preprocessing import preprocess_dataset
        train_dataset = preprocess_dataset(datasets['train'])
        print(f"  ✅ Preprocessing works")
        
        # Test inference (base model)
        from inference import TrOCRInferenceEngine
        engine = TrOCRInferenceEngine("microsoft/trocr-base-handwritten")
        print(f"  ✅ Inference engine works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Quick test failed: {e}")
        return False


def main():
    """Main setup function."""
    print("="*60)
    print("OCR Medical Prescription Parser - Setup")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Setup failed: Incompatible Python version")
        return False
    
    # Check GPU
    gpu_available = check_gpu_availability()
    
    # Check packages
    package_results = check_required_packages()
    missing_packages = [pkg for pkg, installed in package_results if not installed]
    
    # Install missing packages
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        
        install_choice = input("Install missing packages? (y/n): ").lower().strip()
        if install_choice == 'y':
            if not install_missing_packages(missing_packages):
                print("\n❌ Setup failed: Could not install packages")
                return False
        else:
            print("\n⚠️  Setup incomplete: Missing packages not installed")
    
    # Check dataset
    dataset_available = check_dataset()
    
    # Create directories
    create_directories()
    
    # Download models
    models_downloaded = download_base_models()
    
    # Run quick test
    test_passed = run_quick_test()
    
    # Summary
    print("\n" + "="*60)
    print("SETUP SUMMARY")
    print("="*60)
    print(f"Python version: {'✅' if check_python_version() else '❌'}")
    print(f"GPU available: {'✅' if gpu_available else '⚠️'}")
    print(f"Packages: {'✅' if not missing_packages else '⚠️'}")
    print(f"Dataset: {'✅' if dataset_available else '❌'}")
    print(f"Models: {'✅' if models_downloaded else '⚠️'}")
    print(f"Quick test: {'✅' if test_passed else '❌'}")
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    
    if not gpu_available:
        print("• Consider using a GPU for faster training")
        print("• Use smaller batch sizes for CPU training")
    
    if not dataset_available:
        print("• Ensure handwritten_output/ directory contains images and labels.txt")
    
    if missing_packages:
        print("• Install missing packages with: pip install -r requirements.txt")
    
    # Next steps
    print("\nNEXT STEPS:")
    print("• Run full pipeline: python main_pipeline.py")
    print("• Run test pipeline: python main_pipeline.py --test")
    print("• See README.md for detailed usage instructions")
    
    print("="*60)
    
    if test_passed:
        print("🎉 Setup completed successfully!")
        return True
    else:
        print("⚠️  Setup completed with warnings")
        return False


if __name__ == "__main__":
    main()