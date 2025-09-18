# OCR Medical Prescription Parser

A comprehensive end-to-end pipeline for parsing handwritten medical prescriptions using OCR and NER-based correction.

## Features

- **TrOCR Fine-tuning**: Fine-tunes microsoft/trocr-large-handwritten on medical prescription data
- **MediPhi Correction**: Uses microsoft/MediPhi for NER-based error correction
- **Comprehensive Evaluation**: CER, WER, Exact Match Rate, and NER metrics
- **Fuzzy String Matching**: Corrects OCR errors using medical vocabulary
- **Rich Visualizations**: Training curves, error analysis, and result dashboards
- **Modular Design**: Each component can be used independently
- **Research-Ready Outputs**: Generates papers-ready results and plots

## Project Structure

```
.
├── main_pipeline.py          # Main orchestration pipeline
├── dataset_loader.py         # Dataset loading and splitting
├── preprocessing.py          # Text normalization and image preprocessing  
├── training.py              # TrOCR model training
├── inference.py             # Model inference engine
├── mediphi_correction.py    # MediPhi-based correction module
├── evaluation.py            # Comprehensive evaluation metrics
├── visualization.py         # Plotting and visualization
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── handwritten_output/     # Dataset directory (10k images + labels.txt)
├── ocr_project/           # Training outputs (checkpoints, models)
├── results/               # Evaluation results (CSV, JSON)
└── plots/                 # Generated visualizations
```

## Quick Start

### 1. Setup Environment

```bash
# Clone or download the project
# Navigate to project directory

# Create virtual environment (recommended)
python -m venv ocr_env
source ocr_env/bin/activate  # On Windows: ocr_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

Ensure your dataset structure looks like:
```
handwritten_output/
├── 0.png
├── 1.png
├── ...
├── 9999.png
└── labels.txt
```

Where `labels.txt` contains:
```
0.png	Take aspirin 100mg twice daily
1.png	Paracetamol 500mg three times a day
...
```

### 3. Run Complete Pipeline

```bash
# Full pipeline (training + evaluation)
python main_pipeline.py

# Test with smaller dataset (100 samples, 1 epoch)
python main_pipeline.py --test

# Skip training (use base model)
python main_pipeline.py --skip-training

# Custom configuration
python main_pipeline.py --epochs 5 --batch-size 4 --max-samples 1000
```

### 4. Cloud/GPU Setup

For cloud deployment with GPU:

```bash
# Google Colab / Kaggle / AWS EC2
!git clone <your-repo-url>
%cd <project-directory>
!pip install -r requirements.txt

# Check GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Run pipeline
!python main_pipeline.py --epochs 10 --batch-size 8
```

## Usage Examples

### Individual Components

```python
# Load dataset
from dataset_loader import load_dataset
datasets = load_dataset()

# Train model
from training import train_model, TrOCRTrainingConfig
config = TrOCRTrainingConfig(num_train_epochs=10)
trainer = train_model(train_dataset, val_dataset, config)

# Run inference
from inference import TrOCRInferenceEngine
engine = TrOCRInferenceEngine("./ocr_project")
predictions = engine.predict_batch(images)

# Apply corrections
from mediphi_correction import correct_with_mediphi
corrected = correct_with_mediphi(predictions, training_texts)

# Evaluate
from evaluation import ComprehensiveEvaluator
evaluator = ComprehensiveEvaluator()
results = evaluator.evaluate_pipeline(images, ground_truth, ocr_pred, corrected_pred)

# Visualize
from visualization import create_visualizations
create_visualizations(training_history, results, "results.csv")
```

### Custom Configuration

```python
from main_pipeline import OCRPipelineConfig, main

config = OCRPipelineConfig(
    dataset_dir="./my_dataset",
    num_epochs=15,
    batch_size=16,
    learning_rate=3e-5,
    output_dir="./my_model",
    results_dir="./my_results"
)

results = main(config)
```

## Configuration Options

### Dataset Configuration
- `dataset_dir`: Path to image directory
- `labels_file`: Path to labels file
- `train_split`: Training split ratio (default: 0.8)
- `val_split`: Validation split ratio (default: 0.1)  
- `test_split`: Test split ratio (default: 0.1)
- `max_samples`: Limit dataset size for testing

### Training Configuration
- `model_name`: Base model (default: "microsoft/trocr-large-handwritten")
- `num_epochs`: Training epochs (default: 10)
- `batch_size`: Training batch size (default: 8)
- `learning_rate`: Learning rate (default: 5e-5)
- `output_dir`: Model checkpoint directory

### Correction Configuration
- `ner_model`: NER model for correction (default: "microsoft/MediPhi")
- `vocab_file`: Medical vocabulary file path
- `skip_correction`: Skip correction step

## Output Files

The pipeline generates:

### Model Outputs
- `ocr_project/`: Trained TrOCR model checkpoints
- `ocr_project/final_model/`: Final model + tokenizer

### Results
- `results/ocr_predictions.csv`: Raw OCR predictions
- `results/detailed_results.csv`: Sample-by-sample results
- `results/metrics_summary.json`: Aggregated metrics
- `results/correction_results.csv`: Correction details
- `results/experiment_summary.json`: Complete experiment info
- `results/results_report.txt`: Human-readable summary

### Visualizations
- `plots/training_curves.png`: Loss and metric curves
- `plots/metrics_comparison.png`: OCR vs corrected performance
- `plots/error_distributions.png`: CER/WER histograms
- `plots/improvement_analysis.png`: Before/after scatter plots
- `plots/results_dashboard.png`: Comprehensive summary dashboard

## Performance Metrics

The system evaluates:

### OCR Metrics
- **CER (Character Error Rate)**: Character-level errors
- **WER (Word Error Rate)**: Word-level errors
- **Exact Match Rate**: Percentage of perfectly matched samples

### NER Metrics  
- **Precision/Recall/F1**: Per entity type (DRUG, DOSAGE, FREQUENCY, FORM)
- **Overall NER Performance**: Aggregated metrics

### Improvement Metrics
- **Error Reduction**: Absolute and percentage improvements
- **Correction Statistics**: Number and types of corrections made

## Advanced Features

### Custom Medical Vocabulary
```python
from mediphi_correction import MedicalVocabularyBuilder

vocab_builder = MedicalVocabularyBuilder()
vocab_builder.build_from_texts(training_texts)
vocab_builder.save_vocabulary("custom_vocab.json")
```

### Batch Processing
```python
from inference import TrOCRInferenceEngine

engine = TrOCRInferenceEngine("model_path", batch_size=32)
results = engine.predict_batch(image_paths, return_confidence=True)
```

### Custom Evaluation
```python
from evaluation import OCREvaluator, NERChunkEvaluator

ocr_eval = OCREvaluator()
metrics = ocr_eval.evaluate(predictions, references, normalize=True)
```

## Requirements

### Minimum Requirements
- Python 3.8+
- 8GB RAM
- 10GB disk space

### Recommended for Training
- Python 3.9+
- 16GB+ RAM
- GPU with 8GB+ VRAM (RTX 3070/V100/T4)
- 50GB+ disk space

### Cloud Platforms
- Google Colab (Pro recommended)
- Kaggle Kernels
- AWS EC2 (p3.2xlarge or similar)
- Azure ML
- GCP AI Platform

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   python main_pipeline.py --batch-size 2  # Reduce batch size
   ```

2. **Model Download Issues**
   ```python
   # Pre-download models
   from transformers import TrOCRProcessor
   processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
   ```

3. **Dataset Loading Errors**
   ```python
   # Check dataset structure
   import os
   print(os.listdir("handwritten_output/"))
   print(os.path.exists("handwritten_output/labels.txt"))
   ```

4. **Dependency Issues**
   ```bash
   pip install --upgrade transformers datasets torch
   ```

### Performance Tips

1. **Faster Training**
   - Use mixed precision: automatic with `fp16=True`
   - Increase batch size if memory allows
   - Use gradient accumulation for effective larger batches

2. **Memory Optimization**
   - Reduce image resolution in preprocessing
   - Use gradient checkpointing
   - Process datasets in chunks

3. **Better Results**
   - Increase training epochs
   - Tune learning rate
   - Augment training data
   - Improve text normalization

## License

MIT License - see LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{ocr_medical_parser_2024,
  title={OCR Medical Prescription Parser: An End-to-End Pipeline},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/ocr-medical-parser}
}
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## Support

For issues and questions:
- Create GitHub issues for bugs
- Check documentation for common problems
- Review example notebooks for usage patterns