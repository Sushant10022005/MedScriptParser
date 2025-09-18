"""
Main pipeline for end-to-end OCR-based handwritten medical prescription parser.
"""

import os
# Ensure Weights & Biases is completely disabled before importing training
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
import json
import time
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd

# Import all modules
from dataset_loader import load_dataset, DatasetConfig
from preprocessing import preprocess_dataset, TextNormalizer
from training import train_model, TrOCRTrainingConfig
from inference import infer_on_test_dataset, TrOCRInferenceEngine
from mediphi_correction import correct_with_mediphi, save_correction_results
from evaluation import ComprehensiveEvaluator
from visualization import create_visualizations


class OCRPipelineConfig:
    """Configuration for the entire OCR pipeline."""
    
    def __init__(
        self,
        # Dataset settings
        dataset_dir: str = "./handwritten_output",
        labels_file: str = "./handwritten_output/labels.txt",
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        
        # Training settings
        model_name: str = "microsoft/trocr-large-handwritten",
        output_dir: str = "./ocr_project",
        num_epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        
        # Inference settings
        inference_batch_size: int = 16,
        
        # Correction settings
        ner_model: str = "microsoft/MediPhi",
        vocab_file: str = "./medical_vocab.json",
        
        # Output settings
        results_dir: str = "./results",
        plots_dir: str = "./plots",
        
        # General settings
        seed: int = 42,
        max_samples: Optional[int] = None,  # For testing with subset
        skip_training: bool = False,  # Skip training if model exists
        skip_correction: bool = False,  # Skip correction step
    ):
        self.dataset_dir = dataset_dir
        self.labels_file = labels_file
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        
        self.model_name = model_name
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self.inference_batch_size = inference_batch_size
        
        self.ner_model = ner_model
        self.vocab_file = vocab_file
        
        self.results_dir = results_dir
        self.plots_dir = plots_dir
        
        self.seed = seed
        self.max_samples = max_samples
        self.skip_training = skip_training
        self.skip_correction = skip_correction
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)


class OCRPipeline:
    """Main OCR pipeline orchestrator."""
    
    def __init__(self, config: OCRPipelineConfig):
        self.config = config
        self.results = {}
        self.start_time = time.time()
        
        print("="*60)
        print("OCR Medical Prescription Parser Pipeline")
        print("="*60)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Configuration:")
        print(f"  Dataset: {config.dataset_dir}")
        print(f"  Model: {config.model_name}")
        print(f"  Output: {config.output_dir}")
        print(f"  Epochs: {config.num_epochs}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Skip training: {config.skip_training}")
        print("="*60)
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete pipeline."""
        try:
            # Step 1: Load and prepare dataset
            self.load_and_prepare_data()
            
            # Step 2: Train model (if not skipping)
            if not self.config.skip_training:
                self.train_model()
            else:
                print("Skipping training step...")
            
            # Step 3: Run inference
            self.run_inference()
            
            # Step 4: Apply corrections (if not skipping)
            if not self.config.skip_correction:
                self.apply_corrections()
            else:
                print("Skipping correction step...")
            
            # Step 5: Evaluate results
            self.evaluate_results()
            
            # Step 6: Generate visualizations
            self.generate_visualizations()
            
            # Step 7: Save final results
            self.save_final_results()
            
            # Print summary
            self.print_summary()
            
            return self.results
            
        except Exception as e:
            print(f"Pipeline failed with error: {e}")
            raise e
    
    def load_and_prepare_data(self) -> None:
        """Load and prepare the dataset."""
        print("\n" + "-"*40)
        print("STEP 1: Loading and preparing dataset")
        print("-"*40)
        
        # Configure dataset loading
        dataset_config = DatasetConfig(
            dataset_dir=self.config.dataset_dir,
            labels_file=self.config.labels_file,
            train_split=self.config.train_split,
            val_split=self.config.val_split,
            test_split=self.config.test_split,
            max_samples=self.config.max_samples,
            seed=self.config.seed
        )
        
        # Load dataset
        self.datasets = load_dataset(dataset_config)
        
        print(f"Dataset loaded:")
        print(f"  Train: {len(self.datasets['train'])} samples")
        print(f"  Validation: {len(self.datasets['validation'])} samples")  
        print(f"  Test: {len(self.datasets['test'])} samples")
        
        # Preprocess datasets
        print("Preprocessing datasets...")
        self.train_dataset = preprocess_dataset(self.datasets['train'])
        self.val_dataset = preprocess_dataset(self.datasets['validation'])
        self.test_dataset = preprocess_dataset(self.datasets['test'])
        
        print("Dataset preparation completed!")
        
        # Save dataset info
        self.results['dataset_info'] = {
            'train_samples': len(self.datasets['train']),
            'val_samples': len(self.datasets['validation']),
            'test_samples': len(self.datasets['test']),
            'total_samples': len(self.datasets['train']) + len(self.datasets['validation']) + len(self.datasets['test'])
        }
    
    def train_model(self) -> None:
        """Train the TrOCR model."""
        print("\n" + "-"*40)
        print("STEP 2: Training TrOCR model")
        print("-"*40)
        
        # Configure training
        training_config = TrOCRTrainingConfig(
            model_name=self.config.model_name,
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            seed=self.config.seed
        )
        
        # Train model
        training_results = train_model(
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            config=training_config
        )
        
        self.results['training_results'] = training_results
        print("Model training completed!")
    
    def run_inference(self) -> None:
        """Run inference on test dataset."""
        print("\n" + "-"*40)
        print("STEP 3: Running inference on test set")
        print("-"*40)
        
        # Determine model path
        model_path = self.config.output_dir
        if not os.path.exists(os.path.join(model_path, "config.json")):
            print(f"Trained model not found at {model_path}, using base model")
            model_path = self.config.model_name
        
        # Run inference
        inference_results = infer_on_test_dataset(
            model_path=model_path,
            test_dataset=self.datasets['test'],
            output_file=os.path.join(self.config.results_dir, "ocr_predictions.csv"),
            batch_size=self.config.inference_batch_size
        )
        
        self.results['inference_results'] = inference_results
        self.ocr_predictions = [r['predicted_text'] for r in inference_results['results']]
        
        print(f"Inference completed on {len(self.ocr_predictions)} samples")
    
    def apply_corrections(self) -> None:
        """Apply MediPhi-based corrections."""
        print("\n" + "-"*40)
        print("STEP 4: Applying MediPhi corrections")
        print("-"*40)
        
        # Get training texts for vocabulary building
        training_texts = [item['text'] for item in self.datasets['train']]
        
        # Apply corrections
        correction_results = correct_with_mediphi(
            ocr_texts=self.ocr_predictions,
            training_texts=training_texts,
            vocab_file=self.config.vocab_file,
            ner_model=self.config.ner_model
        )
        
        # Extract corrected texts
        self.corrected_predictions = [r.corrected_text for r in correction_results]
        
        # Save correction results
        correction_file = os.path.join(self.config.results_dir, "correction_results.csv")
        save_correction_results(correction_results, correction_file)
        
        self.results['correction_results'] = {
            'correction_file': correction_file,
            'num_corrections': sum(len(r.corrections_made) for r in correction_results),
            'avg_confidence': sum(r.confidence_score for r in correction_results) / len(correction_results)
        }
        
        print(f"Corrections applied to {len(self.corrected_predictions)} samples")
    
    def evaluate_results(self) -> None:
        """Evaluate OCR and correction results."""
        print("\n" + "-"*40)
        print("STEP 5: Evaluating results")
        print("-"*40)
        
        # Get ground truth texts
        ground_truth_texts = [item['text'] for item in self.datasets['test']]
        image_paths = [f"test_image_{i}.png" for i in range(len(ground_truth_texts))]
        
        # Use corrected predictions if available, otherwise use OCR predictions
        corrected_preds = getattr(self, 'corrected_predictions', self.ocr_predictions)
        
        # Run comprehensive evaluation
        evaluator = ComprehensiveEvaluator()
        evaluation_results = evaluator.evaluate_pipeline(
            images=image_paths,
            ground_truth_texts=ground_truth_texts,
            ocr_predictions=self.ocr_predictions,
            corrected_predictions=corrected_preds,
            output_dir=self.config.results_dir
        )
        
        self.results['evaluation_results'] = evaluation_results
        print("Evaluation completed!")
    
    def generate_visualizations(self) -> None:
        """Generate all visualizations."""
        print("\n" + "-"*40)
        print("STEP 6: Generating visualizations")
        print("-"*40)
        
        # Get training history if available
        training_history = None
        if 'training_results' in self.results:
            training_history = self.results['training_results'].get('training_history')
        
        # Generate visualizations
        create_visualizations(
            training_history=training_history,
            evaluation_results=self.results.get('evaluation_results'),
            detailed_results_file=os.path.join(self.config.results_dir, "detailed_results.csv"),
            output_dir=self.config.plots_dir
        )
        
        print("Visualizations generated!")
    
    def save_final_results(self) -> None:
        """Save final results and generate summary."""
        print("\n" + "-"*40)
        print("STEP 7: Saving final results")
        print("-"*40)
        
        # Calculate total runtime
        total_time = time.time() - self.start_time
        
        # Create final summary
        final_summary = {
            'experiment_info': {
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_runtime_seconds': total_time,
                'total_runtime_formatted': f"{total_time // 3600:.0f}h {(total_time % 3600) // 60:.0f}m {total_time % 60:.0f}s"
            },
            'configuration': {
                'model_name': self.config.model_name,
                'num_epochs': self.config.num_epochs,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
                'dataset_size': self.results['dataset_info']['total_samples']
            },
            'results': self.results
        }
        
        # Save summary
        summary_file = os.path.join(self.config.results_dir, "experiment_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(final_summary, f, indent=2, default=str)
        
        # Generate results report
        self.generate_results_report()
        
        print(f"Final results saved to {self.config.results_dir}")
    
    def generate_results_report(self) -> None:
        """Generate a human-readable results report."""
        report_lines = []
        
        # Header
        report_lines.extend([
            "="*80,
            "OCR MEDICAL PRESCRIPTION PARSER - RESULTS REPORT",
            "="*80,
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total runtime: {time.time() - self.start_time:.0f} seconds",
            ""
        ])
        
        # Dataset information
        dataset_info = self.results.get('dataset_info', {})
        report_lines.extend([
            "DATASET INFORMATION:",
            f"  Total samples: {dataset_info.get('total_samples', 'N/A')}",
            f"  Training samples: {dataset_info.get('train_samples', 'N/A')}",
            f"  Validation samples: {dataset_info.get('val_samples', 'N/A')}",
            f"  Test samples: {dataset_info.get('test_samples', 'N/A')}",
            ""
        ])
        
        # Training information
        if 'training_results' in self.results:
            report_lines.extend([
                "TRAINING INFORMATION:",
                f"  Model: {self.config.model_name}",
                f"  Epochs: {self.config.num_epochs}",
                f"  Batch size: {self.config.batch_size}",
                f"  Learning rate: {self.config.learning_rate}",
                f"  Model saved to: {self.config.output_dir}",
                ""
            ])
        
        # Evaluation results
        if 'evaluation_results' in self.results:
            eval_results = self.results['evaluation_results']
            ocr_metrics = eval_results.get('ocr_only', {})
            corrected_metrics = eval_results.get('ocr_plus_correction', {})
            improvements = eval_results.get('improvements', {})
            
            report_lines.extend([
                "EVALUATION RESULTS:",
                "OCR Only Performance:",
                f"  Character Error Rate (CER): {ocr_metrics.get('cer', 'N/A'):.4f}",
                f"  Word Error Rate (WER): {ocr_metrics.get('wer', 'N/A'):.4f}",
                f"  Exact Match Rate: {ocr_metrics.get('exact_match_rate', 'N/A'):.4f}",
                "",
                "OCR + MediPhi Correction Performance:",
                f"  Character Error Rate (CER): {corrected_metrics.get('cer', 'N/A'):.4f}",
                f"  Word Error Rate (WER): {corrected_metrics.get('wer', 'N/A'):.4f}",
                f"  Exact Match Rate: {corrected_metrics.get('exact_match_rate', 'N/A'):.4f}",
                "",
                "Improvements:",
                f"  CER improvement: {improvements.get('cer_improvement_percent', 'N/A'):.2f}%",
                f"  WER improvement: {improvements.get('wer_improvement_percent', 'N/A'):.2f}%",
                f"  EMR improvement: {improvements.get('emr_improvement_percent', 'N/A'):.2f}%",
                ""
            ])
        
        # Files generated
        report_lines.extend([
            "FILES GENERATED:",
            f"  Model checkpoint: {self.config.output_dir}/",
            f"  OCR predictions: {self.config.results_dir}/ocr_predictions.csv",
            f"  Detailed results: {self.config.results_dir}/detailed_results.csv",
            f"  Metrics summary: {self.config.results_dir}/metrics_summary.json",
            f"  Visualizations: {self.config.plots_dir}/",
            f"  Experiment summary: {self.config.results_dir}/experiment_summary.json",
            ""
        ])
        
        # Save report
        report_file = os.path.join(self.config.results_dir, "results_report.txt")
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Results report saved to {report_file}")
    
    def print_summary(self) -> None:
        """Print final summary to console."""
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        if 'evaluation_results' in self.results:
            eval_results = self.results['evaluation_results']
            ocr_metrics = eval_results.get('ocr_only', {})
            corrected_metrics = eval_results.get('ocr_plus_correction', {})
            
            print(f"Final Results:")
            print(f"  OCR CER: {ocr_metrics.get('cer', 0):.4f} → {corrected_metrics.get('cer', 0):.4f}")
            print(f"  OCR WER: {ocr_metrics.get('wer', 0):.4f} → {corrected_metrics.get('wer', 0):.4f}")
            print(f"  Exact Match: {ocr_metrics.get('exact_match_rate', 0):.4f} → {corrected_metrics.get('exact_match_rate', 0):.4f}")
        
        print(f"Total runtime: {time.time() - self.start_time:.0f} seconds")
        print(f"Results saved to: {self.config.results_dir}")
        print("="*60)


def main(config: Optional[OCRPipelineConfig] = None) -> Dict[str, Any]:
    """
    Main function to run the complete OCR pipeline.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        Dictionary with all results
    """
    if config is None:
        config = OCRPipelineConfig()
    
    pipeline = OCRPipeline(config)
    return pipeline.run_full_pipeline()


def create_test_config() -> OCRPipelineConfig:
    """Create a test configuration for quick testing."""
    return OCRPipelineConfig(
        max_samples=100,  # Use only 100 samples for testing
        num_epochs=1,     # Train for only 1 epoch
        batch_size=4,     # Smaller batch size
        skip_training=False,
        skip_correction=False
    )


def create_full_config() -> OCRPipelineConfig:
    """Create a full configuration for complete training."""
    return OCRPipelineConfig(
        num_epochs=10,
        batch_size=8,
        learning_rate=5e-5,
        skip_training=False,
        skip_correction=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR Medical Prescription Parser Pipeline")
    parser.add_argument("--test", action="store_true", help="Run in test mode with smaller dataset")
    parser.add_argument("--skip-training", action="store_true", help="Skip training step")
    parser.add_argument("--skip-correction", action="store_true", help="Skip correction step")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to use")
    parser.add_argument("--output-dir", type=str, default="./ocr_project", help="Output directory")
    
    args = parser.parse_args()
    
    # Create configuration based on arguments
    if args.test:
        config = create_test_config()
    else:
        config = create_full_config()
    
    # Apply command line arguments
    if args.skip_training:
        config.skip_training = True
    if args.skip_correction:
        config.skip_correction = True
    if args.max_samples:
        config.max_samples = args.max_samples
    
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.output_dir = args.output_dir
    
    # Run pipeline
    try:
        results = main(config)
        print("\nPipeline completed successfully!")
        
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        raise