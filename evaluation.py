"""
Evaluation module for OCR and NER metrics.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import evaluate
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from preprocessing import TextNormalizer


@dataclass
class EvaluationResults:
    """Container for evaluation results."""
    cer: float
    wer: float
    exact_match_rate: float
    ner_metrics: Dict[str, Any]
    sample_count: int
    predictions: List[str]
    references: List[str]


class OCREvaluator:
    """Evaluator for OCR metrics (CER, WER, Exact Match Rate)."""
    
    def __init__(self):
        self.cer_metric = evaluate.load("cer")
        self.wer_metric = evaluate.load("wer")
        self.text_normalizer = TextNormalizer()
    
    def evaluate(
        self, 
        predictions: List[str], 
        references: List[str],
        normalize: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate OCR predictions against ground truth.
        
        Args:
            predictions: List of predicted texts
            references: List of ground truth texts
            normalize: Whether to normalize texts before evaluation
            
        Returns:
            Dictionary with CER, WER, and exact match rate
        """
        if normalize:
            predictions = [self.text_normalizer.normalize_for_training(text) for text in predictions]
            references = [self.text_normalizer.normalize_for_training(text) for text in references]
        
        # Compute CER (Character Error Rate)
        cer = self.cer_metric.compute(predictions=predictions, references=references)
        
        # Compute WER (Word Error Rate)
        wer = self.wer_metric.compute(predictions=predictions, references=references)
        
        # Compute Exact Match Rate
        exact_matches = sum(1 for pred, ref in zip(predictions, references) if pred.strip() == ref.strip())
        exact_match_rate = exact_matches / len(predictions) if predictions else 0.0
        
        return {
            "cer": cer,
            "wer": wer,
            "exact_match_rate": exact_match_rate,
            "sample_count": len(predictions)
        }


class NERChunkEvaluator:
    """Evaluator for NER metrics using IOB2 tagging."""
    
    def __init__(self):
        self.entity_types = ["DRUG", "DOSAGE", "FREQUENCY", "FORM"]
    
    def extract_entities_from_ner_output(self, ner_output: List[Dict]) -> List[str]:
        """
        Convert NER model output to IOB2 format.
        
        Args:
            ner_output: List of entity dictionaries from NER model
            
        Returns:
            List of IOB2 tags
        """
        # This is a simplified version - in practice, you'd need to align
        # the entities with the original text tokens
        tags = []
        for entity in ner_output:
            entity_type = entity.get('entity_group', 'O').upper()
            if entity_type in self.entity_types:
                tags.append(f"B-{entity_type}")
            else:
                tags.append("O")
        return tags
    
    def evaluate_ner(
        self, 
        predicted_entities: List[List[str]], 
        true_entities: List[List[str]]
    ) -> Dict[str, Any]:
        """
        Evaluate NER predictions using seqeval.
        
        Args:
            predicted_entities: List of predicted entity sequences (IOB2 format)
            true_entities: List of true entity sequences (IOB2 format)
            
        Returns:
            Dictionary with precision, recall, F1 scores
        """
        # Overall metrics
        overall_f1 = f1_score(true_entities, predicted_entities)
        overall_precision = precision_score(true_entities, predicted_entities)
        overall_recall = recall_score(true_entities, predicted_entities)
        
        # Detailed classification report
        report = classification_report(true_entities, predicted_entities, output_dict=True)
        
        return {
            "overall": {
                "precision": overall_precision,
                "recall": overall_recall,
                "f1": overall_f1
            },
            "per_entity": report,
            "sample_count": len(predicted_entities)
        }


class ComprehensiveEvaluator:
    """Comprehensive evaluator for OCR + NER pipeline."""
    
    def __init__(self):
        self.ocr_evaluator = OCREvaluator()
        self.ner_evaluator = NERChunkEvaluator()
    
    def evaluate_pipeline(
        self,
        images: List[str],
        ground_truth_texts: List[str],
        ocr_predictions: List[str],
        corrected_predictions: List[str],
        predicted_entities: List[List[str]] = None,
        true_entities: List[List[str]] = None,
        output_dir: str = "./evaluation_results"
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of the OCR + correction pipeline.
        
        Args:
            images: List of image file paths
            ground_truth_texts: List of ground truth texts
            ocr_predictions: List of OCR predictions (before correction)
            corrected_predictions: List of corrected predictions (after MediPhi)
            predicted_entities: List of predicted entity sequences
            true_entities: List of true entity sequences
            output_dir: Directory to save evaluation results
            
        Returns:
            Dictionary with all evaluation results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Evaluate OCR only
        print("Evaluating OCR predictions...")
        ocr_metrics = self.ocr_evaluator.evaluate(ocr_predictions, ground_truth_texts)
        
        # Evaluate OCR + correction
        print("Evaluating corrected predictions...")
        corrected_metrics = self.ocr_evaluator.evaluate(corrected_predictions, ground_truth_texts)
        
        # Evaluate NER if entities are provided
        ner_metrics = {}
        if predicted_entities and true_entities:
            print("Evaluating NER predictions...")
            ner_metrics = self.ner_evaluator.evaluate_ner(predicted_entities, true_entities)
        
        # Create detailed results DataFrame
        results_df = pd.DataFrame({
            'image_path': images,
            'ground_truth': ground_truth_texts,
            'ocr_prediction': ocr_predictions,
            'corrected_prediction': corrected_predictions,
            'cer_before': [self.ocr_evaluator.cer_metric.compute(
                predictions=[ocr_pred], references=[gt]
            ) for ocr_pred, gt in zip(ocr_predictions, ground_truth_texts)],
            'cer_after': [self.ocr_evaluator.cer_metric.compute(
                predictions=[corr_pred], references=[gt]
            ) for corr_pred, gt in zip(corrected_predictions, ground_truth_texts)],
            'exact_match_ocr': [
                pred.strip() == gt.strip() 
                for pred, gt in zip(ocr_predictions, ground_truth_texts)
            ],
            'exact_match_corrected': [
                pred.strip() == gt.strip() 
                for pred, gt in zip(corrected_predictions, ground_truth_texts)
            ]
        })
        
        # Save detailed results
        results_csv_path = os.path.join(output_dir, "detailed_results.csv")
        results_df.to_csv(results_csv_path, index=False)
        print(f"Detailed results saved to {results_csv_path}")
        
        # Calculate improvement metrics
        cer_improvement = ocr_metrics["cer"] - corrected_metrics["cer"]
        wer_improvement = ocr_metrics["wer"] - corrected_metrics["wer"]
        emr_improvement = corrected_metrics["exact_match_rate"] - ocr_metrics["exact_match_rate"]
        
        # Compile final results
        final_results = {
            "ocr_only": ocr_metrics,
            "ocr_plus_correction": corrected_metrics,
            "improvements": {
                "cer_improvement": cer_improvement,
                "wer_improvement": wer_improvement,
                "emr_improvement": emr_improvement,
                "cer_improvement_percent": (cer_improvement / ocr_metrics["cer"]) * 100 if ocr_metrics["cer"] > 0 else 0,
                "wer_improvement_percent": (wer_improvement / ocr_metrics["wer"]) * 100 if ocr_metrics["wer"] > 0 else 0,
                "emr_improvement_percent": (emr_improvement / ocr_metrics["exact_match_rate"]) * 100 if ocr_metrics["exact_match_rate"] > 0 else 0
            },
            "ner_metrics": ner_metrics,
            "sample_statistics": {
                "total_samples": len(images),
                "avg_text_length": np.mean([len(text) for text in ground_truth_texts]),
                "avg_ocr_length": np.mean([len(text) for text in ocr_predictions]),
                "avg_corrected_length": np.mean([len(text) for text in corrected_predictions])
            }
        }
        
        # Save metrics summary
        metrics_json_path = os.path.join(output_dir, "metrics_summary.json")
        with open(metrics_json_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        print(f"Metrics summary saved to {metrics_json_path}")
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Total samples: {len(images)}")
        print(f"\nOCR Only Metrics:")
        print(f"  CER: {ocr_metrics['cer']:.4f}")
        print(f"  WER: {ocr_metrics['wer']:.4f}")
        print(f"  Exact Match Rate: {ocr_metrics['exact_match_rate']:.4f}")
        print(f"\nOCR + Correction Metrics:")
        print(f"  CER: {corrected_metrics['cer']:.4f}")
        print(f"  WER: {corrected_metrics['wer']:.4f}")
        print(f"  Exact Match Rate: {corrected_metrics['exact_match_rate']:.4f}")
        print(f"\nImprovements:")
        print(f"  CER improvement: {cer_improvement:.4f} ({final_results['improvements']['cer_improvement_percent']:.2f}%)")
        print(f"  WER improvement: {wer_improvement:.4f} ({final_results['improvements']['wer_improvement_percent']:.2f}%)")
        print(f"  EMR improvement: {emr_improvement:.4f} ({final_results['improvements']['emr_improvement_percent']:.2f}%)")
        
        if ner_metrics:
            print(f"\nNER Metrics:")
            print(f"  Overall F1: {ner_metrics['overall']['f1']:.4f}")
            print(f"  Overall Precision: {ner_metrics['overall']['precision']:.4f}")
            print(f"  Overall Recall: {ner_metrics['overall']['recall']:.4f}")
        
        return final_results


def evaluate_model_predictions(
    predictions_file: str,
    ground_truth_file: str,
    output_dir: str = "./evaluation_results"
) -> Dict[str, Any]:
    """
    Evaluate model predictions from saved files.
    
    Args:
        predictions_file: Path to CSV file with predictions
        ground_truth_file: Path to ground truth labels file
        output_dir: Directory to save evaluation results
        
    Returns:
        Evaluation results dictionary
    """
    # Load predictions
    pred_df = pd.read_csv(predictions_file)
    
    # Load ground truth
    gt_texts = []
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                gt_texts.append(parts[1])
    
    # Extract required columns
    images = pred_df['image_path'].tolist() if 'image_path' in pred_df.columns else [f"image_{i}" for i in range(len(pred_df))]
    ocr_predictions = pred_df['ocr_prediction'].tolist()
    corrected_predictions = pred_df['corrected_prediction'].tolist() if 'corrected_prediction' in pred_df.columns else ocr_predictions
    
    # Run evaluation
    evaluator = ComprehensiveEvaluator()
    return evaluator.evaluate_pipeline(
        images=images,
        ground_truth_texts=gt_texts,
        ocr_predictions=ocr_predictions,
        corrected_predictions=corrected_predictions,
        output_dir=output_dir
    )


if __name__ == "__main__":
    # Test evaluation module
    print("Testing evaluation module...")
    
    # Sample data for testing
    sample_ground_truth = [
        "Take aspirin 100mg twice daily",
        "Paracetamol 500mg three times a day",
        "Apply cream to affected area"
    ]
    
    sample_ocr_predictions = [
        "Take asprin 100mg twice daily",  # OCR error: aspirin -> asprin
        "Paracetamol 500mg thre times a day",  # OCR error: three -> thre
        "Apply cream to affeted area"  # OCR error: affected -> affeted
    ]
    
    sample_corrected_predictions = [
        "Take aspirin 100mg twice daily",  # Corrected
        "Paracetamol 500mg three times a day",  # Corrected
        "Apply cream to affected area"  # Corrected
    ]
    
    sample_images = ["test1.png", "test2.png", "test3.png"]
    
    # Test OCR evaluator
    ocr_evaluator = OCREvaluator()
    ocr_results = ocr_evaluator.evaluate(sample_ocr_predictions, sample_ground_truth)
    print("OCR Evaluation Results:", ocr_results)
    
    # Test comprehensive evaluator
    comp_evaluator = ComprehensiveEvaluator()
    results = comp_evaluator.evaluate_pipeline(
        images=sample_images,
        ground_truth_texts=sample_ground_truth,
        ocr_predictions=sample_ocr_predictions,
        corrected_predictions=sample_corrected_predictions,
        output_dir="./test_evaluation"
    )
    
    print("Comprehensive evaluation test completed!")