"""
Inference module for TrOCR model.
"""

import os
import torch
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from datasets import Dataset
from tqdm import tqdm
from preprocessing import TextNormalizer


class TrOCRInferenceEngine:
    """Inference engine for TrOCR model."""
    
    def __init__(
        self, 
        model_path: str,
        device: Optional[str] = None,
        batch_size: int = 8
    ):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the trained model directory
            device: Device to run inference on ('cuda' or 'cpu')
            batch_size: Batch size for inference
        """
        self.model_path = model_path
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.text_normalizer = TextNormalizer()
        
        self.processor = None
        self.model = None
        
        # Load model and processor
        self.load_model()
    
    def load_model(self):
        """Load the trained TrOCR model and processor."""
        print(f"Loading model from {self.model_path}...")
        
        try:
            self.processor = TrOCRProcessor.from_pretrained(self.model_path)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to base model...")
            
            # Fallback to base model
            base_model = "microsoft/trocr-large-handwritten"
            self.processor = TrOCRProcessor.from_pretrained(base_model)
            self.model = VisionEncoderDecoderModel.from_pretrained(base_model)
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Base model loaded on {self.device}")
    
    def preprocess_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """
        Preprocess a single image for inference.
        
        Args:
            image: Image path or PIL Image object
            
        Returns:
            Preprocessed image tensor
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be a file path or PIL Image object")
        
        # Use processor to prepare image
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        return pixel_values.to(self.device)
    
    def preprocess_batch(self, images: List[Union[str, Image.Image]]) -> torch.Tensor:
        """
        Preprocess a batch of images for inference.
        
        Args:
            images: List of image paths or PIL Image objects
            
        Returns:
            Batch of preprocessed image tensors
        """
        # Load and convert images if they are paths
        pil_images = []
        for img in images:
            if isinstance(img, str):
                pil_images.append(Image.open(img).convert('RGB'))
            else:
                pil_images.append(img)
        
        # Process batch
        batch = self.processor(pil_images, return_tensors="pt", padding=True)
        return batch.pixel_values.to(self.device)
    
    def predict_single(
        self, 
        image: Union[str, Image.Image], 
        return_confidence: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """
        Predict text for a single image.
        
        Args:
            image: Image path or PIL Image object
            return_confidence: Whether to return confidence scores
            
        Returns:
            Predicted text or dictionary with text and confidence
        """
        with torch.no_grad():
            pixel_values = self.preprocess_image(image)
            
            # Generate prediction
            generated_ids = self.model.generate(
                pixel_values,
                max_length=128,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=2.0,
                return_dict_in_generate=return_confidence,
                output_scores=return_confidence
            )
            
            if return_confidence:
                sequences = generated_ids.sequences
                scores = generated_ids.sequences_scores
                predicted_text = self.processor.decode(sequences[0], skip_special_tokens=True)
                confidence = torch.exp(scores[0]).item()
                
                return {
                    'text': predicted_text,
                    'confidence': confidence,
                    'normalized_text': self.text_normalizer.normalize_for_training(predicted_text)
                }
            else:
                predicted_text = self.processor.decode(generated_ids[0], skip_special_tokens=True)
                return predicted_text
    
    def predict_batch(
        self, 
        images: List[Union[str, Image.Image]], 
        return_confidence: bool = False,
        show_progress: bool = True
    ) -> List[Union[str, Dict[str, Any]]]:
        """
        Predict text for a batch of images.
        
        Args:
            images: List of image paths or PIL Image objects
            return_confidence: Whether to return confidence scores
            show_progress: Whether to show progress bar
            
        Returns:
            List of predicted texts or dictionaries with text and confidence
        """
        predictions = []
        
        # Process in batches
        for i in tqdm(range(0, len(images), self.batch_size), 
                     desc="Running inference", disable=not show_progress):
            batch_images = images[i:i + self.batch_size]
            
            with torch.no_grad():
                pixel_values = self.preprocess_batch(batch_images)
                
                # Generate predictions
                generated_ids = self.model.generate(
                    pixel_values,
                    max_length=128,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    length_penalty=2.0,
                    return_dict_in_generate=return_confidence,
                    output_scores=return_confidence
                )
                
                if return_confidence:
                    sequences = generated_ids.sequences
                    scores = generated_ids.sequences_scores
                    
                    for j in range(len(batch_images)):
                        predicted_text = self.processor.decode(sequences[j], skip_special_tokens=True)
                        confidence = torch.exp(scores[j]).item() if j < len(scores) else 0.0
                        
                        predictions.append({
                            'text': predicted_text,
                            'confidence': confidence,
                            'normalized_text': self.text_normalizer.normalize_for_training(predicted_text)
                        })
                else:
                    for j in range(len(batch_images)):
                        predicted_text = self.processor.decode(generated_ids[j], skip_special_tokens=True)
                        predictions.append(predicted_text)
        
        return predictions
    
    def predict_dataset(
        self, 
        dataset: Dataset, 
        image_column: str = 'image',
        save_path: Optional[str] = None,
        return_confidence: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Run inference on a dataset.
        
        Args:
            dataset: Dataset with images
            image_column: Name of the image column
            save_path: Path to save predictions (optional)
            return_confidence: Whether to return confidence scores
            
        Returns:
            List of prediction results
        """
        images = [item[image_column] for item in dataset]
        image_paths = [item.get('image_path', f'image_{i}') for i, item in enumerate(dataset)]
        
        print(f"Running inference on {len(images)} images...")
        predictions = self.predict_batch(images, return_confidence=return_confidence)
        
        # Create results
        results = []
        for i, pred in enumerate(predictions):
            if return_confidence:
                result = {
                    'image_path': image_paths[i],
                    'predicted_text': pred['text'],
                    'normalized_text': pred['normalized_text'],
                    'confidence': pred['confidence']
                }
            else:
                result = {
                    'image_path': image_paths[i],
                    'predicted_text': pred,
                    'normalized_text': self.text_normalizer.normalize_for_training(pred)
                }
            results.append(result)
        
        # Save results if path provided
        if save_path:
            df = pd.DataFrame(results)
            df.to_csv(save_path, index=False)
            print(f"Predictions saved to {save_path}")
        
        return results


def load_inference_engine(model_path: str, device: Optional[str] = None) -> TrOCRInferenceEngine:
    """
    Load TrOCR inference engine.
    
    Args:
        model_path: Path to trained model
        device: Device to use for inference
        
    Returns:
        Initialized inference engine
    """
    return TrOCRInferenceEngine(model_path, device)


def infer_on_images(
    model_path: str,
    image_paths: List[str],
    output_file: Optional[str] = None,
    batch_size: int = 8,
    device: Optional[str] = None
) -> List[str]:
    """
    Run inference on a list of image paths.
    
    Args:
        model_path: Path to trained model
        image_paths: List of image file paths
        output_file: Path to save results (optional)
        batch_size: Batch size for inference
        device: Device to use for inference
        
    Returns:
        List of predicted texts
    """
    # Initialize inference engine
    engine = TrOCRInferenceEngine(model_path, device, batch_size)
    
    # Run inference
    predictions = engine.predict_batch(image_paths, return_confidence=True)
    
    # Extract texts
    predicted_texts = [pred['text'] if isinstance(pred, dict) else pred for pred in predictions]
    
    # Save results if requested
    if output_file:
        results_df = pd.DataFrame({
            'image_path': image_paths,
            'predicted_text': predicted_texts,
            'confidence': [pred['confidence'] if isinstance(pred, dict) else 1.0 for pred in predictions]
        })
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    
    return predicted_texts


def infer_on_test_dataset(
    model_path: str,
    test_dataset: Dataset,
    output_file: str = "./test_predictions.csv",
    batch_size: int = 8,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run inference on test dataset and save results.
    
    Args:
        model_path: Path to trained model
        test_dataset: Test dataset
        output_file: Path to save predictions
        batch_size: Batch size for inference
        device: Device to use for inference
        
    Returns:
        Dictionary with inference results and statistics
    """
    # Initialize inference engine
    engine = TrOCRInferenceEngine(model_path, device, batch_size)
    
    # Run inference
    results = engine.predict_dataset(
        test_dataset, 
        save_path=output_file,
        return_confidence=True
    )
    
    # Calculate statistics
    confidences = [r['confidence'] for r in results]
    text_lengths = [len(r['predicted_text']) for r in results]
    
    stats = {
        'total_samples': len(results),
        'avg_confidence': sum(confidences) / len(confidences),
        'min_confidence': min(confidences),
        'max_confidence': max(confidences),
        'avg_text_length': sum(text_lengths) / len(text_lengths),
        'predictions_file': output_file
    }
    
    print(f"Inference completed on {stats['total_samples']} samples")
    print(f"Average confidence: {stats['avg_confidence']:.4f}")
    print(f"Average text length: {stats['avg_text_length']:.1f}")
    
    return {
        'results': results,
        'statistics': stats
    }


if __name__ == "__main__":
    # Test inference module
    print("Testing inference module...")
    
    # Test with base model (since we don't have a trained model yet)
    try:
        # Initialize with base model
        engine = TrOCRInferenceEngine("microsoft/trocr-large-handwritten")
        
        # Test single image inference (using a sample from our dataset)
        sample_image_path = "handwritten_output/0.png"
        if os.path.exists(sample_image_path):
            prediction = engine.predict_single(sample_image_path, return_confidence=True)
            print(f"Sample prediction: {prediction}")
        else:
            print("Sample image not found, skipping test")
        
        print("Inference module test completed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        print("Note: This is expected if running without GPU or model files")