"""
Training module for TrOCR fine-tuning on medical prescriptions.
"""

import os
# Hard-disable Weights & Biases in all contexts (no prompts, no offline runs)
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
# Reduce CUDA fragmentation during long eval/generate
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor, 
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset
import evaluate
from preprocessing import TextNormalizer


@dataclass
class TrOCRTrainingConfig:
    """Configuration for TrOCR training."""
    model_name: str = "microsoft/trocr-large-handwritten"
    output_dir: str = "./trocr_medical_checkpoints"
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    learning_rate: float = 5e-5
    warmup_steps: int = 500
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    eval_strategy: str = "steps"  # Changed from evaluation_strategy
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_cer"
    greater_is_better: bool = False
    save_total_limit: int = 3
    seed: int = 42
    fp16: bool = True  # Use mixed precision if GPU available
    # Use 0 workers on Windows to avoid multiprocessing spawn/pickle issues
    dataloader_num_workers: int = 0 if os.name == "nt" else 4
    max_target_length: int = 128
    early_stopping_patience: int = 3


class TrOCRDataCollator:
    """Data collator for TrOCR training."""
    
    def __init__(self, processor: TrOCRProcessor, max_target_length: int = 128):
        self.processor = processor
        self.max_target_length = max_target_length
    
    def __call__(self, batch):
        """Builds model inputs on the fly from raw dataset rows to reduce RAM usage."""
        # Expect dataset rows to have 'image' (datasets.Image -> PIL) and 'text'/'normalized_text'
        images = [item['image'] for item in batch]
        texts = [item.get('normalized_text', item['text']) for item in batch]

        # Build pixel tensors using the image processor only
        pixel_values = self.processor(images=images, return_tensors="pt").pixel_values

        # Tokenize labels with the tokenizer
        tokenized = self.processor.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        )
        labels = tokenized.input_ids
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            'pixel_values': pixel_values,
            'labels': labels,
        }


class TrOCRTrainer:
    """Trainer class for TrOCR fine-tuning."""
    
    def __init__(self, config: TrOCRTrainingConfig):
        self.config = config
        self.processor = None
        self.model = None
        self.text_normalizer = TextNormalizer()
        
        # Load evaluation metrics
        self.cer_metric = evaluate.load("cer")
        self.wer_metric = evaluate.load("wer")
        
        # Set random seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
    def load_model_and_processor(self):
        """Load the TrOCR model and processor."""
        print(f"Loading model and processor: {self.config.model_name}")
        
        self.processor = TrOCRProcessor.from_pretrained(self.config.model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.config.model_name)
        
        # Set special tokens
        self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        
        # Set beam search parameters
        self.model.config.eos_token_id = self.processor.tokenizer.sep_token_id
        self.model.config.max_length = self.config.max_target_length
        self.model.config.early_stopping = True
        self.model.config.no_repeat_ngram_size = 3
        self.model.config.length_penalty = 2.0
        self.model.config.num_beams = 4
        
        # Reduce memory footprint without changing numerical training objective
        try:
            self.model.gradient_checkpointing_enable()
        except Exception:
            pass
        # Disable cache for correct gradients with checkpointing
        if hasattr(self.model, 'config'):
            self.model.config.use_cache = False
        print("Model and processor loaded successfully!")
        
    def compute_metrics(self, eval_pred):
        """Compute CER and WER metrics."""
        # EvalPrediction can contain nested structures depending on transformers version
        predictions = getattr(eval_pred, 'predictions', None)
        labels = getattr(eval_pred, 'label_ids', None)
        # Fallback for transformers versions that pass a simple (preds, labels) tuple
        if predictions is None and (isinstance(eval_pred, (list, tuple)) and len(eval_pred) == 2):
            predictions, labels = eval_pred

        # Unwrap common tuple/list wrappers from generate outputs
        if isinstance(predictions, (list, tuple)) and len(predictions) > 0 and not isinstance(predictions[0], (int, np.integer)):
            # Prefer the first element if it's a (sequences, ...) tuple
            if isinstance(predictions[0], (np.ndarray, list)):
                predictions = predictions[0]

        # Ensure numpy arrays (convert from torch if needed)
        try:
            import torch as _torch  # local alias to avoid shadowing
        except Exception:
            _torch = None
        if _torch is not None and isinstance(predictions, _torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if _torch is not None and isinstance(labels, _torch.Tensor):
            labels = labels.detach().cpu().numpy()
        if isinstance(predictions, list):
            predictions = np.array(predictions)
        if isinstance(labels, list):
            labels = np.array(labels)

        # Squeeze away beams or singleton dims: expect shape (N, T)
        if isinstance(predictions, np.ndarray) and predictions.ndim == 3:
            # Take the first beam if present: (batch, beams, seq_len) -> (batch, seq_len)
            predictions = predictions[:, 0, :]
        if isinstance(labels, np.ndarray) and labels.ndim == 3:
            labels = labels[:, 0, :]
        # If 1D, make it a batch of size 1
        if isinstance(predictions, np.ndarray) and predictions.ndim == 1:
            predictions = predictions[None, :]
        if isinstance(labels, np.ndarray) and labels.ndim == 1:
            labels = labels[None, :]

        # Convert to integer dtype to satisfy tokenizer fast API
        if isinstance(predictions, np.ndarray) and not np.issubdtype(predictions.dtype, np.integer):
            predictions = predictions.astype(np.int64)
        if isinstance(labels, np.ndarray) and not np.issubdtype(labels.dtype, np.integer):
            labels = labels.astype(np.int64)

        # Sanitize prediction ids to valid tokenizer range to avoid OverflowError in fast decode
        tok = self.processor.tokenizer
        pad_id = getattr(tok, 'pad_token_id', 0) or 0
        unk_id = getattr(tok, 'unk_token_id', pad_id)
        max_id = None
        try:
            max_id = len(tok) - 1  # total vocab including added tokens
        except Exception:
            pass
        vs = getattr(tok, 'vocab_size', None)
        if isinstance(vs, int):
            max_id = max(vs - 1, max_id) if isinstance(max_id, int) else vs - 1
        if not isinstance(max_id, int):
            try:
                vocab = tok.get_vocab()
                if isinstance(vocab, dict) and vocab:
                    max_id = max(vocab.values())
            except Exception:
                pass
        if not isinstance(max_id, int):
            max_id = 1_000_000  # conservative cap to avoid overflow in tokenizer

        def _sanitize_ids(arr_like):
            # Handles numpy arrays or nested Python lists
            if isinstance(arr_like, np.ndarray):
                x = arr_like
                x = np.where(x < 0, pad_id, x)
                x = np.where(x > max_id, unk_id, x)
                return x
            if isinstance(arr_like, list):
                def clamp_id(v):
                    try:
                        v_int = int(v)
                    except Exception:
                        return unk_id
                    if v_int < 0:
                        return pad_id
                    if v_int > max_id:
                        return unk_id
                    return v_int
                if len(arr_like) > 0 and isinstance(arr_like[0], list):
                    return [[clamp_id(t) for t in seq] for seq in arr_like]
                else:
                    return [clamp_id(t) for t in arr_like]
            return arr_like

        predictions = _sanitize_ids(predictions)

        # Replace -100 in labels with pad token id for proper decoding
        if labels is not None:
            pad_id = self.processor.tokenizer.pad_token_id
            labels = np.where(labels != -100, labels, pad_id)

        # Decode predictions and labels (convert numpy arrays to lists for fast tokenizers)
        def _to_list(x):
            return x.tolist() if isinstance(x, np.ndarray) else x
        pred_texts = self.processor.batch_decode(_to_list(predictions), skip_special_tokens=True)
        label_texts = self.processor.batch_decode(_to_list(labels), skip_special_tokens=True)
        
        # Normalize texts for fair comparison
        pred_texts = [self.text_normalizer.normalize_for_training(text) for text in pred_texts]
        label_texts = [self.text_normalizer.normalize_for_training(text) for text in label_texts]
        
        # Compute CER (Character Error Rate)
        cer = self.cer_metric.compute(predictions=pred_texts, references=label_texts)
        
        # Compute WER (Word Error Rate) 
        wer = self.wer_metric.compute(predictions=pred_texts, references=label_texts)
        
        # Compute Exact Match Rate
        exact_matches = sum(1 for pred, ref in zip(pred_texts, label_texts) if pred.strip() == ref.strip())
        exact_match_rate = exact_matches / len(pred_texts)
        
        return {
            "cer": cer,
            "wer": wer, 
            "exact_match_rate": exact_match_rate,
            "num_samples": len(pred_texts)
        }
    
    def train(self, train_dataset: Dataset, eval_dataset: Dataset) -> Dict[str, Any]:
        """Train the TrOCR model."""
        if self.model is None or self.processor is None:
            self.load_model_and_processor()
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Setup training arguments (robust to Transformers version differences)
        import inspect
        ta_sig = inspect.signature(TrainingArguments.__init__).parameters
        def supported(key: str) -> bool:
            return key in ta_sig

        # Build kwargs and only keep supported keys
        ta_kwargs = {
            'output_dir': self.config.output_dir,
            'num_train_epochs': self.config.num_train_epochs,
            'per_device_train_batch_size': self.config.per_device_train_batch_size,
            'per_device_eval_batch_size': self.config.per_device_eval_batch_size,
            # Accumulate eval predictions to CPU regularly to avoid OOM
            'eval_accumulation_steps': 1,
            'learning_rate': self.config.learning_rate,
            'warmup_steps': self.config.warmup_steps,
            'logging_steps': self.config.logging_steps,
            'eval_steps': self.config.eval_steps,
            'save_steps': self.config.save_steps,
            'save_strategy': self.config.save_strategy,
            'load_best_model_at_end': self.config.load_best_model_at_end,
            'metric_for_best_model': self.config.metric_for_best_model,
            'greater_is_better': self.config.greater_is_better,
            'save_total_limit': self.config.save_total_limit,
            'seed': self.config.seed,
            'fp16': (self.config.fp16 and torch.cuda.is_available()),
            'dataloader_num_workers': (0 if os.name == "nt" else self.config.dataloader_num_workers),
            'dataloader_pin_memory': False,
            'dataloader_persistent_workers': False,
            'remove_unused_columns': False,
            'report_to': 'none',
            'run_name': 'trocr-training',
            # Generate sequences during eval to compute CER/WER correctly
            'predict_with_generate': True,
            'generation_max_length': self.config.max_target_length,
            'generation_num_beams': 4,
        }
        # Prefer evaluation_strategy; fall back to eval_strategy if needed
        if 'evaluation_strategy' in ta_sig:
            ta_kwargs['evaluation_strategy'] = self.config.eval_strategy
        elif 'eval_strategy' in ta_sig:
            ta_kwargs['eval_strategy'] = self.config.eval_strategy
        # Use half precision during eval if supported to reduce memory
        if 'fp16_full_eval' in ta_sig:
            ta_kwargs['fp16_full_eval'] = (torch.cuda.is_available())
        # Filter unsupported keys for this Transformers version
        ta_kwargs = {k: v for k, v in ta_kwargs.items() if supported(k)}
        training_args = TrainingArguments(**ta_kwargs)
        
        # Create data collator
        data_collator = TrOCRDataCollator(self.processor, self.config.max_target_length)
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.config.early_stopping_patience)]
        )
        
        print(f"Starting training with the following configuration:")
        print(f"  Model: {self.config.model_name}")
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Eval samples: {len(eval_dataset)}")
        print(f"  Epochs: {self.config.num_train_epochs}")
        print(f"  Batch size: {self.config.per_device_train_batch_size}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Output dir: {self.config.output_dir}")
        print(f"  FP16: {training_args.fp16}")
        print(f"  Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        
        # Train the model
        train_result = trainer.train()
        
        # Save the final model
        print("Saving final model...")
        trainer.save_model()
        self.processor.save_pretrained(self.config.output_dir)
        
        # Get training history
        training_history = {
            'train_loss_history': [log['train_loss'] for log in trainer.state.log_history if 'train_loss' in log],
            'eval_loss_history': [log['eval_loss'] for log in trainer.state.log_history if 'eval_loss' in log],
            'eval_cer_history': [log['eval_cer'] for log in trainer.state.log_history if 'eval_cer' in log],
            'eval_wer_history': [log['eval_wer'] for log in trainer.state.log_history if 'eval_wer' in log],
            'eval_exact_match_rate_history': [log['eval_exact_match_rate'] for log in trainer.state.log_history if 'eval_exact_match_rate' in log],
        }
        
        print("Training completed!")
        # Guard in case histories are empty early on
        if training_history['eval_cer_history']:
            print(f"Best CER: {min(training_history['eval_cer_history']):.4f}")
        if training_history['eval_wer_history']:
            print(f"Best WER: {min(training_history['eval_wer_history']):.4f}")
        if training_history['eval_exact_match_rate_history']:
            print(f"Best Exact Match Rate: {max(training_history['eval_exact_match_rate_history']):.4f}")
        
        return {
            'train_result': train_result,
            'training_history': training_history,
            'model_path': self.config.output_dir
        }


def train_model(
    train_dataset: Dataset,
    eval_dataset: Dataset,
    config: Optional[TrOCRTrainingConfig] = None
) -> Dict[str, Any]:
    """
    Main function to train TrOCR model.
    
    Args:
        train_dataset: Training dataset 
        eval_dataset: Evaluation dataset
        config: Training configuration
        
    Returns:
        Training results dictionary
    """
    if config is None:
        config = TrOCRTrainingConfig()
    
    trainer = TrOCRTrainer(config)
    return trainer.train(train_dataset, eval_dataset)


if __name__ == "__main__":
    # Test training setup (without actual training)
    print("Testing training module setup...")
    
    config = TrOCRTrainingConfig(
        num_train_epochs=1,  # Just for testing
        per_device_train_batch_size=2,  # Small batch for testing
        eval_steps=50,
        save_steps=50
    )
    
    trainer = TrOCRTrainer(config)
    trainer.load_model_and_processor()
    
    print("Training module setup test complete!")
    print(f"Model loaded: {config.model_name}")
    print(f"Processor loaded successfully")
    print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")