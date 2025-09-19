"""
MediPhi-based correction module for OCR post-processing.
"""

import re
import json
import pickle
import os
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import Counter, defaultdict
from dataclasses import dataclass
import pandas as pd
from dataclasses import asdict, is_dataclass

# Optional deps for robust JSON serialization
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore
try:
    import torch as _torch  # type: ignore
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    _torch = None  # type: ignore
    _TORCH_AVAILABLE = False

def _to_native(obj):
    """Recursively convert objects to JSON-serializable Python types.
    - numpy scalars -> Python scalars
    - numpy arrays -> lists
    - torch tensors -> lists
    - dataclasses -> dicts
    """
    # numpy types
    if np is not None:
        if isinstance(obj, getattr(np, 'generic', ())):  # numpy scalar
            try:
                return obj.item()
            except Exception:
                pass
        if isinstance(obj, getattr(np, 'ndarray', ())):
            try:
                return obj.tolist()
            except Exception:
                pass
    # torch tensors
    if _TORCH_AVAILABLE and isinstance(obj, getattr(_torch, 'Tensor', ())):
        try:
            return obj.detach().cpu().tolist()
        except Exception:
            pass
    # dataclasses
    if is_dataclass(obj):
        try:
            return {k: _to_native(v) for k, v in asdict(obj).items()}
        except Exception:
            pass
    # containers
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_native(v) for v in obj]
    return obj

try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    try:
        import Levenshtein
        RAPIDFUZZ_AVAILABLE = False
        print("Warning: rapidfuzz not available, using Levenshtein")
    except ImportError:
        print("Warning: Neither rapidfuzz nor Levenshtein available. Install one for fuzzy matching.")
        RAPIDFUZZ_AVAILABLE = None

try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: transformers not available for MediPhi")
    TRANSFORMERS_AVAILABLE = False


@dataclass
class MedicalEntity:
    """Medical entity extracted by NER."""
    text: str
    label: str
    start: int
    end: int
    confidence: float = 0.0


@dataclass
class CorrectionResult:
    """Result of OCR correction."""
    original_text: str
    corrected_text: str
    entities: List[MedicalEntity]
    corrections_made: List[Dict[str, Any]]
    confidence_score: float


class MedicalVocabularyBuilder:
    """Builds medical vocabulary from training data for fuzzy matching."""
    
    def __init__(self):
        self.drug_vocab: Set[str] = set()
        self.dosage_vocab: Set[str] = set()
        self.frequency_vocab: Set[str] = set()
        self.form_vocab: Set[str] = set()
        self.general_vocab: Set[str] = set()
        
        # Common medical patterns
        self.drug_patterns = [
            r'\b[A-Z][a-z]+(?:ol|in|ine|ide|ate|ium)\b',  # Common drug suffixes
            r'\b[A-Z][a-z]*[A-Z][a-z]*\b',  # CamelCase drug names
        ]
        
        self.dosage_patterns = [
            r'\d+(?:\.\d+)?\s*(?:mg|g|ml|mcg|units?)\b',
            r'\d+(?:\.\d+)?\s*(?:milligrams?|grams?|milliliters?|micrograms?)\b'
        ]
        
        self.frequency_patterns = [
            r'\b(?:once|twice|thrice|three times?|four times?)\s+(?:daily|a day|per day)\b',
            r'\b(?:every|each)\s+\d+\s+(?:hours?|hrs?)\b',
            r'\b(?:morning|evening|night|bedtime)\b',
            r'\b(?:with|after|before)\s+(?:meals?|food)\b'
        ]
        
        self.form_patterns = [
            r'\b(?:tablet|capsule|syrup|injection|cream|ointment|drops?)s?\b',
            r'\b(?:pill|medicine|medication|drug)s?\b'
        ]
    
    def build_from_texts(self, texts: List[str]) -> None:
        """Build vocabulary from a list of medical texts."""
        print("Building medical vocabulary from training texts...")
        
        for text in texts:
            text_lower = text.lower()
            
            # Extract drug names (capitalized words, drug patterns)
            for pattern in self.drug_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                self.drug_vocab.update([m.lower() for m in matches])
            
            # Extract dosages
            for pattern in self.dosage_patterns:
                matches = re.findall(pattern, text_lower)
                self.dosage_vocab.update(matches)
            
            # Extract frequencies
            for pattern in self.frequency_patterns:
                matches = re.findall(pattern, text_lower)
                self.frequency_vocab.update(matches)
            
            # Extract forms
            for pattern in self.form_patterns:
                matches = re.findall(pattern, text_lower)
                self.form_vocab.update(matches)
            
            # General vocabulary (all words)
            words = re.findall(r'\b[a-zA-Z]{2,}\b', text_lower)
            self.general_vocab.update(words)
        
        print(f"Vocabulary built: {len(self.drug_vocab)} drugs, {len(self.dosage_vocab)} dosages, "
              f"{len(self.frequency_vocab)} frequencies, {len(self.form_vocab)} forms, "
              f"{len(self.general_vocab)} general terms")
    
    def save_vocabulary(self, filepath: str) -> None:
        """Save vocabulary to file."""
        vocab_data = {
            'drugs': list(self.drug_vocab),
            'dosages': list(self.dosage_vocab),
            'frequencies': list(self.frequency_vocab),
            'forms': list(self.form_vocab),
            'general': list(self.general_vocab)
        }
        
        with open(filepath, 'w') as f:
            json.dump(vocab_data, f, indent=2)
        print(f"Vocabulary saved to {filepath}")
    
    def load_vocabulary(self, filepath: str) -> None:
        """Load vocabulary from file."""
        with open(filepath, 'r') as f:
            vocab_data = json.load(f)
        
        self.drug_vocab = set(vocab_data.get('drugs', []))
        self.dosage_vocab = set(vocab_data.get('dosages', []))
        self.frequency_vocab = set(vocab_data.get('frequencies', []))
        self.form_vocab = set(vocab_data.get('forms', []))
        self.general_vocab = set(vocab_data.get('general', []))
        
        print(f"Vocabulary loaded from {filepath}")


class MediPhiNER:
    """NER component using MediPhi or similar medical NER model."""
    
    def __init__(self, model_name: str = "microsoft/MediPhi"):
        self.model_name = model_name
        self.ner_pipeline = None
        
        # Load model if transformers is available
        if TRANSFORMERS_AVAILABLE:
            try:
                print(f"Loading NER model: {model_name}")
                self.ner_pipeline = pipeline(
                    "ner", 
                    model=model_name, 
                    tokenizer=model_name,
                    aggregation_strategy="simple"
                )
                print("NER model loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load {model_name}: {e}")
                print("Using rule-based NER fallback")
                self.ner_pipeline = None
        else:
            print("Transformers not available, using rule-based NER")
    
    def extract_entities(self, text: str) -> List[MedicalEntity]:
        """Extract medical entities from text."""
        if self.ner_pipeline:
            return self._extract_with_model(text)
        else:
            return self._extract_with_rules(text)
    
    def _extract_with_model(self, text: str) -> List[MedicalEntity]:
        """Extract entities using the NER model."""
        try:
            results = self.ner_pipeline(text)
            entities = []
            
            for result in results:
                entity = MedicalEntity(
                    text=result['word'],
                    label=result['entity_group'],
                    start=result['start'],
                    end=result['end'],
                    confidence=result['score']
                )
                entities.append(entity)
            
            return entities
            
        except Exception as e:
            print(f"NER model failed: {e}. Falling back to rules.")
            return self._extract_with_rules(text)
    
    def _extract_with_rules(self, text: str) -> List[MedicalEntity]:
        """Extract entities using rule-based approach."""
        entities = []
        
        # Drug patterns
        drug_patterns = [
            r'\b[A-Z][a-z]+(?:ol|in|ine|ide|ate|ium)\b',
            r'\b(?:aspirin|paracetamol|ibuprofen|amoxicillin|metformin|insulin)\b'
        ]
        
        for pattern in drug_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(MedicalEntity(
                    text=match.group(),
                    label="DRUG",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.8
                ))
        
        # Dosage patterns
        dosage_pattern = r'\d+(?:\.\d+)?\s*(?:mg|g|ml|mcg|units?)\b'
        for match in re.finditer(dosage_pattern, text, re.IGNORECASE):
            entities.append(MedicalEntity(
                text=match.group(),
                label="DOSAGE",
                start=match.start(),
                end=match.end(),
                confidence=0.9
            ))
        
        # Frequency patterns
        freq_patterns = [
            r'\b(?:once|twice|thrice|three times?|four times?)\s+(?:daily|a day|per day)\b',
            r'\b(?:every|each)\s+\d+\s+(?:hours?|hrs?)\b'
        ]
        
        for pattern in freq_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(MedicalEntity(
                    text=match.group(),
                    label="FREQUENCY",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.8
                ))
        
        # Form patterns
        form_pattern = r'\b(?:tablet|capsule|syrup|injection|cream|ointment|drops?)s?\b'
        for match in re.finditer(form_pattern, text, re.IGNORECASE):
            entities.append(MedicalEntity(
                text=match.group(),
                label="FORM",
                start=match.start(),
                end=match.end(),
                confidence=0.8
            ))
        
        return entities


class FuzzyCorrector:
    """Fuzzy string matching corrector."""
    
    def __init__(self, vocabulary: MedicalVocabularyBuilder):
        self.vocabulary = vocabulary
        self.threshold = 80  # Similarity threshold for corrections
    
    def find_best_match(self, word: str, vocab_set: Set[str], threshold: int = None) -> Optional[str]:
        """Find best fuzzy match for a word in vocabulary."""
        if not vocab_set or not word:
            return None
        
        threshold = threshold or self.threshold
        
        if RAPIDFUZZ_AVAILABLE:
            match = process.extractOne(word.lower(), vocab_set, scorer=fuzz.ratio)
            if match and match[1] >= threshold:
                return match[0]
        elif RAPIDFUZZ_AVAILABLE is False:  # Levenshtein available
            best_match = None
            best_score = 0
            
            for vocab_word in vocab_set:
                score = 100 - (Levenshtein.distance(word.lower(), vocab_word) / max(len(word), len(vocab_word)) * 100)
                if score >= threshold and score > best_score:
                    best_match = vocab_word
                    best_score = score
            
            return best_match
        
        return None
    
    def correct_word(self, word: str, entity_type: str = None) -> Optional[str]:
        """Correct a single word using appropriate vocabulary."""
        if entity_type == "DRUG":
            return self.find_best_match(word, self.vocabulary.drug_vocab)
        elif entity_type == "DOSAGE":
            return self.find_best_match(word, self.vocabulary.dosage_vocab)
        elif entity_type == "FREQUENCY":
            return self.find_best_match(word, self.vocabulary.frequency_vocab)
        elif entity_type == "FORM":
            return self.find_best_match(word, self.vocabulary.form_vocab)
        else:
            # Try general vocabulary
            return self.find_best_match(word, self.vocabulary.general_vocab)


class MediPhiCorrector:
    """Main correction engine combining NER and fuzzy matching."""
    
    def __init__(
        self, 
        vocab_builder: MedicalVocabularyBuilder = None,
        ner_model: str = "microsoft/MediPhi"
    ):
        self.vocab_builder = vocab_builder or MedicalVocabularyBuilder()
        self.ner = MediPhiNER(ner_model)
        self.fuzzy_corrector = FuzzyCorrector(self.vocab_builder)
        
        # Common OCR error patterns
        self.ocr_patterns = {
            'rn': 'm',
            'cl': 'd',
            'ii': 'll',
            '0': 'o',
            '1': 'l',
            '5': 's',
            '8': 'B',
        }
    
    def build_vocabulary_from_training_data(self, training_texts: List[str]) -> None:
        """Build vocabulary from training data."""
        self.vocab_builder.build_from_texts(training_texts)
        self.fuzzy_corrector = FuzzyCorrector(self.vocab_builder)
    
    def apply_ocr_corrections(self, text: str) -> str:
        """Apply common OCR error corrections."""
        corrected = text
        for error, correction in self.ocr_patterns.items():
            corrected = re.sub(error, correction, corrected)
        return corrected
    
    def correct_text(self, text: str) -> CorrectionResult:
        """
        Correct OCR text using NER and fuzzy matching.
        
        Args:
            text: OCR text to correct
            
        Returns:
            CorrectionResult with corrected text and metadata
        """
        original_text = text
        corrections_made = []
        
        # Step 1: Apply basic OCR pattern corrections
        text = self.apply_ocr_corrections(text)
        if text != original_text:
            corrections_made.append({
                'type': 'ocr_pattern',
                'original': original_text,
                'corrected': text
            })
        
        # Step 2: Extract entities
        entities = self.ner.extract_entities(text)
        
        # Step 3: Apply fuzzy corrections to entities and general words
        corrected_text = text
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text)
        
        for word in words:
            # Find if this word is part of an entity
            entity_type = None
            for entity in entities:
                if word.lower() in entity.text.lower():
                    entity_type = entity.label
                    break
            
            # Try to correct the word
            correction = self.fuzzy_corrector.correct_word(word, entity_type)
            if correction and correction != word.lower():
                # Replace in text (case-insensitive)
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                corrected_text = pattern.sub(correction, corrected_text, count=1)
                
                corrections_made.append({
                    'type': 'fuzzy_match',
                    'entity_type': entity_type,
                    'original': word,
                    'corrected': correction
                })
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(entities, corrections_made)
        
        return CorrectionResult(
            original_text=original_text,
            corrected_text=corrected_text,
            entities=entities,
            corrections_made=corrections_made,
            confidence_score=confidence_score
        )
    
    def _calculate_confidence(self, entities: List[MedicalEntity], corrections: List[Dict]) -> float:
        """Calculate confidence score for the correction."""
        if not entities:
            return 0.5
        
        # Base confidence from NER entities
        entity_confidence = sum(entity.confidence for entity in entities) / len(entities)
        
        # Penalty for many corrections (might indicate poor OCR)
        correction_penalty = min(0.2 * len(corrections), 0.5)
        
        # Bonus for medical entities found
        entity_bonus = min(0.1 * len(entities), 0.3)
        
        confidence = entity_confidence + entity_bonus - correction_penalty
        return max(0.1, min(1.0, confidence))


def correct_with_mediphi(
    ocr_texts: List[str],
    training_texts: List[str] = None,
    vocab_file: str = None,
    ner_model: str = "microsoft/MediPhi"
) -> List[CorrectionResult]:
    """
    Main function to correct OCR texts using MediPhi.
    
    Args:
        ocr_texts: List of OCR texts to correct
        training_texts: List of training texts to build vocabulary
        vocab_file: Path to saved vocabulary file
        ner_model: NER model name
        
    Returns:
        List of correction results
    """
    # Initialize corrector
    vocab_builder = MedicalVocabularyBuilder()
    
    # Build or load vocabulary
    if vocab_file and os.path.exists(vocab_file):
        vocab_builder.load_vocabulary(vocab_file)
    elif training_texts:
        vocab_builder.build_from_texts(training_texts)
        if vocab_file:
            vocab_builder.save_vocabulary(vocab_file)
    
    corrector = MediPhiCorrector(vocab_builder, ner_model)
    
    # Correct all texts
    results = []
    print(f"Correcting {len(ocr_texts)} texts...")
    
    for i, text in enumerate(ocr_texts):
        if i % 100 == 0:
            print(f"Progress: {i}/{len(ocr_texts)}")
        
        result = corrector.correct_text(text)
        results.append(result)
    
    print("Correction completed!")
    return results


def save_correction_results(
    results: List[CorrectionResult],
    output_file: str
) -> None:
    """Save correction results to CSV file."""
    data = []
    for i, result in enumerate(results):
        data.append({
            'sample_id': i,
            'original_text': result.original_text,
            'corrected_text': result.corrected_text,
            'num_entities': len(result.entities),
            'num_corrections': len(result.corrections_made),
            'confidence_score': result.confidence_score,
            'entities': json.dumps(_to_native([{
                'text': e.text,
                'label': e.label,
                'confidence': e.confidence
            } for e in result.entities])),
            'corrections': json.dumps(_to_native(result.corrections_made))
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Correction results saved to {output_file}")


if __name__ == "__main__":
    # Test the correction module
    print("Testing MediPhi correction module...")
    
    # Sample OCR texts with errors
    sample_ocr_texts = [
        "Take aspnn 100mg twice daily",  # asprin -> aspirin
        "Paracetamol 500mg thre times a day",  # thre -> three
        "Apply cream to affeted area",  # affeted -> affected
        "Insulin 10 units before meals"
    ]
    
    # Sample training texts for vocabulary
    sample_training_texts = [
        "Take aspirin 100mg twice daily",
        "Paracetamol 500mg three times a day",
        "Apply cream to affected area",
        "Insulin 10 units before meals",
        "Ibuprofen 400mg as needed for pain",
        "Amoxicillin capsules 250mg three times daily"
    ]
    
    # Test correction
    try:
        results = correct_with_mediphi(
            ocr_texts=sample_ocr_texts,
            training_texts=sample_training_texts,
            vocab_file="test_vocab.json"
        )
        
        print("\nCorrection Results:")
        for i, result in enumerate(results):
            print(f"\nSample {i+1}:")
            print(f"Original: {result.original_text}")
            print(f"Corrected: {result.corrected_text}")
            print(f"Entities: {len(result.entities)}")
            print(f"Corrections: {len(result.corrections_made)}")
            print(f"Confidence: {result.confidence_score:.3f}")
        
        # Save results
        save_correction_results(results, "test_corrections.csv")
        
    except Exception as e:
        print(f"Test failed: {e}")
        print("Note: This is expected if MediPhi model is not available")