import re
import time
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    BertForMaskedLM, 
    AutoTokenizer, 
    AutoModelForMaskedLM
)
import pandas as pd
import numpy as np
import logging
import argparse
import json
import os
import nltk
from nlp_id.postag import PosTag
from nlp_id.tokenizer import Tokenizer
from nlp_id.stopword import StopWord
from nlp_id.lemmatizer import Lemmatizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Download punkt for nlp-id
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set up logging directory
log_dir = 'results/log'
os.makedirs(log_dir, exist_ok=True)

# Configure main logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False  # Prevent propagation to root logger

# Create file handler for main logger
main_handler = logging.FileHandler(os.path.join(log_dir, 'main_log.txt'), mode='w', encoding='utf-8')
main_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(main_handler)

# Configure POS logger
pos_logger = logging.getLogger('pos_tagger')
pos_logger.setLevel(logging.INFO)
pos_logger.propagate = False  # Prevent propagation to root logger

# Create file handler for POS logger
pos_handler = logging.FileHandler(os.path.join(log_dir, 'pos_tagging_log.txt'), mode='w', encoding='utf-8')
pos_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
pos_logger.addHandler(pos_handler)

# Remove any existing handlers from root logger to prevent console output
root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Initialize Indonesian NLP tools
lemmatizer = Lemmatizer()
stopword_remover_factory = StopWordRemoverFactory()
stopword_remover = stopword_remover_factory.create_stop_word_remover()

# Initialize nlp-id stopwords
stopword = StopWord()
indonesian_stopwords = set(stopword.get_stopword())

# Additional Indonesian stopwords
additional_stopwords = {
    'yang', 'di', 'ke', 'dari', 'pada', 'dalam', 'untuk', 'dengan', 'dan', 'atau',
    'ini', 'itu', 'juga', 'sudah', 'saya', 'anda', 'dia', 'mereka', 'kita', 'akan',
    'bisa', 'ada', 'tidak', 'saat', 'oleh', 'setelah', 'tentang', 'seperti', 'ketika',
    'bagi', 'sampai', 'karena', 'jika', 'namun', 'serta', 'lain', 'sebuah', 'bahwa'
}
indonesian_stopwords.update(additional_stopwords)

def is_valid_pos_sequence(pos_tags):
    """Enhanced check for valid Indonesian noun phrases"""
    # Convert tuple format ('word', 'TAG') to just 'TAG' if needed
    pos_tags = [tag[1] if isinstance(tag, tuple) else tag for tag in pos_tags]
    
    # Single word - more permissive with tags
    if len(pos_tags) == 1:
        # Allow nouns, proper nouns, adjectives, and foreign words
        return pos_tags[0] in ['NN', 'NNP', 'JJ', 'FW']
    
    # Multi-word patterns - common Indonesian phrase patterns
    if len(pos_tags) >= 2:
        # Noun phrases
        if pos_tags[-1] in ['NN', 'NNP']:  # Last word should typically be a noun
            # Allow combinations leading to a noun
            valid_prefixes = ['JJ', 'NN', 'NNP', 'VB', 'FW']
            return all(tag in valid_prefixes for tag in pos_tags[:-1])
            
        # Adjective phrases (when last word is adjective)
        if pos_tags[-1] == 'JJ':
            # Allow noun/proper noun + adjective
            if all(tag in ['NN', 'NNP'] for tag in pos_tags[:-1]):
                return True
        
        # Technical terms (allow foreign words)
        if 'FW' in pos_tags:
            return True
    
    return False

def normalize_text(text):
    """Normalize text for consistent matching"""
    if isinstance(text, list):
        text = text[0] if text else ""
    # Convert to lowercase
    text = text.lower()
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text

def extract_noun_phrases(self, text, pos_tagger=None):
    """Extract noun phrases using POS patterns"""
    # Use class POS tagger if not provided
    pos_tagger = pos_tagger or self.pos_tagger
    
    # Get POS tags
    try:
        # Normalize and clean text before tokenization
        text = re.sub(r'\s+', ' ', text.strip())
        
        tokens = self.tokenizer_id.tokenize(text)
        
        # Safety check for empty tokens
        if not tokens:
            pos_logger.info("\nEmpty tokens after tokenization for text: " + text[:50] + "...")
            return []
            
        pos_tags = pos_tagger.get_pos_tag(' '.join(str(token) for token in tokens))
        
        # Ensure pos_tags and tokens have the same length
        if len(pos_tags) != len(tokens):
            pos_logger.info(f"\nMismatch between tokens ({len(tokens)}) and POS tags ({len(pos_tags)})")
            min_length = min(len(tokens), len(pos_tags))
            tokens = tokens[:min_length]
            pos_tags = pos_tags[:min_length]
        
        # Log POS tagging results
        pos_logger.info("\nPOS Tagging Results:")
        for token, tag in zip(tokens, pos_tags):
            pos_logger.info(f"{str(token):20} -> {tag}")
        
        # Initialize variables for phrase extraction
        phrases = []
        
        i = 0
        while i < len(tokens):
            # Valid starting tags: Noun, Proper Noun, or Foreign Word
            if i < len(pos_tags) and pos_tags[i][1] in ['NN', 'NNP', 'FW']:
                # Start with single word (with safety check)
                phrase_words = [str(tokens[i])]
                
                # Look ahead only one more word (with safety check)
                if (i + 1) < len(tokens) and (i + 1) < len(pos_tags) and pos_tags[i + 1][1] in ['NN', 'NNP', 'JJ', 'FW']:
                    phrase_words.append(str(tokens[i + 1]))
                    i += 2
                else:
                    i += 1
                    
                phrase = ' '.join(phrase_words)
                
                # Log accepted phrase
                pos_logger.info(f"\nAccepted phrase: {phrase}")
                pos_logger.info(f"POS sequence: {' '.join(tag[1] for tag in pos_tags[i-len(phrase_words):i])}")
                
                phrases.append(phrase)
            else:
                i += 1
                
        # Remove duplicates while preserving order
        unique_phrases = []
        seen = set()
        for phrase in phrases:
            if phrase not in seen:
                unique_phrases.append(phrase)
                seen.add(phrase)
        
        pos_logger.info(f"\nExtracted {len(unique_phrases)} unique phrases:")
        for phrase in unique_phrases:
            pos_logger.info(f"- {phrase}")
        
        return [[phrase, 0] for phrase in unique_phrases]
        
    except Exception as e:
        pos_logger.error(f"Error in extract_noun_phrases: {str(e)}")
        pos_logger.error(f"For text: {text[:100]}...")
        return []  # Return empty list on error

class InputTextObjIndonesian:
    """Represent the input text for Indonesian keyphrase extraction"""
    
    def __init__(self, text="", pos_tagger=None, tokenizer_id=None):
        """
        :param text: Input text
        :param pos_tagger: POS tagger instance
        :param tokenizer_id: Tokenizer instance
        """
        candidates = []
        logger.info(f"Processing text of length: {len(text)}")
        
        # Use provided NLP tools or create new ones
        self.pos_tagger = pos_tagger or PosTag()
        self.tokenizer_id = tokenizer_id or Tokenizer()
        
        # Split into sentences
        sentences = text.split('.')
        
        # Log the start of processing for this text
        pos_logger.info("\n\n=== New Text Processing ===")
        pos_logger.info(f"Original text: {text}\n")
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # Extract noun phrases
            noun_phrases = self.extract_noun_phrases(sentence)
            for phrase in noun_phrases:
                if isinstance(phrase, list):
                    candidates.append(phrase)
                else:
                    candidates.append([phrase, 0])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for candidate in candidates:
            phrase = candidate[0]
            if phrase not in seen:
                unique_candidates.append(candidate)
                seen.add(phrase)
        
        logger.info(f"Total candidates found: {len(unique_candidates)}")
        pos_logger.info(f"\nTotal unique candidates found: {len(unique_candidates)}")
        pos_logger.info("Final candidates:")
        for candidate in unique_candidates:
            pos_logger.info(f"- {candidate[0]}")
            
        self.keyphrase_candidate = unique_candidates
    
    def is_valid_keyphrase(self, phrase):
        """Check if a phrase is valid as a keyphrase"""
        if len(phrase) < 3:  # Too short
            return False
            
        words = phrase.lower().split()
        if len(words) > 4:  # Too long
            return False
            
        # Check if all words are stopwords
        if all(word in indonesian_stopwords for word in words):
            return False
            
        # Check if contains unwanted characters
        if re.search(r'[^a-zA-Z\s]', phrase):
            return False
            
        return True

    def extract_noun_phrases(self, text):
        """Extract noun phrases with strict 1-2 word limit"""
        tokens = self.tokenizer_id.tokenize(text)
        pos_tags = self.pos_tagger.get_pos_tag(' '.join(str(token) for token in tokens))
        
        # Ensure pos_tags and tokens have the same length
        if len(pos_tags) != len(tokens):
            logger.warning(f"Mismatch between tokens ({len(tokens)}) and POS tags ({len(pos_tags)})")
            min_length = min(len(tokens), len(pos_tags))
            tokens = tokens[:min_length]
            pos_tags = pos_tags[:min_length]
        
        # Log POS tagging results
        pos_logger.info("\nPOS Tagging Results:")
        for token, tag in zip(tokens, pos_tags):
            pos_logger.info(f"{str(token):20} -> {tag}")
        
        phrases = []
        i = 0
        while i < len(pos_tags):
            # Valid starting tags: Noun, Proper Noun, or Foreign Word
            if i < len(tokens) and pos_tags[i][1] in ['NN', 'NNP', 'FW']:
                # Start with single word
                phrase_words = [str(tokens[i])]
                
                # Look ahead only one more word
                if i + 1 < len(tokens) and i + 1 < len(pos_tags) and pos_tags[i + 1][1] in ['NN', 'NNP', 'JJ', 'FW']:
                    phrase_words.append(str(tokens[i + 1]))
                    i += 2
                else:
                    i += 1
                    
                phrase = ' '.join(phrase_words)
                
                # Log accepted phrase
                pos_logger.info(f"\nAccepted phrase: {phrase}")
                pos_logger.info(f"POS sequence: {' '.join(tag[1] for tag in pos_tags[i-len(phrase_words):i])}")
                
                phrases.append(phrase)
            else:
                i += 1
        
        # Remove duplicates while preserving order
        unique_phrases = []
        seen = set()
        for phrase in phrases:
            if phrase not in seen:
                unique_phrases.append(phrase)
                seen.add(phrase)
        
        pos_logger.info(f"\nExtracted {len(unique_phrases)} unique phrases:")
        for phrase in unique_phrases:
            pos_logger.info(f"- {phrase}")
        
        return [[phrase, 0] for phrase in unique_phrases]

    def filter_candidates(self, phrases):
        """Filter candidates with strict length limit"""
        filtered = []
        seen = set()
        
        for phrase, score in phrases:
            # Normalize for comparison
            words = phrase.split()
            
            # Skip if too long or already seen
            if len(words) > 2 or phrase in seen:
                continue
            
            # Keep technical terms and acronyms as is
            if len(words) == 1 and (any(c.isupper() for c in phrase) or 
                                   phrase in ['SEO', 'LMS']):
                filtered.append([phrase, score])
                seen.add(phrase)
                continue
            
            # For two-word phrases, check if it's a valid combination
            if len(words) == 2:
                filtered.append([phrase, score])
                seen.add(phrase)
        
        return filtered

def clean_indonesian_text(text):
    """Clean Indonesian text while preserving important linguistic features"""
    # Replace multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Preserve sentence boundaries
    text = re.sub(r'([.!?])\s*', r'\1\n', text)
    
    # Remove special characters but keep periods for sentence boundaries
    # text = re.sub(r'[^\w\s\.,!?\n]', '', text)
    
    # Remove standalone numbers but keep numbers within words
    # text = re.sub(r'\b\d+\b', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Split into sentences and clean each one
    sentences = [s.strip() for s in text.split('\n') if s.strip()]
    
    return '\n'.join(sentences)

class KPEDatasetIndonesian(Dataset):
    """Dataset for Indonesian keyphrase extraction"""
    
    def __init__(self, docs_pairs):
        self.docs_pairs = docs_pairs
        self.total_examples = len(self.docs_pairs)
    
    def __len__(self):
        return self.total_examples
    
    def __getitem__(self, idx):
        doc_pair = self.docs_pairs[idx]
        ori_example = doc_pair[0]
        masked_example = doc_pair[1]
        doc_id = doc_pair[2]
        return [ori_example, masked_example, doc_id]

def mean_pooling(model_output, attention_mask):
    """Mean pooling of transformer outputs"""
    hidden_states = model_output.hidden_states[-1]  # Use last layer
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    return torch.sum(hidden_states * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def max_pooling(model_output, attention_mask):
    """Max pooling of transformer outputs"""
    hidden_states = model_output.hidden_states[-1]  # Use last layer
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    masked_states = hidden_states * input_mask_expanded
    # Replace padded values with large negative values to ensure they don't get selected
    masked_states[masked_states == 0] = -1e9
    return torch.max(masked_states, dim=1)[0]

class IndonesianMDERank:
    """MDERank implementation for Indonesian"""
    _instance = None
    _is_initialized = False
    
    def __new__(cls, model_name="indolem/indobert-base-uncased", device=None):
        if cls._instance is None:
            cls._instance = super(IndonesianMDERank, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, model_name="indolem/indobert-base-uncased", device=None):
        """Initialize the model and tokenizer"""
        # Skip initialization if already done
        if IndonesianMDERank._is_initialized:
            return
            
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        logger.info("Loading model and tokenizer...")
        
        # Load model and tokenizer only once during initialization
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        self.model.eval()  # Set model to evaluation mode
        
        # Initialize Indonesian NLP tools
        self.pos_tagger = PosTag()
        self.tokenizer_id = Tokenizer()
        self.lemmatizer = lemmatizer
        self.stopword_remover = stopword_remover
        logger.info("Model and tools initialized successfully")
        
        IndonesianMDERank._is_initialized = True
    
    def preprocess_text(self, text):
        """Preprocess Indonesian text"""
        # Basic cleaning
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        
        # Split into words
        words = text.split()
        processed_words = []
        
        for word in words:
            # Skip stopwords
            if word in indonesian_stopwords:
                continue
                
            # Apply lemmatization for regular words
            if len(word) > 3 and not re.search(r'[0-9\W]', word):
                lemmatized = self.lemmatizer.lemmatize(word)
                processed_words.append(lemmatized)
            else:
                processed_words.append(word)
        
        # Rejoin words
        text = ' '.join(processed_words)
        text = normalize_text(text)
        return text
    
    def generate_masked_doc(self, doc, candidates, mask_token="[MASK]"):
        """
        Generate masked versions of the document for each candidate.
        Uses class-level NLP tools to avoid reinitializing.
        """
        # Normalize document text
        doc = normalize_text(doc)
        
        # Handle both string and list inputs
        if isinstance(candidates, str):
            candidates = [candidates]
        elif isinstance(candidates, list) and len(candidates) > 0 and isinstance(candidates[0], list):
            # Extract phrases from [phrase, score] format
            candidates = [c[0] for c in candidates]
        
        masked_docs = []
        for candidate in candidates:
            # Normalize candidate
            candidate = normalize_text(candidate)
            
            # Find all occurrences of the candidate in the document
            start_idx = 0
            occurrences = []
            
            while True:
                start_idx = doc.find(candidate, start_idx)
                if start_idx == -1:
                    break
                
                # Verify this is a complete word/phrase match
                end_idx = start_idx + len(candidate)
                is_word_boundary_start = start_idx == 0 or not doc[start_idx - 1].isalnum()
                is_word_boundary_end = end_idx == len(doc) or not doc[end_idx].isalnum()
                
                if is_word_boundary_start and is_word_boundary_end:
                    occurrences.append(start_idx)
                start_idx = end_idx
            
            if not occurrences:
                logger.debug(f"Candidate not found in text: {candidate}")
                continue
            
            for start_idx in occurrences:
                # Get POS tags for the candidate using class-level pos tagger
                candidate_tokens = self.tokenizer_id.tokenize(candidate)
                candidate_pos = self.pos_tagger.get_pos_tag(' '.join(str(token) for token in candidate_tokens))
                
                # Create masked text
                masked_text = doc[:start_idx] + mask_token + doc[start_idx + len(candidate):]
                
                # Store the masked text and candidate along with its POS tags
                masked_docs.append({
                    'masked_text': masked_text,
                    'candidate': candidate,
                    'pos_tags': candidate_pos
                })
        
        return masked_docs
    
    def extract_keyphrases(self, text, num_keyphrases=10):
        """Extract keyphrases using masked language modeling"""
        # Log the start of extraction
        pos_logger.info("\n\n=== Starting Keyphrase Extraction ===")
        pos_logger.info(f"Input text length: {len(text)}")
        
        # Get candidates
        text_obj = InputTextObjIndonesian(text, self.pos_tagger, self.tokenizer_id)
        candidates = text_obj.keyphrase_candidate
        
        # Filter candidates to ensure 1-2 word phrases only
        candidates = text_obj.filter_candidates(candidates)
        
        if not candidates:
            logger.warning("No candidates found in the text")
            pos_logger.info("No candidates found in the text")
            return []
        
        # Debug logging
        logger.info(f"Number of candidates: {len(candidates)}")
        if len(candidates) > 0:
            logger.info(f"First few candidates: {candidates[:5]}")
        
        # Generate masked versions and prepare data
        docs_pairs = []
        for idx, candidate in enumerate(candidates):
            try:
                # Get the candidate phrase
                candidate_phrase = candidate[0]
                
                # Skip empty or invalid candidates
                if not candidate_phrase or not isinstance(candidate_phrase, str):
                    continue
                
                # Original document encoding
                ori_encoding = self.tokenizer(
                    text, 
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True
                )
                
                # Generate masked versions
                masked_docs = self.generate_masked_doc(text, candidate_phrase)
                if not masked_docs:  # Skip if candidate not found in text
                    continue
                
                # Use first masked version
                masked_encoding = self.tokenizer(
                    masked_docs[0]['masked_text'],
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True
                )
            
                masked_encoding["candidate"] = candidate_phrase
                docs_pairs.append([ori_encoding, masked_encoding, idx])
                
            except Exception as e:
                logger.error(f"Error processing candidate {candidate_phrase}: {str(e)}")
                continue
        
        if not docs_pairs:
            logger.warning("No valid candidates found after processing")
            return []
        
        # Create dataset and dataloader
        dataset = KPEDatasetIndonesian(docs_pairs)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # Get document embeddings and calculate similarities
        cos_similarities = []
        candidates_list = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Processing candidates"):
                try:
                    ori_doc, masked_doc, doc_id = batch
                    
                    # Move tensors to device
                    ori_input_ids = ori_doc["input_ids"].squeeze(0).to(self.device)
                    ori_attention_mask = ori_doc["attention_mask"].squeeze(0).to(self.device)
                    
                    masked_input_ids = masked_doc["input_ids"].squeeze(0).to(self.device)
                    masked_attention_mask = masked_doc["attention_mask"].squeeze(0).to(self.device)
                    
                    # Get embeddings
                    ori_outputs = self.model(
                        input_ids=ori_input_ids,
                        attention_mask=ori_attention_mask,
                        output_hidden_states=True
                    )
                    
                    masked_outputs = self.model(
                        input_ids=masked_input_ids,
                        attention_mask=masked_attention_mask,
                        output_hidden_states=True
                    )
                    
                    # Use mean pooling only
                    ori_embed = mean_pooling(ori_outputs, ori_attention_mask)
                    masked_embed = mean_pooling(masked_outputs, masked_attention_mask)
                    
                    # Calculate similarity (lower is better)
                    similarity = torch.cosine_similarity(ori_embed, masked_embed, dim=1).cpu().item()
                    
                    cos_similarities.append(similarity)
                    candidates_list.append(masked_doc["candidate"][0])
                    
                except Exception as e:
                    logger.error(f"Error in processing batch: {str(e)}")
                    continue
        
        # Sort candidates by score (ascending) before logging
        sorted_candidates = sorted(zip(candidates_list, cos_similarities), key=lambda x: x[1])
        
        # Log all candidates with their scores (sorted)
        logger.info("\n=== All Candidate Keyphrases with Similarity Scores (Sorted) ===")
        pos_logger.info("\n=== All Candidate Keyphrases with Similarity Scores (Sorted) ===")
        logger.info("Note: Lower scores indicate more important keyphrases")
        pos_logger.info("Note: Lower scores indicate more important keyphrases")
        
        for candidate, score in sorted_candidates:
            logger.info(f"Candidate: {candidate:<40} Score: {score:.4f}")
            pos_logger.info(f"Candidate: {candidate:<40} Score: {score:.4f}")
        
        # Rank candidates by similarity (lower is better)
        ranked_pairs = sorted(zip(candidates_list, cos_similarities), 
                             key=lambda x: x[1])
        
        # Remove duplicates while preserving order
        seen = set()
        ranked_keyphrases = []
        
        # Log all candidates with their scores
        logger.info("\n=== Candidate Keyphrases with Similarity Scores ===")
        pos_logger.info("\n=== Candidate Keyphrases with Similarity Scores ===")
        logger.info("Note: Lower scores indicate more important keyphrases")
        pos_logger.info("Note: Lower scores indicate more important keyphrases")
        
        for candidate, similarity in ranked_pairs:
            if candidate not in seen:
                ranked_keyphrases.append({
                    'phrase': candidate,
                    'score': similarity
                })
                seen.add(candidate)
                # Log each candidate and its score
                logger.info(f"Candidate: {candidate:<40} Score: {similarity:.4f}")
                pos_logger.info(f"Candidate: {candidate:<40} Score: {similarity:.4f}")
        
        # Get top keyphrases
        top_keyphrases = [item['phrase'] for item in ranked_keyphrases[:num_keyphrases]]
        
        # Log final selected keyphrases
        logger.info("\n=== Final Selected Keyphrases ===")
        pos_logger.info("\n=== Final Selected Keyphrases ===")
        for i, keyphrase in enumerate(ranked_keyphrases[:num_keyphrases], 1):
            logger.info(f"{i}. {keyphrase['phrase']:<40} Score: {keyphrase['score']:.4f}")
            pos_logger.info(f"{i}. {keyphrase['phrase']:<40} Score: {keyphrase['score']:.4f}")
        
        return top_keyphrases

def is_valid_indonesian_keyphrase(phrase, min_length=2):
    """Check if a phrase is a valid Indonesian keyphrase"""
    words = phrase.split()
    filtered_words = [w for w in words if w.lower() not in indonesian_stopwords]
    return len(filtered_words) >= min_length 

def load_test_data(filepath):
    """Load test data from JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        return []

def evaluate_keyphrases(references, predictions):
    """
    Evaluate keyphrase extraction performance
    
    Args:
        references (list): List of reference keyphrases
        predictions (list): List of predicted keyphrases
        
    Returns:
        dict: Dictionary containing precision, recall, F1 score, and other metrics
    """
    # Convert to sets
    reference_set = set(references)
    prediction_set = set(predictions)
    
    # Calculate standard metrics
    true_positives = len(reference_set.intersection(prediction_set))
    
    precision = true_positives / len(prediction_set) if prediction_set else 0.0
    recall = true_positives / len(reference_set) if reference_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Calculate Mean Average Precision (MAP)
    ap = 0.0
    correct_count = 0
    
    for i, pred in enumerate(predictions):
        if pred in reference_set:
            correct_count += 1
            ap += correct_count / (i + 1)
    
    map_score = ap / len(reference_set) if reference_set else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'map': map_score
    }

def main():
    parser = argparse.ArgumentParser(description='Indonesian Keyphrase Extraction using MDERank')
    parser.add_argument('--input', type=str, required=True, help='Path to test data JSON file')
    parser.add_argument('--output', type=str, default='results.json', help='Path to output JSON file')
    parser.add_argument('--model', type=str, default='indobenchmark/indobert-base-p1', help='Pretrained model name')
    parser.add_argument('--num-keyphrases', type=int, default=5, help='Number of keyphrases to extract')
    
    args = parser.parse_args()
    
    # Load test data
    test_data = load_test_data(args.input)
    if not test_data:
        logger.error("No test data found or error loading test data.")
        return
    
    # Initialize MDERank model once
    mderank = IndonesianMDERank(model_name=args.model)
    
    # Process each document
    results = []
    all_metrics = []
    
    for i, doc in enumerate(test_data):
        logger.info(f"\nProcessing document {i+1}/{len(test_data)}")
        
        try:
            # Extract keyphrases using the same model instance
            keyphrases = mderank.extract_keyphrases(doc['text'], num_keyphrases=args.num_keyphrases)
            
            # Get reference keyphrases
            references = doc['keyphrases']
            
            # Evaluate
            metrics = evaluate_keyphrases(references, keyphrases)
            all_metrics.append(metrics)
            
            # Store results
            results.append({
                'id': i,
                'text': doc['text'][:200] + '...' if len(doc['text']) > 200 else doc['text'],
                'reference_keyphrases': references,
                'predicted_keyphrases': keyphrases,
                'metrics': metrics
            })
            
            logger.info(f"Document: {doc['text'][:100]}...")
            logger.info(f"Reference keyphrases: {references}")
            logger.info(f"Predicted keyphrases: {keyphrases}")
            logger.info(f"Metrics: {metrics}")
            
        except Exception as e:
            logger.error(f"Error processing document {i}: {str(e)}")
    
    # Calculate average metrics
    avg_metrics = {
        'precision': sum(m['precision'] for m in all_metrics) / len(all_metrics) if all_metrics else 0,
        'recall': sum(m['recall'] for m in all_metrics) / len(all_metrics) if all_metrics else 0,
        'f1': sum(m['f1'] for m in all_metrics) / len(all_metrics) if all_metrics else 0,
        'map': sum(m['map'] for m in all_metrics) / len(all_metrics) if all_metrics else 0
    }
    
    # Save results
    output = {
        'results': results,
        'average_metrics': avg_metrics
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\nAverage metrics across {len(results)} documents:")
    logger.info(f"Precision: {avg_metrics['precision']:.3f}")
    logger.info(f"Recall: {avg_metrics['recall']:.3f}")
    logger.info(f"F1: {avg_metrics['f1']:.3f}")
    logger.info(f"MAP: {avg_metrics['map']:.3f}")
    logger.info(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()