import os
import json
import logging
import argparse
import codecs
from tqdm import tqdm
from mderank_indonesian import IndonesianMDERank, clean_indonesian_text

def setup_logging(log_path):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def load_semeval_data(data_path, labels_path):
    """Load dataset in SemEval2017 format"""
    data = {}
    labels = {}
    
    # Load documents
    for filename in os.listdir(data_path):
        if filename.endswith('.txt'):
            doc_id = filename[:-4]  # Remove .txt extension
            with codecs.open(os.path.join(data_path, filename), 'r', 'utf-8') as f:
                text = f.read().replace("%", "")
                data[doc_id] = text.lower()
                logging.info(f"Loaded document {doc_id} with length {len(text)}")
    
    # Load keyphrases
    for filename in os.listdir(labels_path):
        if filename.endswith('.key'):
            doc_id = filename[:-4]  # Remove .key extension
            with codecs.open(os.path.join(labels_path, filename), 'r', 'utf-8') as f:
                keyphrases = [line.strip() for line in f.readlines()]
                labels[doc_id] = keyphrases
                logging.info(f"Loaded keyphrases for {doc_id}: {keyphrases}")
    
    return data, labels

def evaluate_keyphrases(extracted_keyphrases, reference_keyphrases):
    """Evaluate extracted keyphrases against reference keyphrases"""
    extracted_set = set(kp.lower() for kp in extracted_keyphrases)
    reference_set = set(kp.lower() for kp in reference_keyphrases)
    
    correct = len(extracted_set.intersection(reference_set))
    precision = correct / len(extracted_set) if extracted_set else 0
    recall = correct / len(reference_set) if reference_set else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'correct': correct,
        'extracted': len(extracted_set),
        'reference': len(reference_set)
    }

def main():
    parser = argparse.ArgumentParser(description='MDERank for Indonesian Keyphrase Extraction')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Base directory containing dataset files')
    parser.add_argument('--model_name', type=str, default='indolem/indobert-base-uncased', help='Model name or path')
    parser.add_argument('--num_keyphrases', type=int, default=10, help='Number of keyphrases to extract')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for processing')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cuda/cpu)')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for results')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    log_path = os.path.join(args.output_dir, 'extraction.log')
    setup_logging(log_path)
    
    # Initialize model
    logging.info(f"Initializing model {args.model_name} on {args.device}")
    model = IndonesianMDERank(model_name=args.model_name, device=args.device)
    
    # Load dataset in SemEval format
    docs_path = os.path.join(args.dataset_dir, "docsutf8")
    keys_path = os.path.join(args.dataset_dir, "keys")
    logging.info(f"Loading dataset from {args.dataset_dir}")
    data, labels = load_semeval_data(docs_path, keys_path)
    logging.info(f"Loaded {len(data)} documents")
    
    # Process documents
    results = []
    total_metrics = {'precision': 0, 'recall': 0, 'f1': 0}
    
    for doc_id, text in tqdm(data.items(), desc="Processing documents"):
        try:
            logging.info(f"\nProcessing document {doc_id}")
            logging.info(f"Text length: {len(text)}")
            logging.info(f"First 100 characters: {text[:100]}")
            
            # Clean and preprocess text
            text = clean_indonesian_text(text)
            logging.info(f"Cleaned text length: {len(text)}")
            logging.info(f"First 100 characters after cleaning: {text[:100]}")
            
            # Extract keyphrases
            extracted_keyphrases = model.extract_keyphrases(text, num_keyphrases=args.num_keyphrases)
            logging.info(f"Extracted keyphrases: {extracted_keyphrases}")
            
            # Get reference keyphrases
            reference_keyphrases = labels.get(doc_id, [])
            logging.info(f"Reference keyphrases: {reference_keyphrases}")
            
            # Evaluate results
            metrics = evaluate_keyphrases(extracted_keyphrases, reference_keyphrases)
            
            # Update total metrics
            for key in total_metrics:
                total_metrics[key] += metrics[key]
            
            # Store results
            result = {
                'id': doc_id,
                'extracted_keyphrases': extracted_keyphrases,
                'reference_keyphrases': reference_keyphrases,
                'metrics': metrics
            }
            results.append(result)
            
            # Log progress
            logging.info(f"Document {doc_id}: F1={metrics['f1']:.4f}, "
                        f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}")
            
        except Exception as e:
            logging.error(f"Error processing document {doc_id}: {str(e)}")
            logging.error(f"Error type: {type(e)}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            continue  # Continue with next document even if this one fails
    
    # Calculate average metrics
    num_docs = len(data)
    avg_metrics = {k: v/num_docs for k, v in total_metrics.items()}
    
    # Log final results
    logging.info("\nFinal Results:")
    logging.info(f"Average Precision: {avg_metrics['precision']:.4f}")
    logging.info(f"Average Recall: {avg_metrics['recall']:.4f}")
    logging.info(f"Average F1: {avg_metrics['f1']:.4f}")
    
    # Save results
    output_path = os.path.join(args.output_dir, 'results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'results': results,
            'average_metrics': avg_metrics,
            'args': vars(args)
        }, f, ensure_ascii=False, indent=2)
    
    logging.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    main() 