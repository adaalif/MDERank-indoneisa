# MDERank Indonesia

Library for MDERank model for Automatic Keyphrase Extraction in Indonesian language.

This is an adaptation of the original MDERank model, specifically optimized for Indonesian language processing. It uses Indonesian-specific NLP tools and BERT models to achieve better keyphrase extraction results for Indonesian text.

Original Paper: MDERank: A Masked Document Embedding Rank Approach for Unsupervised Keyphrase Extraction
Original Repo: https://github.com/LinhanZ/mderank

## Features

- Optimized for Indonesian language processing
- Uses Indonesian-specific NLP tools:
  - nlp-id for POS tagging and tokenization
  - Sastrawi for stemming and stopword removal
- Supports Indonesian BERT models:
  - indolem/indobert-base-uncased
  - indobenchmark/indobert-base-p1
- Enhanced POS tagging and phrase extraction for Indonesian
- Custom stopword handling for Indonesian
- Improved noun phrase detection for Indonesian grammar

## Installation

This project has been developed under Python 3.9.6

1. Install required libraries:
```bash
pip install -r requirements.txt
pip install -r requirements-pytorch.txt
```


## Usage

### Evaluation Mode
To evaluate on a dataset:
```bash
bash eval_indonesian.sh
```

The script expects data in the following structure:
- `data/datasetname/docsutf8/` - Contains input documents
- `data/datasetname/keys/` - Contains reference keyphrases

### Execution Mode
To extract keyphrases from a folder of documents:
```bash
bash run.sh
```



## Configuration

Key parameters in the Indonesian version:
- `--model_name_or_path`: Choose between Indonesian BERT models
  - `indolem/indobert-base-uncased`
  - `indobenchmark/indobert-base-p1`
- `--lang id`: Specifies Indonesian language processing
- `--k_value`: Number of keyphrases to extract (default: 15)
- `--doc_embed_mode`: Embedding mode (mean/max/cls)
- `--batch_size`: Batch size for processing (default: 1)
- `--layer_num`: Layer number for embeddings (default: -1 for last layer)

## Data Format

Input documents should be in UTF-8 format. The system supports:
- Plain text files
- JSON files with 'text' and 'keyphrases' fields
- Multiple document formats in a directory

## Performance

The Indonesian version has been optimized for:
- Better handling of Indonesian compound words
- Improved detection of Indonesian noun phrases
- More accurate stopword filtering for Indonesian
- Better handling of Indonesian prefixes and suffixes

## Acknowledgments

This implementation was developed as part of research in Indonesian language processing and keyphrase extraction.

## Citation

If you use this implementation, please cite:

```bibtext
@inproceedings{Calleja2024,
  author    = {Pablo Calleja and Patricia Martín-Chozas and Elena Montiel-Ponsoda},
  title     = {Benchmark for Automatic Keyword Extraction in Spanish: Datasets and Methods},
  booktitle = {Poster Proceedings of the 40th Annual Conference of the Spanish Association for Natural Language Processing 2024 (SEPLN-P 2024)},
  series    = {CEUR Workshop Proceedings},
  volume    = {3846},
  pages     = {132--141},
  year      = {2024},
  publisher = {CEUR-WS.org},
  address   = {Valladolid, Spain},
  month     = {September 24-27},
  urn       = {urn:nbn:de:0074-3846-7},
  url       = {https://ceur-ws.org/Vol-3846/}
}
``` 