# This is script for Indonesian MDERank testing
dataset_name=SemEval2017_Indonesian

# Create necessary directories if they don't exist
mkdir -p log_path
mkdir -p results

# Run Indonesian MDERank evaluation
echo "Running Indonesian keyphrase extraction..."
python MDERank-indonesia/mderank_exec_indonesian.py \
    --dataset_dir data/$dataset_name \
    --model_name indolem/indobert-base-uncased \
    --num_keyphrases 10 \
    --batch_size 1 \
    --device cpu \
    --output_dir results

# Print results after completion
echo "Evaluation completed. Results are stored in results/results.json"
echo "Check results/extraction.log for detailed extraction information"

# Alternative models you can try:
# --model_name bert-base-multilingual-uncased
# --model_name cahya/bert-base-indonesian-522M
# --model_name indobenchmark/indobert-base-p1 