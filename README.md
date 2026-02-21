
# MLOps Assignment 3 - BERT Fine-Tuning for CoLA

## Project Overview
This project demonstrates an end-to-end MLOps pipeline for fine-tuning a BERT-base model on the CoLA (Corpus of Linguistic Acceptability) dataset. 

## Model Registry (Task 7 & 8)
The trained model weights and tokenizer artifacts are hosted on the Hugging Face Model Hub:
**Link:** [https://huggingface.co/ihagoyal21/my-exam-model](https://huggingface.co/ihagoyal21/my-exam-model)

## Evaluation Results (Task 9)

I verified the model performance using both the local artifacts and the remote Hugging Face Registry to ensure consistency.

### Local Model Results:
- Eval Loss: 0.2767
- Throughput: 12.58 samples/sec

### Hugging Face Registry Results:
- Eval Loss: 0.2930
- Throughput: 14.38 samples/sec

Conclusion: The model pulled from the registry maintains the performance observed during the local development phase, verifying a successful deployment pipeline.
  
## How to Run Production Evaluation
To reproduce these results, run:
\`docker build -f Dockerfile.prod -t mldl-prod-eval .\`
\`docker run --rm -e HF_TOKEN="your_token_here" mldl-prod-eval\`

