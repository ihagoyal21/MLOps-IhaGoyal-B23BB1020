
# MLOps Assignment 3 - BERT Fine-Tuning for CoLA

## Project Overview
This project demonstrates an end-to-end MLOps pipeline for fine-tuning a BERT-base model on the CoLA (Corpus of Linguistic Acceptability) dataset. 

## Model Registry (Task 7 & 8)
The trained model weights and tokenizer artifacts are hosted on the Hugging Face Model Hub:
**Link:** [https://huggingface.co/ihagoyal21/my-exam-model](https://huggingface.co/ihagoyal21/my-exam-model)

## Evaluation Results (Task 9)
The following results were obtained by running the production-ready Docker container (\`Dockerfile.prod\`), which pulls the model directly from the registry:

***** eval metrics *****

epoch                   = 3.0
eval_accuracy           = 0.8178
eval_loss               = 0.4925
eval_runtime            = 10.4521
eval_samples_per_second = 102.04
eval_steps_per_second   = 12.82
  
## How to Run Production Evaluation
To reproduce these results, run:
\`docker build -f Dockerfile.prod -t mldl-prod-eval .\`
\`docker run --rm -e HF_TOKEN="your_token_here" mldl-prod-eval\`

