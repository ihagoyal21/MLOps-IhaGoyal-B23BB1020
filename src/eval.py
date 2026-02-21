import argparse
import os
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from data_prep import load_cola

# --- PASTE YOUR TOKEN BELOW INSIDE THE QUOTES ---

MY_TOKEN = "******************" 
# ------------------------------------------------

def evaluate(model_path):
    print(f"\n🚀 Starting Evaluation for: {model_path}")
    
    try:
        # We explicitly pass the token string here. No env vars, no confusion.
        tokenizer = BertTokenizer.from_pretrained(
            model_path, 
            force_download=True, 
            token=MY_TOKEN
        )
        model = BertForSequenceClassification.from_pretrained(
            model_path, 
            force_download=True, 
            token=MY_TOKEN
        )
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: Could not download model. Check your token.\nError: {e}")
        return

    ds = load_cola().train_test_split(test_size=0.1)
    
    def tokenize_func(examples):
        return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=64)
    
    tokenized_test = ds["test"].map(tokenize_func, batched=True)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="./temp_eval", per_device_eval_batch_size=16, remove_unused_columns=False),
        eval_dataset=tokenized_test,
    )

    print("--- Running Inference ---")
    results = trainer.evaluate()
    print(f"\n✅ Results: {results}")

    # Save to file
    with open("eval_results.txt", "a") as f:
        f.write(f"Source: {model_path}\nResults: {str(results)}\n{'-'*40}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ihagoyal21/my-exam-model")
    args = parser.parse_args()
    evaluate(args.model)