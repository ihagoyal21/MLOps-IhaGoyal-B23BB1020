from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from data_prep import load_cola

def train():
    # 1. Load Tokenizer and Model (Task 4)
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 2. Prepare Data
    ds = load_cola().train_test_split(test_size=0.1)
    def tokenize_func(examples):
        return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=64)
    
    tokenized_ds = ds.map(tokenize_func, batched=True)

    # 3. Training Arguments (Task 5)
    args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=1, # 1 for exam speed
        per_device_train_batch_size=16
    )

    # 4. Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        tokenizer=tokenizer
    )

    trainer.train()
    model.save_pretrained("./local_model")
    tokenizer.save_pretrained("./local_model")

if __name__ == "__main__":
    train()