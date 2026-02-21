from transformers import BertForSequenceClassification, BertTokenizer
model = BertForSequenceClassification.from_pretrained("./local_model")
tokenizer = BertTokenizer.from_pretrained("./local_model")

# Replace with your username!
model.push_to_hub("ihagoyal21/my-exam-model")
tokenizer.push_to_hub("ihagoyal21/my-exam-model")