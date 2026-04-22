from transformers import MarianMTModel, MarianTokenizer
import sacrebleu
from striprtf.striprtf import rtf_to_text

model_name = "Helsinki-NLP/opus-mt-bn-en"
print("Loading model...")
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
print("Model loaded!")

with open("Q1/input.rtf", "r", encoding="utf-8") as f:
    raw = f.read()
input_text = rtf_to_text(raw)
lines = [l.strip() for l in input_text.split("\n") if l.strip()]
print(f"Total lines to translate: {len(lines)}")

translated = []
for i, line in enumerate(lines):
    tokens = tokenizer([line], return_tensors="pt", padding=True, truncation=True, max_length=512)
    out = model.generate(**tokens)
    result = tokenizer.decode(out[0], skip_special_tokens=True)
    translated.append(result)
    print(f"[{i+1}/{len(lines)}] {result}")

with open("output.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(translated))
print("\nFirst translation:", translated[0])

with open("Q1/output.rtf", "r", encoding="utf-8") as f:
    raw_ref = f.read()
ref_text = rtf_to_text(raw_ref)
refs = [l.strip() for l in ref_text.split("\n") if l.strip()]

bleu = sacrebleu.corpus_bleu(translated, [refs[:len(translated)]])
print(f"\nBLEU Score: {bleu.score:.2f}")
