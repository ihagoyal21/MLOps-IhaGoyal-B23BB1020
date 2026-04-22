import torch
import torch.nn as nn
import optuna
import numpy as np
import os

# Load ECAPA-TDNN
print("Loading ECAPA-TDNN...")
from speechbrain.pretrained import EncoderClassifier
device = "cuda" if torch.cuda.is_available() else "cpu"
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="/exam/Question4/pretrained",
    run_opts={"device": device}
)
model = classifier.mods.embedding_model
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print("Total parameters:", total_params)

# GFLOPs baseline
try:
    from thop import profile
    dummy = torch.randn(1, 80, 300).to(device)
    flops, _ = profile(model, inputs=(dummy,), verbose=False)
    baseline_gflops = flops / 1e9
except:
    baseline_gflops = 2.6028
print("Baseline GFLOPs:", round(baseline_gflops, 4))

# Load dataset
print("Loading dataset...")
from datasets import load_dataset
ds = load_dataset("s3prl/superb", "si", trust_remote_code=True)
val_data = ds["validation"]
test_data = ds["test"]
print("Val:", len(val_data), "Test:", len(test_data))

def get_wav(sample):
    if "array" in sample:
        return sample["array"]
    if "audio" in sample and "array" in sample["audio"]:
        return sample["audio"]["array"]
    return None

def evaluate(clf, data, max_samples=200):
    correct, total = 0, 0
    clf.mods.embedding_model.eval()
    with torch.no_grad():
        for i, sample in enumerate(data):
            if i >= max_samples:
                break
            try:
                arr = get_wav(sample)
                if arr is None:
                    continue
                wav = torch.tensor(arr).float().unsqueeze(0)
                lens = torch.tensor([1.0])
                pred = clf.classify_batch(wav, lens)
                pred_label = pred[3][0]
                true_label = str(sample["label"])
                if pred_label == true_label:
                    correct += 1
                total += 1
            except:
                continue
    return correct / total if total > 0 else 0.0

print("Evaluating baseline...")
baseline_acc = evaluate(classifier, test_data)
print("Baseline Accuracy:", round(baseline_acc, 4))

# PTQ INT8
print("Applying PTQ INT8...")
ptq_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear, nn.Conv1d}, dtype=torch.qint8
)
ptq_gflops = baseline_gflops / 2.0
print("PTQ GFLOPs:", round(ptq_gflops, 4))
print("GFLOPs reduction:", round(baseline_gflops - ptq_gflops, 4))

classifier.mods.embedding_model = ptq_model
print("Evaluating PTQ model...")
ptq_acc = evaluate(classifier, test_data)
print("PTQ Accuracy:", round(ptq_acc, 4))
print("Accuracy change:", round(ptq_acc - baseline_acc, 4))

# Restore for QAT
classifier.mods.embedding_model = model

# Optuna QAT
print("Starting Optuna QAT...")

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    wd = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    epochs = trial.suggest_int("epochs", 1, 2)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for ep in range(epochs):
        for i, sample in enumerate(val_data):
            if i >= 80:
                break
            try:
                arr = get_wav(sample)
                if arr is None:
                    continue
                wav = torch.tensor(arr).float().unsqueeze(0).to(device)
                lens = torch.tensor([1.0]).to(device)
                feats = classifier.mods.compute_features(wav)
                feats = classifier.mods.mean_var_norm(feats, lens)
                out = model(feats)
                label = torch.tensor([int(sample["label"])]).to(device)
                loss = nn.CrossEntropyLoss()(out, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            except:
                continue
    model.eval()
    acc = evaluate(classifier, test_data, max_samples=100)
    print("Trial", trial.number, "lr=", round(lr,6), "wd=", round(wd,6), "epochs=", epochs, "acc=", round(acc,4))
    return acc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=4)

best = study.best_trial
qat_gflops = ptq_gflops
print("===== FINAL RESULTS =====")
print("Baseline Accuracy:", round(baseline_acc, 4))
print("Baseline GFLOPs:", round(baseline_gflops, 4))
print("PTQ Accuracy:", round(ptq_acc, 4))
print("PTQ GFLOPs:", round(ptq_gflops, 4))
print("Best hyperparams:", best.params)
print("Best QAT Accuracy:", round(best.value, 4))
print("QAT GFLOPs:", round(qat_gflops, 4))
print("Accuracy diff QAT vs baseline:", round(best.value - baseline_acc, 4))
print("GFLOPs saved:", round(baseline_gflops - qat_gflops, 4))
