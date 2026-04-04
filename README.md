
# DLOps Assignment 5: LoRA & Adversarial Attacks (IBM ART)

**Author:** Iha Goyal  
**Roll Number:** B23BB1020  

## 🔗 Important Links
* **GitHub Branch (Assignment 5):** https://github.com/ihagoyal21/MLOps-IhaGoyal-B23BB1020/tree/Assignment-5


* **Hugging Face Model (Q1 Best LoRA Weights):** https://huggingface.co/ihagoyal21/B23BB1020-Ass5-Q1-LoRA/tree/main

* **WandB Dashboard (Q1 - ViT LoRA):** https://wandb.ai/ihagoyal-mun-indian-instit/DLOps_Ass5_Q1/workspace?nw=nwuserihagoyalmun


* **WandB Dashboard (Q2 - Adversarial Attacks):** https://wandb.ai/ihagoyal-mun-indian-instit/DLOps_Ass5_Q2/table?nw=nwuserihagoyalmun


---

## ⚙️ Installation & Setup
To run the experiments, ensure you have a Python environment set up (preferably within a Docker container as per assignment guidelines) and install the required dependencies.

```bash
pip install -r requirements.txt
```

*Note: The `requirements.txt` includes `torch`, `torchvision`, `transformers`, `peft==0.18.1`, `optuna`, and `wandb`.*

---

## 🚀 How to Run the Code

### Q1: ViT-Small Finetuning (CIFAR-100)
1. **Run the Baseline Model (No LoRA):**
   ```bash
   python q1_baseline.py
   ```
2. **Run the LoRA Finetuning (Optuna Grid Search & Best Config):**
   ```bash
   python q1_lora.py
   ```

### Q2: Adversarial Attacks & Detectors (CIFAR-10)
1. **Run FGSM Attack (Scratch vs. IBM ART):**
   ```bash
   python q2_part1_fgsm.py
   ```
2. **Run Adversarial Detectors (PGD & BIM):**
   ```bash
   python q2_part2_detectors.py
   ```
*(Note: Qualitative image samples of clean vs. adversarial images across FGSM, PGD, and BIM attacks can be viewed directly on the Q2 WandB Dashboard linked above).*

---

## 📊 Q1 Results & Tables

### Step 3: Best LoRA Configuration Training Progression
**Hyperparameters:** Rank = 8, Alpha = 8, Dropout = 0.1  
*Target Modules: Query, Key, Value*

| Epoch | Training Loss | Validation Loss | Training Accuracy | Validation Accuracy |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 55.123 | 36.576 | 85.07% | 88.81% |
| 2 | 25.673 | 36.811 | 92.16% | **88.89%** |
| 3 | 17.818 | 40.62 | 94.57% | 88.66% |
| 4 | 13.410 | 43.017 | 96.24% | 88.53% |
| 5 | 10.820 | 46.431 | 96.76% | 88.50% |
| 6 | 9.723 | 51.187 | 97.47% | 88.08% |
| 7 | 8.307 | 51.261 | 97.47% | 88.05% |
| 8 | 8.236 | 52.938 | 97.78% | 88.44% |
| 9 | 7.253 | 56.437 | 98.00% | 87.67% |
| 10 | 6.823 | 60.436 | 98.10% | 88.08% |

*(Note: Exact loss values are logged and available on the WandB dashboard).*

### Step 4: Testing Summary Table
Comparison of the baseline ViT-Small model versus the best LoRA configuration injected into the Q, K, and V projection matrices.

| LORA layers (with/without) | Rank | Alpha | Dropout | Overall Test Accuracy | Trainable Parameters used |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Without (Baseline)** | N/A | N/A | N/A | **83.00%** | **38,500** |
| **With (Q, K, V)** | 8 | 8 | 0.1 | **88.89%** | **259,684** |

---

## 🛠️ Technical Specifications
* **Base Model:** `WinKawaks/vit-small-patch16-224`
* **Frameworks:** PyTorch, Transformers, PEFT (v0.18.1), Optuna, Weights & Biases (WandB)
* **Dataset:** CIFAR-100 (Q1) / CIFAR-10 (Q2)
```
