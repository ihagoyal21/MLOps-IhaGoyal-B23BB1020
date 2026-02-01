## Lab 2: CNN on CIFAR-10

## Submission Links
- **W&B Public Report:** https://wandb.ai/iha-goyal21-iit-jodhpur/Lab2_Final_Submission/reports/ML-DL-OPS-LAB2---VmlldzoxNTgxMDA3OQ

## Key Findings
- **Final Training Loss:** 0.01429
- **Model Complexity:** 15.66 MFLOPS | 2.12M Parameters
- **Training Status:** Successfully converged in 25 epochs. 
- **Observations:** Gradient histograms (Requirement #6) show a stable Gaussian distribution, confirming that Batch Normalization effectively prevented vanishing or exploding gradients.

## Requirements Checklist
- [x] Custom Dataset implementation (`MyCIFAR10Dataset`)
- [x] FLOPs calculation using `calflops`
- [x] Weights & Biases logging (Loss, Gradients, and Weights)
- [x] Publicly accessible W&B Report
