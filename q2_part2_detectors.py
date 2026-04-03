import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet34, resnet18
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import wandb
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescent, BasicIterativeMethod

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandb.init(project="DLOps_Ass5_Q2", name="ResNet34_Detectors")

transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# To save massive amounts of time on Colab, we use a subset of training data (10,000 images) to train the detector
subset_indices = np.random.choice(len(trainset), 10000, replace=False)
clean_images = torch.stack([trainset[i][0] for i in subset_indices])
clean_labels = torch.tensor([trainset[i][1] for i in subset_indices])

# 1. Load the Base Model from Part 1
base_model = resnet18(num_classes=10).to(device)
base_model.load_state_dict(torch.load("resnet18_clean.pth"))
base_model.eval()

classifier = PyTorchClassifier(
    model=base_model, clip_values=(0.0, 1.0), loss=nn.CrossEntropyLoss(),
    optimizer=None, input_shape=(3, 32, 32), nb_classes=10
)

# 2. Generate PGD and BIM Attacks via ART
print("Generating PGD Attacks...")
pgd = ProjectedGradientDescent(estimator=classifier, eps=0.1)
pgd_adv_images = pgd.generate(x=clean_images.numpy())

print("Generating BIM Attacks...")
bim = BasicIterativeMethod(estimator=classifier, eps=0.1)
bim_adv_images = bim.generate(x=clean_images.numpy())

# Log visual samples
wandb.log({"PGD_Samples": [wandb.Image(img.transpose(1, 2, 0)) for img in pgd_adv_images[:5]]})
wandb.log({"BIM_Samples": [wandb.Image(img.transpose(1, 2, 0)) for img in bim_adv_images[:5]]})

# 3. Create Binary Datasets (0=Clean, 1=Adversarial)
clean_y = torch.zeros(len(clean_images), dtype=torch.long)
adv_y = torch.ones(len(clean_images), dtype=torch.long)

# Dataset A: Clean + PGD
x_pgd = torch.cat([clean_images, torch.tensor(pgd_adv_images)])
y_pgd = torch.cat([clean_y, adv_y])
pgd_loader = DataLoader(TensorDataset(x_pgd, y_pgd), batch_size=128, shuffle=True)

# Dataset B: Clean + BIM
x_bim = torch.cat([clean_images, torch.tensor(bim_adv_images)])
y_bim = torch.cat([clean_y, adv_y])
bim_loader = DataLoader(TensorDataset(x_bim, y_bim), batch_size=128, shuffle=True)

# 4. Train Detector Function
def train_detector(dataloader, attack_name):
    print(f"Training ResNet-34 Detector for {attack_name}...")
    detector = resnet34(num_classes=2).to(device) # Binary Classification
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(detector.parameters(), lr=1e-3)
    
    epochs = 5 # Binary classification converges fast
    for epoch in range(epochs):
        detector.train()
        correct, total = 0, 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = detector(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
            
        acc = 100. * correct / total
        print(f"Epoch {epoch+1} | {attack_name} Detection Acc: {acc:.2f}%")
        wandb.log({f"{attack_name}_Detector_Acc": acc})
        
    torch.save(detector.state_dict(), f"resnet34_{attack_name}_detector.pth")
    return detector

# Train both detectors
detector_pgd = train_detector(pgd_loader, "PGD")
detector_bim = train_detector(bim_loader, "BIM")

print("Part 2 Complete! All weights saved.")
wandb.finish()
