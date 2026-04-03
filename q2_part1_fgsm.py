import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np
import wandb
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize WandB
wandb.init(project="DLOps_Ass5_Q2", name="ResNet18_FGSM_Attacks")

# 1. Data Preparation (No mean/std normalization to keep pixel values 0-1 for easy adversarial clipping)
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform) # Wait, assignment says CIFAR-10 for Q2!
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

# 2. Train Base ResNet-18 from scratch
model = resnet18(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("Training base ResNet-18 to >= 72%...")
epochs = 12 # Should hit 72%+ within 10-15 epochs
for epoch in range(epochs):
    model.train()
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()
    
    # Check Accuracy
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, pred = model(inputs).max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
    acc = 100. * correct / total
    print(f"Epoch {epoch+1} | Clean Test Acc: {acc:.2f}%")
    wandb.log({"Clean_Test_Acc": acc})

# Save clean weights
torch.save(model.state_dict(), "resnet18_clean.pth")

# 3. Custom FGSM Implementation
def custom_fgsm(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return torch.clamp(perturbed_image, 0, 1)

# 4. IBM ART Setup
classifier = PyTorchClassifier(
    model=model, clip_values=(0.0, 1.0), loss=criterion,
    optimizer=optimizer, input_shape=(3, 32, 32), nb_classes=10
)
art_fgsm = FastGradientMethod(estimator=classifier, eps=0.1)

# 5. Evaluate & Log 10 Samples
print("Running Attacks and logging samples to WandB...")
model.eval()
sample_images = []
epsilons = [0.05, 0.1, 0.15, 0.2]

# Just grab one batch for the visualization and accuracy drop
dataiter = iter(testloader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)

# A. Custom FGSM Attack
images.requires_grad = True
outputs = model(images)
loss = criterion(outputs, labels)
model.zero_grad()
loss.backward()
data_grad = images.grad.data
custom_adv_images = custom_fgsm(images, 0.1, data_grad)

# B. ART FGSM Attack
art_adv_images = art_fgsm.generate(x=images.detach().cpu().numpy())
art_adv_images = torch.tensor(art_adv_images).to(device)

# Log 10 samples to WandB
for i in range(10):
    clean_img = wandb.Image(images[i].detach().cpu().numpy().transpose(1, 2, 0), caption=f"Clean_{i}")
    custom_img = wandb.Image(custom_adv_images[i].detach().cpu().numpy().transpose(1, 2, 0), caption=f"CustomFGSM_{i}")
    art_img = wandb.Image(art_adv_images[i].detach().cpu().numpy().transpose(1, 2, 0), caption=f"ART_FGSM_{i}")
    sample_images.extend([clean_img, custom_img, art_img])

wandb.log({"FGSM_Comparison_Samples": sample_images})
print("Part 1 Complete! Model saved and images logged to WandB.")
wandb.finish()
