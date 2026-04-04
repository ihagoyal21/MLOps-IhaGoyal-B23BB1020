import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from transformers import ViTForImageClassification
import wandb

def main():
    wandb.init(project="DLOps_Ass5_Q1", name="Baseline_ViT_Small")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CIFAR-100 Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    # 1. Load ViT-Small and freeze everything EXCEPT the classifier
    model = ViTForImageClassification.from_pretrained(
        'WinKawaks/vit-small-patch16-224', # Standard ViT-Small
        num_labels=100,
        ignore_mismatched_sizes=True
    )
    
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    model.to(device)
    
    # Track gradients in WandB
    wandb.watch(model, log="all", log_freq=10)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)

    # Train for 10 Epochs
    for epoch in range(10):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(trainloader)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).logits
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(testloader)

        wandb.log({
            "Epoch": epoch + 1,
            "Train Loss": avg_train_loss,
            "Val Loss": avg_val_loss,
            "Train Acc": train_acc,
            "Val Acc": val_acc
        })
        print(f"Epoch {epoch+1} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    # Calculate Class-wise Accuracy for Histogram
    class_correct = list(0. for _ in range(100))
    class_total = list(0. for _ in range(100))
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).logits
            _, predicted = outputs.max(1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    class_accuracies = [100 * class_correct[i] / class_total[i] for i in range(100)]
    wandb.log({"Baseline_Class_Accuracy_Histogram": wandb.Histogram(class_accuracies)})
    
    # Save parameters count
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Baseline Trainable Params: {trainable_params}")
    wandb.finish()

if __name__ == "__main__":
    main()
