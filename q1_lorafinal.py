%%writefile q1_lora.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from transformers import ViTForImageClassification
from peft import LoraConfig, get_peft_model
import optuna
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

def objective(trial):
    r = trial.suggest_categorical("r", [2, 4, 8])
    alpha = trial.suggest_categorical("alpha", [2, 4, 8])
    
    run_name = f"LoRA_r{r}_alpha{alpha}"
    wandb.init(project="DLOps_Ass5_Q1", name=run_name, reinit=True)

    base_model = ViTForImageClassification.from_pretrained(
        'WinKawaks/vit-small-patch16-224', 
        num_labels=100,
        ignore_mismatched_sizes=True
    )
    
    # Inject LoRA into Q, K, V and keep classifier trainable
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["query", "key", "value"],
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier"]
    )
    
    model = get_peft_model(base_model, config)
    model.to(device)
    
    # Watch model for Gradients
    wandb.watch(model, log="all", log_freq=10)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.config.update({"Trainable_Params": trainable_params})

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val_acc = 0.0

    # 10 Epochs as required
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
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Only save the best overall model weights
            model.save_pretrained(f"./best_lora_weights_r{r}_a{alpha}")

    # Class-wise Histogram for this LoRA config
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
    wandb.log({f"Class_Acc_Histogram_r{r}_a{alpha}": wandb.Histogram(class_accuracies)})

    wandb.finish()
    return best_val_acc

if __name__ == "__main__":
    # Force Optuna to check all 9 combinations using GridSampler
    search_space = {"r": [2, 4, 8], "alpha": [2, 4, 8]}
    study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space), direction="maximize")
    study.optimize(objective)

    print("Best parameters:", study.best_params)
    print("Best Validation Accuracy:", study.best_value)