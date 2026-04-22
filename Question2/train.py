import os, torch, cv2, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn

class CityscapesDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 96), interpolation=cv2.INTER_NEAREST)
        img = img.astype(np.float32) / 255.0
        mask = cv2.imread(self.mask_paths[idx])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, (128, 96), interpolation=cv2.INTER_NEAREST)
        mask = np.max(mask, axis=-1)
        img = torch.from_numpy(img).permute(2, 0, 1)
        mask = torch.from_numpy(mask).long()
        return img, mask

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_classes=23):
        super().__init__()
        self.enc1 = DoubleConv(3, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(256, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.out = nn.Conv2d(64, n_classes, 1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)

def compute_metrics(preds, masks, n_classes=23):
    preds = preds.argmax(dim=1)
    iou_list, dice_list = [], []
    for cls in range(n_classes):
        p = (preds == cls); m = (masks == cls)
        inter = (p & m).sum().float()
        union = (p | m).sum().float()
        if union > 0:
            iou_list.append((inter/union).item())
            dice_list.append((2*inter/(p.sum()+m.sum()+1e-8)).item())
    return np.mean(iou_list) if iou_list else 0, np.mean(dice_list) if dice_list else 0

rgb_dir = "/exam/Question2/MLDLOPs_2026_Major_Exam/CameraRGB"
mask_dir = "/exam/Question2/MLDLOPs_2026_Major_Exam/CameraMask"
imgs  = sorted([os.path.join(rgb_dir,  f) for f in os.listdir(rgb_dir)  if f.endswith('.png')])
masks = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')])
print(f"Total images: {len(imgs)}")
tr_imgs, te_imgs, tr_masks, te_masks = train_test_split(imgs, masks, test_size=0.2, random_state=42)
train_loader = DataLoader(CityscapesDataset(tr_imgs, tr_masks), batch_size=16, shuffle=True, num_workers=4)
test_loader  = DataLoader(CityscapesDataset(te_imgs, te_masks), batch_size=16, shuffle=False, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
model = UNet(23).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

losses, mious, mdices = [], [], []
for epoch in range(20):
    model.train(); ep_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward(); optimizer.step()
        ep_loss += loss.item()
    model.eval(); all_iou, all_dice = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            iou, dice = compute_metrics(model(xb), yb)
            all_iou.append(iou); all_dice.append(dice)
    miou = np.mean(all_iou); mdice = np.mean(all_dice)
    losses.append(ep_loss/len(train_loader))
    mious.append(miou); mdices.append(mdice)
    print(f"Epoch {epoch+1}/20 | Loss:{losses[-1]:.4f} | mIOU:{miou:.4f} | mDice:{mdice:.4f}")

os.makedirs("/exam/Question2/Question2", exist_ok=True)
torch.save(model.state_dict(), "/exam/Question2/unet_model.pth")
fig, axes = plt.subplots(1,3,figsize=(15,4))
axes[0].plot(losses); axes[0].set_title("Training Loss")
axes[1].plot(mious);  axes[1].set_title("mIOU")
axes[2].plot(mdices); axes[2].set_title("mDice")
plt.tight_layout()
plt.savefig("/exam/Question2/Question2/training_plots.png")
print(f"\nFINAL mIOU:{mious[-1]:.4f} | mDice:{mdices[-1]:.4f}")
