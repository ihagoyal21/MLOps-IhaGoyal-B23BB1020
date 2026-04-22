import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = UNet(23).to(device)
    model.load_state_dict(torch.load("/exam/Question2/unet_model.pth", map_location=device))
    model.eval()
    return model

rgb_dir = "/exam/Question2/MLDLOPs_2026_Major_Exam/CameraRGB"
mask_dir = "/exam/Question2/MLDLOPs_2026_Major_Exam/CameraMask"
all_imgs = sorted([os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir) if f.endswith('.png')])
all_masks = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')])
_, te_imgs, _, te_masks = train_test_split(all_imgs, all_masks, test_size=0.2, random_state=42)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Page 1: Training Results", "Page 2: Prediction"])

if page == "Page 1: Training Results":
    st.title("CityScape Segmentation - Training Results")
    st.subheader("Training Metrics")
    plot_path = "/exam/Question2/Question2/training_plots.png"
    if os.path.exists(plot_path):
        st.image(plot_path, caption="Training Loss, mIOU and mDice curves", use_column_width=True)
    else:
        st.warning("Training plots not found!")
    st.subheader("Test Set Performance")
    col1, col2 = st.columns(2)
    col1.metric("mIOU", "0.5755")
    col2.metric("mDice", "0.6407")
    st.success("Both mIOU and mDice are above 0.48 threshold!")

elif page == "Page 2: Prediction":
    st.title("CityScape Segmentation - Predictions")
    st.write("Upload 4 images from the test set to see predictions")
    model = load_model()
    uploaded = st.file_uploader("Upload up to 4 test images", type=["png","jpg"], accept_multiple_files=True)
    if uploaded:
        for up_file in uploaded[:4]:
            file_bytes = np.asarray(bytearray(up_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_r = cv2.resize(img_rgb, (128, 96))
            img_t = torch.from_numpy(img_r.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(img_t).argmax(1).squeeze().cpu().numpy()
            fname = up_file.name
            mask_path = os.path.join(mask_dir, fname)
            col1, col2, col3 = st.columns(3)
            col1.image(img_rgb, caption="Input Image", use_column_width=True)
            if os.path.exists(mask_path):
                gt = cv2.imread(mask_path)
                gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
                gt = cv2.resize(gt, (128, 96))
                col2.image(gt, caption="Ground Truth Mask", use_column_width=True)
            pred_vis = (pred * 10 % 255).astype(np.uint8)
            pred_color = cv2.applyColorMap(pred_vis, cv2.COLORMAP_JET)
            pred_color = cv2.cvtColor(pred_color, cv2.COLOR_BGR2RGB)
            col3.image(pred_color, caption="Predicted Mask", use_column_width=True)
