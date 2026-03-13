#!/usr/bin/env python3
"""
Tek bir görsel üzerinde PSPNet (RescueNet 3-class) inference yapar;
orijinal görsel, renkli maske ve overlay yan yana gösterir.

Kullanım:
  python predict_and_show.py --image path/to/image.jpg --checkpoint path/to/train_epoch_1.pth
  python predict_and_show.py --image path/to/image.jpg --checkpoint path/to/train_epoch_1.pth --out result.png
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

# Proje kökü (Segmentation-Experiments)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Sabitler (config ile aynı)
CLASSES = 3  # Background, Building Light, Building Heavy
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
# Girdi boyutu: (H-1) ve (W-1) 8'in katı olmalı
IN_H, IN_W = 769, 1025  # 4:3 oran, eğitimle aynı

# Sınıf renkleri (RGB): Background, Building Light, Building Heavy
COLORS = np.array([
    [0, 0, 0],           # 0: Background - siyah
    [0, 255, 0],         # 1: Building Light - yeşil
    [255, 0, 0],         # 2: Building Heavy - kırmızı
], dtype=np.uint8)


def load_model(checkpoint_path, device):
    from models.pspnet import PSPNet
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    model = PSPNet(
        layers=101,
        bins=(1, 2, 3, 6),
        dropout=0.1,
        classes=CLASSES,
        zoom_factor=8,
        use_ppm=True,
        criterion=criterion,
        BatchNorm=nn.BatchNorm2d,
        pretrained=False,
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    # DataParallel ile kaydedildiyse "module." öneki kaldır
    new_state = {}
    for k, v in state.items():
        name = k[7:] if k.startswith("module.") else k
        new_state[name] = v
    model.load_state_dict(new_state, strict=True)
    model.to(device)
    model.eval()
    return model


def preprocess(image_path):
    img = Image.open(image_path).convert("RGB")
    img = np.array(img)
    # Resize
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((IN_W, IN_H), Image.BILINEAR)
    img = np.array(pil_img)
    # HWC -> CHW, float, normalize
    img = img.astype(np.float32) / 255.0
    img = (img - np.array(MEAN)) / np.array(STD)
    img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
    return img, np.array(Image.open(image_path).convert("RGB"))


def predict(model, tensor, device):
    with torch.no_grad():
        tensor = tensor.to(device)
        out = model(tensor)
        # out: (1, C, H, W) logits
        pred = out.argmax(1).squeeze(0).cpu().numpy()
    return pred


def mask_to_rgb(mask):
    h, w = mask.shape
    rgb = COLORS[mask.flat].reshape(h, w, 3)
    return rgb


def overlay(image, mask, alpha=0.5):
    rgb_mask = mask_to_rgb(mask)
    img_float = image.astype(np.float32) / 255.0
    mask_float = rgb_mask.astype(np.float32) / 255.0
    # Sadece sınıf > 0 olan yerlerde overlay
    blend = img_float.copy()
    for c in range(3):
        blend[:, :, c] = np.where(mask > 0, alpha * mask_float[:, :, c] + (1 - alpha) * img_float[:, :, c], img_float[:, :, c])
    return (blend * 255).clip(0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="PSPNet RescueNet 3-class inference ve görselleştirme")
    parser.add_argument("--image", type=str, required=True, help="Girdi görsel yolu")
    parser.add_argument("--checkpoint", type=str, required=True, help="train_epoch_X.pth dosya yolu")
    parser.add_argument("--out", type=str, default=None, help="Çıktı görseli kaydedilecek path (opsiyonel)")
    parser.add_argument("--no-show", action="store_true", help="Pencere açmadan sadece kaydet")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print("Görsel bulunamadı:", args.image)
        sys.exit(1)
    if not os.path.isfile(args.checkpoint):
        print("Checkpoint bulunamadı:", args.checkpoint)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Cihaz:", device)
    print("Model yükleniyor...")
    model = load_model(args.checkpoint, device)
    print("Görsel yükleniyor ve ön işleniyor...")
    tensor, _ = preprocess(args.image)
    # Gösterim için görseli pred ile aynı boyutta (IN_H x IN_W) kullan
    img_display = np.array(Image.open(args.image).convert("RGB").resize((IN_W, IN_H), Image.BILINEAR))

    print("Inference çalıştırılıyor...")
    pred = predict(model, tensor, device)

    mask_rgb = mask_to_rgb(pred)
    over = overlay(img_display, pred, alpha=0.5)

    _, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(img_display)
    axes[0].set_title("Orijinal")
    axes[0].axis("off")
    axes[1].imshow(mask_rgb)
    axes[1].set_title("Segmentasyon (Yeşil: Bina hafif, Kırmızı: Bina ağır)")
    axes[1].axis("off")
    axes[2].imshow(over)
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    plt.tight_layout()

    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(args.out, dpi=150, bbox_inches="tight")
        print("Kaydedildi:", args.out)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
