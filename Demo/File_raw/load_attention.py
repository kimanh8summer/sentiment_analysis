import os
import torch
import pandas as pd
import collections
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
from transformers import BertTokenizer, BertModel, ViTModel
import torchvision.transforms as T

# CONFIG 
INPUT_PATH = "/kaggle/input/datasets/nanguyen16/checkpoint" 
MODEL_PATH = f"{INPUT_PATH}/best_vit.pt"
CSV_PATH = f"{INPUT_PATH}/mvsa_clean/mvsa.csv"
MVSA_IMAGE_DIR = "/kaggle/input/datasets/nanguyen16/mvsa-dataset/MVSA/data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
basic_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ==========================================
# 2. KIẾN TRÚC MODEL & DATASET
# ==========================================
class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model=768, num_heads=8):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.img_proj = nn.Linear(768, d_model)
        self.t2i_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True, dropout=0.1)
        self.i2t_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True, dropout=0.1)
        self.norm_T = nn.LayerNorm(d_model); self.norm_I = nn.LayerNorm(d_model)
        self.gate_fc = nn.Linear(d_model * 2, d_model)
        self.classifier = nn.Sequential(nn.Linear(d_model, 512), nn.GELU(), nn.Dropout(0.5), nn.Linear(512, 3))

    def forward(self, ids, mask, img, return_attn=False):
        text_out = self.bert(input_ids=ids, attention_mask=mask)
        H_T = text_out.last_hidden_state 
        vit_out = self.vit(pixel_values=img)
        H_I = self.img_proj(vit_out.last_hidden_state) 
        key_padding_mask = (mask == 0)
        attn_T2I, weights_T2I = self.t2i_attn(query=H_T, key=H_I, value=H_I, average_attn_weights=True)
        H_T = self.norm_T(H_T + attn_T2I)
        attn_I2T, weights_I2T = self.i2t_attn(query=H_I, key=H_T, value=H_T, key_padding_mask=key_padding_mask, average_attn_weights=True)
        H_I = self.norm_I(H_I + attn_I2T)
        v_T = torch.sum(H_T * mask.unsqueeze(-1), 1) / torch.clamp(mask.unsqueeze(-1).sum(1), min=1e-9)
        v_I = torch.mean(H_I, dim=1)
        v_fuse = torch.sigmoid(self.gate_fc(torch.cat([v_T, v_I], dim=-1))) * v_T + (1 - torch.sigmoid(self.gate_fc(torch.cat([v_T, v_I], dim=-1)))) * v_I
        logits = self.classifier(v_fuse)
        if return_attn: return logits, weights_T2I, weights_I2T
        return logits

class MVSADataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path); self.img_dir = img_dir; self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try: img = Image.open(os.path.join(self.img_dir, row["image"])).convert("RGB")
        except: img = Image.new("RGB", (224, 224), (0, 0, 0))
        if self.transform: img = self.transform(img)
        enc = tokenizer(str(row["text"]), padding="max_length", truncation=True, max_length=64, return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(), "attention_mask": enc["attention_mask"].squeeze(), "image": img, "label": int(row["label"]), "idx": idx}

# ==========================================
# 3. CHƯƠNG TRÌNH CHÍNH
# ==========================================
def main():
    # 1. Load Model
    model = CrossAttentionFusion().to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # 2. Chuẩn bị Data
    dataset = MVSADataset(CSV_PATH, MVSA_IMAGE_DIR, transform=basic_transform)
    n = len(dataset)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
    test_indices = indices[int(0.8*n) + int(0.1*n):] 
    test_loader = DataLoader(torch.utils.data.Subset(dataset, test_indices), batch_size=32)

    # 3. Lọc mẫu đúng
    correct_indices = []
    with torch.no_grad():
        for b in tqdm(test_loader, desc="Đang lọc mẫu"):
            ids, mask, img, y = b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE), b["image"].to(DEVICE), b["label"].to(DEVICE)
            preds = model(ids, mask, img).argmax(1)
            matched = (preds == y).cpu().numpy()
            correct_indices.extend(np.array(b["idx"])[matched].tolist())

    print(f" Tìm thấy {len(correct_indices)} mẫu đoán đúng.")

    # 4. Hiển thị
    start_idx = 0 
    num_show = 100
    selected = correct_indices[start_idx : start_idx + num_show]

    fig, axes = plt.subplots(len(selected), 2, figsize=(20, len(selected) * 5))
    
    for i, idx in enumerate(selected):
        item = dataset[idx]
        ids, mask, img_t = item['input_ids'].unsqueeze(0).to(DEVICE), item['attention_mask'].unsqueeze(0).to(DEVICE), item['image'].unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logits, w_t2i, w_i2t = model(ids, mask, img_t, return_attn=True)
        
        # --- CỘT 1: TEXT ATTENTION ---
        valid_len = mask.sum().item()
        tokens = tokenizer.convert_ids_to_tokens(item['input_ids'])[:valid_len]
        # Trọng số i2t: 
        text_attn = w_i2t[0].mean(dim=0).cpu().numpy()[:valid_len]
        sns.heatmap([text_attn], annot=[tokens], fmt='', cmap='YlOrRd', ax=axes[i, 0], cbar=False)
        axes[i, 0].set_title(f"Idx: {idx} | Label: {item['label']} (Dự đoán đúng)")

        # --- CỘT 2: IMAGE ATTENTION ---
        # Trọng số t2i: 
        img_attn = w_t2i[0, :valid_len, 1:].mean(dim=0).cpu().numpy() 
        heatmap = cv2.resize(img_attn.reshape(14, 14), (224, 224))
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        orig_img = np.clip((item['image'].permute(1, 2, 0).cpu().numpy() * 0.5) + 0.5, 0, 1)
        axes[i, 1].imshow(orig_img)
        axes[i, 1].imshow(heatmap, cmap='jet', alpha=0.4)
        axes[i, 1].axis('off')
        axes[i, 1].set_title("Vùng ảnh được mô hình chú ý")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

