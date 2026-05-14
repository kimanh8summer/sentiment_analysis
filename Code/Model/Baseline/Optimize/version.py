import os, torch, collections, random, numpy as np, pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, ViTModel
import torchvision.transforms as T
from sklearn.metrics import f1_score

# --- 1. CONFIGURATION ---
seed_everything = lambda s: (random.seed(s), np.random.seed(s), torch.manual_seed(s), torch.cuda.manual_seed(s))
seed_everything(42)

class CFG:
    CSV_PATH = "/kaggle/input/datasets/nanguyen16/checkpoint/mvsa_clean/mvsa.csv"
    IMG_DIR = "/kaggle/input/datasets/nanguyen16/mvsa-dataset/MVSA/data"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 16 
    STAGE_1_EPOCHS = 5
    STAGE_2_EPOCHS = 10

# DANH SÁCH THÍ NGHIỆM CỦA CẬU
EXPERIMENTS = [
    {"name":"V2_cross2",      "LOSS":"ce",    "SMOOTH":0.0, "PARTIAL":False},
    {"name":"V8_Focal",       "LOSS":"focal", "SMOOTH":0.0, "PARTIAL":True},
    {"name":"V8_CE_Smooth",   "LOSS":"ce",    "SMOOTH":0.1, "PARTIAL":True},
]

# --- 2. DATASET (TỐI ƯU HÓA) ---
class MVSADataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            image = Image.open(os.path.join(self.img_dir, row["image"])).convert("RGB")
        except:
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        if self.transform: image = self.transform(image)
        
        enc = self.tokenizer(str(row["text"]), padding="max_length", truncation=True, max_length=64, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "image": image,
            "label": torch.tensor(int(row["label"]))
        }

# --- 3. MODEL ARCHITECTURE ---
class UltimateModel(nn.Module):
    def __init__(self, cfg_exp):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.img_proj = nn.Linear(768, 768)
        
        for p in self.bert.parameters(): p.requires_grad = False
        for p in self.vit.parameters(): p.requires_grad = False
        
        if cfg_exp.get("PARTIAL", False):
            for name, p in self.bert.named_parameters():
                if "encoder.layer." in name:
                    layer_id = int(name.split("encoder.layer.")[1].split(".")[0])
                    if layer_id >= 6: p.requires_grad = True
        
        self.t2i = nn.MultiheadAttention(768, 8, batch_first=True)
        self.i2t = nn.MultiheadAttention(768, 8, batch_first=True)
        self.normT, self.normI = nn.LayerNorm(768), nn.LayerNorm(768)
        self.gate = nn.Linear(768*2, 768)
        self.classifier = nn.Sequential(nn.Linear(768, 512), nn.GELU(), nn.Dropout(0.5), nn.Linear(512, 3))

    def unfreeze_all(self):
        for p in self.parameters(): p.requires_grad = True

    def forward(self, ids, mask, img):
        H_T = self.bert(ids, attention_mask=mask).last_hidden_state
        H_I = self.img_proj(self.vit(pixel_values=img).last_hidden_state)
        
        T2I, _ = self.t2i(H_T, H_I, H_I)
        H_T = self.normT(H_T + T2I)
        I2T, _ = self.i2t(H_I, H_T, H_T, key_padding_mask=(mask == 0))
        H_I = self.normI(H_I + I2T)
        
        v_T = (H_T * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1e-9)
        v_I = H_I.mean(1)
        g = torch.sigmoid(self.gate(torch.cat([v_T, v_I], dim=-1)))
        return self.classifier(g * v_T + (1 - g) * v_I)

# --- 4. TRAIN FUNCTION (TĂNG TỐC VỚI FP16) ---
def run_experiment(cfg_exp, train_loader, val_loader):
    print(f"\n🧪 ĐANG CHẠY: {cfg_exp['name']}")
    model = UltimateModel(cfg_exp).to(CFG.DEVICE)
    best_f1 = 0
    
    def focal_loss(l, y):
        ce = nn.functional.cross_entropy(l, y, reduction='none')
        return ((1 - torch.exp(-ce))**2 * ce).mean()
    
    loss_fn = focal_loss if cfg_exp["LOSS"] == "focal" else nn.CrossEntropyLoss(label_smoothing=cfg_exp["SMOOTH"])
    scaler = torch.amp.GradScaler('cuda')

    for stage, epochs, lr in [("S1", CFG.STAGE_1_EPOCHS, 2e-5), ("S2", CFG.STAGE_2_EPOCHS, 1e-6)]:
        if stage == "S2": model.unfreeze_all()
        opt = torch.optim.AdamW(filter(lambda p:p.requires_grad, model.parameters()), lr=lr)
        
        for ep in range(epochs):
            model.train()
            pbar = tqdm(train_loader, desc=f"{cfg_exp['name']}-{stage}-Ep{ep+1}")
            for b in pbar:
                ids, mask, img, y = b["input_ids"].to(CFG.DEVICE), b["attention_mask"].to(CFG.DEVICE), b["image"].to(CFG.DEVICE), b["label"].to(CFG.DEVICE)
                opt.zero_grad()
                
                with torch.amp.autocast('cuda'):
                    out = model(ids, mask, img)
                    loss = loss_fn(out, y)
                
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                pbar.set_postfix(loss=loss.item())

            # Eval
            model.eval(); y_t, y_p = [], []
            with torch.no_grad():
                for b in val_loader:
                    out = model(b["input_ids"].to(CFG.DEVICE), b["attention_mask"].to(CFG.DEVICE), b["image"].to(CFG.DEVICE))
                    y_t += b["label"].tolist(); y_p += out.argmax(1).cpu().tolist()
            
            f1 = f1_score(y_t, y_p, average="macro")
            print(f"📈 {cfg_exp['name']} Ep{ep+1} F1: {f1:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), f"best_{cfg_exp['name']}.pt")
    return best_f1

def main():
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    full_ds = MVSADataset(CFG.CSV_PATH, CFG.IMG_DIR, tf)
    
    n = len(full_ds)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
    tr_n = int(0.8*n)
    
    train_loader = DataLoader(torch.utils.data.Subset(full_ds, idx[:tr_n]), 
                              batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(torch.utils.data.Subset(full_ds, idx[tr_n:tr_n+int(0.1*n)]), 
                            batch_size=CFG.BATCH_SIZE, num_workers=2, pin_memory=True)
    
    for exp in EXPERIMENTS:
        f1 = run_experiment(exp, train_loader, val_loader)
        print(f"✅ {exp['name']} Best F1: {f1:.4f}")

if __name__ == "__main__": main()


##########
import os
import torch
import random
import collections
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, ViTModel
import torchvision.transforms as T
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# ==========================================
# 1. HỆ THỐNG & CẤU HÌNH (CONFIGURATION)
# ==========================================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

class CFG:
    # Đường dẫn dữ liệu của Nhất Hoàng (nanguyen16)
    CSV_PATH = "/kaggle/input/datasets/nanguyen16/checkpoint/mvsa_clean/mvsa.csv"
    IMG_DIR = "/kaggle/input/datasets/nanguyen16/mvsa-dataset/MVSA/data"
    CACHE_PATH = "mvsa_align_v9_1_cache.pt"
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    
    # Chiến thuật 2 giai đoạn
    STAGE_1_EPOCHS = 5
    STAGE_2_EPOCHS = 10
    
    # Learning Rate cực nhỏ ở Stage 2 để giữ biểu đồ ổn định
    LEARNING_RATE_STAGE_1 = 2e-5
    LEARNING_RATE_STAGE_2 = 5e-7 
    
    LABEL_SMOOTHING = 0.1
    ALIGN_LAMBDA = 0.1 # Trọng số cho nhiệm vụ Alignment

# ==========================================
# 2. DATASET & CACHING (TỐC ĐỘ 5 PHÚT/EPOCH)
# ==========================================
class FastAlignDataset(Dataset):
    def __init__(self, data, indices):
        self.images = data['images'][indices]
        self.input_ids = data['input_ids'][indices]
        self.attention_masks = data['attention_masks'][indices]
        self.labels = data['labels'][indices]

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "image": self.images[idx],
            "label": self.labels[idx]
        }

def get_or_create_cache():
    if os.path.exists(CFG.CACHE_PATH):
        print(f"📦 Đang nạp Cache dữ liệu: {CFG.CACHE_PATH}")
        return torch.load(CFG.CACHE_PATH)
    
    print("🛠️ Đang tạo Cache dữ liệu (Chỉ chạy 1 lần)...")
    df = pd.read_csv(CFG.CSV_PATH)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    cache = {'images': [], 'input_ids': [], 'attention_masks': [], 'labels': []}
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Samples"):
        try:
            img = Image.open(os.path.join(CFG.IMG_DIR, row["image"])).convert("RGB")
            img_tensor = transform(img)
        except:
            img_tensor = torch.zeros((3, 224, 224))
            
        encoding = tokenizer(str(row["text"]), padding="max_length", truncation=True, max_length=64, return_tensors="pt")
        
        cache['images'].append(img_tensor)
        cache['input_ids'].append(encoding["input_ids"].squeeze())
        cache['attention_masks'].append(encoding["attention_mask"].squeeze())
        cache['labels'].append(torch.tensor(int(row["label"])))
        
    for key in cache: cache[key] = torch.stack(cache[key])
    torch.save(cache, CFG.CACHE_PATH)
    return cache

# ==========================================
# 3. KIẾN TRÚC MÔ HÌNH V9.1 (MODALITY ALIGNMENT)
# ==========================================
class UltimateModelV9_1(nn.Module):
    def __init__(self):
        super().__init__()
        # Backbones
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.image_projection = nn.Linear(768, 768)
        
        # Cross-Attention
        self.text_to_image_attn = nn.MultiheadAttention(768, 8, batch_first=True)
        self.image_to_text_attn = nn.MultiheadAttention(768, 8, batch_first=True)
        
        # Norm & Gate
        self.norm_text = nn.LayerNorm(768)
        self.norm_image = nn.LayerNorm(768)
        self.gate = nn.Linear(768 * 2, 768)
        
        # Output Heads
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 3)
        )
        # Projection Head cho nhiệm vụ Alignment
        self.align_proj = nn.Linear(768, 256)

    def forward(self, ids, mask, img):
        # Đặc trưng gốc
        H_text = self.bert(ids, attention_mask=mask).last_hidden_state
        H_image = self.image_projection(self.vit(pixel_values=img).last_hidden_state)
        
        # Cross-Attention tương tác 2 chiều
        attn_T2I, _ = self.text_to_image_attn(H_text, H_image, H_image)
        H_text = self.norm_text(H_text + attn_T2I)
        
        attn_I2T, _ = self.image_to_text_attn(H_image, H_text, H_text, key_padding_mask=(mask == 0))
        H_image = self.norm_image(H_image + attn_I2T)
        
        # Pooling
        v_text = (H_text * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1e-9)
        v_image = H_image.mean(1)
        
        # Đặc trưng nhánh phục vụ ALIGNMENT
        z_text = nn.functional.normalize(self.align_proj(v_text), dim=-1)
        z_image = nn.functional.normalize(self.align_proj(v_image), dim=-1)
        
        # Gating Fusion phục vụ CLASSIFICATION
        g = torch.sigmoid(self.gate(torch.cat([v_text, v_image], dim=-1)))
        v_fusion = g * v_text + (1 - g) * v_image
        
        logits = self.classifier(v_fusion)
        return logits, z_text, z_image

# ==========================================
# 4. HÀM LOSS VÀ ĐO LƯỜNG (LOSS & METRICS)
# ==========================================
def contrastive_alignment_loss(z_text, z_image, temp=0.07):
    # Tính ma trận tương đồng giữa Text và Image
    similarity = torch.matmul(z_text, z_image.T) / temp
    labels = torch.arange(similarity.size(0)).to(z_text.device)
    
    # Loss theo cả 2 hướng: Text-to-Image và Image-to-Text
    loss_t2i = nn.functional.cross_entropy(similarity, labels)
    loss_i2t = nn.functional.cross_entropy(similarity.T, labels)
    return (loss_t2i + loss_i2t) / 2

def save_training_plots(train_losses, val_f1s):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Total Loss'); plt.title("Training Loss History"); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(val_f1s, label='F1 Macro', color='green'); plt.title("Validation F1 History"); plt.legend()
    plt.savefig("v9_1_alignment_results.png")
    plt.show()

# ==========================================
# 5. CHƯƠNG TRÌNH HUẤN LUYỆN (TRAINING)
# ==========================================
def main():
    # Chuẩn bị dữ liệu
    cache_data = get_or_create_cache()
    num_samples = len(cache_data['labels'])
    indices = torch.randperm(num_samples, generator=torch.Generator().manual_seed(42))
    
    train_idx = indices[:int(0.8 * num_samples)]
    val_idx = indices[int(0.8 * num_samples):int(0.9 * num_samples)]
    
    train_loader = DataLoader(FastAlignDataset(cache_data, train_idx), batch_size=CFG.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(FastAlignDataset(cache_data, val_idx), batch_size=CFG.BATCH_SIZE)

    model = UltimateModelV9_1().to(CFG.DEVICE)
    classification_loss_fn = nn.CrossEntropyLoss(label_smoothing=CFG.LABEL_SMOOTHING)
    
    best_f1_score = 0
    history_loss, history_f1 = [], []

    # Bắt đầu huấn luyện
    stages = [
        ("STAGE_1_FROZEN", CFG.STAGE_1_EPOCHS, CFG.LEARNING_RATE_STAGE_1),
        ("STAGE_2_UNFROZEN", CFG.STAGE_2_EPOCHS, CFG.LEARNING_RATE_STAGE_2)
    ]

    for stage_name, epochs, current_lr in stages:
        print(f"\n🚀 Bắt đầu {stage_name}...")
        if stage_name == "STAGE_2_UNFROZEN":
            for param in model.parameters(): param.requires_grad = True
        
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=current_lr)
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in progress_bar:
                ids = batch["input_ids"].to(CFG.DEVICE)
                mask = batch["attention_mask"].to(CFG.DEVICE)
                imgs = batch["image"].to(CFG.DEVICE)
                labels = batch["label"].to(CFG.DEVICE)
                
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    logits, z_text, z_image = model(ids, mask, imgs)
                    loss_cls = classification_loss_fn(logits, labels)
                    loss_align = contrastive_alignment_loss(z_text, z_image)
                    
                    # Tổng Loss: Phân loại + 0.1 * Căn chỉnh
                    total_loss = loss_cls + CFG.ALIGN_LAMBDA * loss_align
                
                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item()
                progress_bar.set_postfix(loss=total_loss.item())

            # Đánh giá sau mỗi Epoch
            model.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                for batch in val_loader:
                    logits, _, _ = model(batch["input_ids"].to(CFG.DEVICE), batch["attention_mask"].to(CFG.DEVICE), batch["image"].to(CFG.DEVICE))
                    y_true += batch["label"].tolist()
                    y_pred += logits.argmax(1).cpu().tolist()
            
            current_f1 = f1_score(y_true, y_pred, average="macro")
            history_loss.append(epoch_loss / len(train_loader))
            history_f1.append(current_f1)
            
            print(f"⭐ Epoch {epoch+1} Kết quả - F1 Macro: {current_f1:.4f}")
            
            if current_f1 > best_f1_score:
                best_f1_score = current_f1
                torch.save(model.state_dict(), "best_model_v9_1_alignment.pt")
                print(f"💾 Đã lưu Checkpoint tốt nhất (F1: {best_f1_score:.4f})")

    # Vẽ biểu đồ kết thúc
    save_training_plots(history_loss, history_f1)
    print(f"\n🎯 Hoàn tất huấn luyện V9.1! Best F1 Score đạt được: {best_f1_score:.4f}")

if __name__ == "__main__":
    main()