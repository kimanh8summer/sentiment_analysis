import os, torch, collections, random, numpy as np, pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, ViTModel
import torchvision.transforms as T
from sklearn.metrics import (f1_score, confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import label_binarize

# --- 1. CẤU HÌNH HỆ THỐNG ---
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
    CSV_PATH = "/kaggle/input/datasets/thkimanhnguyn/cp-csv/mvsa_clean/mvsa.csv"
    IMG_DIR = "/kaggle/input/datasets/thkimanhnguyn/mvsa-dataset-raw/MVSA/data"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 16
    STAGE_1_EPOCHS = 5
    STAGE_2_EPOCHS = 10
    LEARNING_RATE_S1 = 2e-5
    LEARNING_RATE_S2 = 1e-6

# Cấu hình V9: Sử dụng CLS token, 2 lớp Cross-Attention, CE Loss + Smoothing, SupCon và Partial FT
V9_CONFIG = {
    "name": "V9_supcon",
    "USE_CLS": True,
    "CROSS": 2,
    "LOSS": "ce",
    "SMOOTH": 0.1,
    "CONTRAST": True,
    "SUPCON": True,
    "PARTIAL": True
}

# --- 2. DATASET & PREPROCESSING ---
class MVSADataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Load Image
        try:
            image = Image.open(os.path.join(self.img_dir, row["image"])).convert("RGB")
        except:
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        if self.transform: image = self.transform(image)
        
        # Tokenize Text
        enc = self.tokenizer(str(row["text"]), padding="max_length", truncation=True, max_length=64, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "image": image,
            "label": torch.tensor(int(row["label"]))
        }

# --- 3. KIẾN TRÚC MÔ HÌNH V9 ---
class UltimateModelV9(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.img_proj = nn.Linear(768, 768)
        
        # Đóng băng mặc định
        for p in self.bert.parameters(): p.requires_grad = False
        for p in self.vit.parameters(): p.requires_grad = False
        
        # Partial Fine-tuning: Mở khóa từ layer 6 của BERT
        if cfg.get("PARTIAL", False):

            # mở BERT layer 6-11
            for name, p in self.bert.named_parameters():
                if "encoder.layer." in name:
                    layer_id = int(name.split("encoder.layer.")[1].split(".")[0])
                    if layer_id >= 6:
                        p.requires_grad = True
        
            # mở ViT layer 8-11
            for name, p in self.vit.named_parameters():
                if "encoder.layer." in name:
                    layer_id = int(name.split("encoder.layer.")[1].split(".")[0])
                    if layer_id >= 8:
                        p.requires_grad = True
                        
        
        # Cross-Attention Modules
        self.t2i = nn.MultiheadAttention(768, 8, batch_first=True)
        self.i2t = nn.MultiheadAttention(768, 8, batch_first=True)
        self.normT = nn.LayerNorm(768)
        self.normI = nn.LayerNorm(768)
        
        self.gate = nn.Linear(768 * 2, 768)
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 3)
        )
        # Projection head cho Contrastive/SupCon
        self.proj = nn.Linear(768, 256)

    def unfreeze_all(self):

        # full FT cho BERT
        for p in self.bert.parameters():
            p.requires_grad = True
    
        # freeze lại toàn bộ ViT
        for p in self.vit.parameters():
            p.requires_grad = False
    
        # chỉ mở last 4 blocks của ViT
        for name, p in self.vit.named_parameters():
            if "encoder.layer." in name:
                layer_id = int(name.split("encoder.layer.")[1].split(".")[0])
                if layer_id >= 8:
                    p.requires_grad = True

    def forward(self, ids, mask, img):
        H_T = self.bert(ids, attention_mask=mask).last_hidden_state
        H_I = self.img_proj(self.vit(pixel_values=img).last_hidden_state)
        
        # Cross-Attention 
        T2I, _ = self.t2i(H_T, H_I, H_I)
        H_T = self.normT(H_T + T2I)
        I2T, _ = self.i2t(H_I, H_T, H_T, key_padding_mask=(mask == 0))
        H_I = self.normI(H_I + I2T)
        
        # Fusion
        v_T = (H_T * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1e-9)
        v_I = H_I.mean(1)
        g = torch.sigmoid(self.gate(torch.cat([v_T, v_I], dim=-1)))
        v_fuse = g * v_T + (1 - g) * v_I
        
        logits = self.classifier(v_fuse)
        z = nn.functional.normalize(self.proj(v_fuse), dim=-1)
        return logits, z

# --- 4. LOSS & VISUALIZATION FUNCTIONS ---
def supcon_loss(z, labels, temp=0.07):
    # Tránh lỗi khi batch size nhỏ
    z = torch.cat([z, z], dim=0)
    sim = torch.matmul(z, z.T) / temp
    mask = torch.eye(sim.size(0), dtype=torch.bool).to(z.device)
    sim.masked_fill_(mask, -1e4) # Tránh log(0) và tràn số
    labels = torch.cat([labels, labels], dim=0).unsqueeze(1)
    pos_mask = (labels == labels.T).float() * (~mask)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    loss = -(pos_mask * log_prob).sum(1) / pos_mask.sum(1).clamp(min=1)
    return loss.mean()

def save_plots(exp_name, train_losses, val_f1s, y_true, y_pred, y_probs):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(20, 12))
    
    # 1. Loss Curve
    plt.subplot(2, 2, 1); plt.plot(train_losses, label='Train Loss', color='b'); plt.title(f"{exp_name} - Loss History"); plt.legend()
    # 2. F1 Curve
    plt.subplot(2, 2, 2); plt.plot(val_f1s, label='Val F1 Macro', color='g'); plt.title(f"{exp_name} - F1 Macro History"); plt.legend()
    # 3. Confusion Matrix
    plt.subplot(2, 2, 3)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Neg','Neu','Pos'], yticklabels=['Neg','Neu','Pos'])
    plt.title("Confusion Matrix")
    # 4. ROC Curve
    plt.subplot(2, 2, 4)
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    for i, label in enumerate(['Neg', 'Neu', 'Pos']):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        plt.plot(fpr, tpr, label=f'{label} (AUC = {auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], 'k--'); plt.title("ROC Curves"); plt.legend()
    
    plt.savefig(f"{exp_name}_metrics.png")
    plt.show()

# --- 5. MAIN TRAINING LOOP ---
def main():
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    dataset = MVSADataset(CFG.CSV_PATH, CFG.IMG_DIR, transform)
    
    # Chia dataset (80/10/10)
    n = len(dataset)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
    train_loader = DataLoader(torch.utils.data.Subset(dataset, indices[:int(0.8*n)]), batch_size=CFG.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, indices[int(0.8*n):int(0.9*n)]), batch_size=CFG.BATCH_SIZE)

    model = UltimateModelV9(V9_CONFIG).to(CFG.DEVICE)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=V9_CONFIG["SMOOTH"])
    
    train_losses, val_f1s = [], []
    best_f1 = 0

    # Stage 1: Freeze backbone, Stage 2: Unfreeze All
    for stage, epochs, lr in [("S1", CFG.STAGE_1_EPOCHS, CFG.LEARNING_RATE_S1), ("S2", CFG.STAGE_2_EPOCHS, CFG.LEARNING_RATE_S2)]:
        if stage == "S2": model.unfreeze_all()
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        
        for ep in range(epochs):
            model.train(); total_loss = 0
            for b in tqdm(train_loader, desc=f"V9-{stage}-Ep{ep+1}"):
                ids, mask, img, y = b["input_ids"].to(CFG.DEVICE), b["attention_mask"].to(CFG.DEVICE), b["image"].to(CFG.DEVICE), b["label"].to(CFG.DEVICE)
                opt.zero_grad()
                with torch.amp.autocast('cuda'):
                    logits, z = model(ids, mask, img)
                    loss = loss_fn(logits, y) + 0.1 * supcon_loss(z, y)
                loss.backward(); opt.step(); total_loss += loss.item()
            
            # Validation
            model.eval(); y_t, y_p = [], []
            with torch.no_grad():
                for b in val_loader:
                    ids, mask, img, y = b["input_ids"].to(CFG.DEVICE), b["attention_mask"].to(CFG.DEVICE), b["image"].to(CFG.DEVICE), b["label"].to(CFG.DEVICE)
                    logits, _ = model(ids, mask, img)
                    y_t += y.cpu().tolist(); y_p += logits.argmax(1).cpu().tolist()
            
            f1 = f1_score(y_t, y_p, average="macro")
            train_losses.append(total_loss/len(train_loader)); val_f1s.append(f1)
            print(f"Epoch {ep+1} - F1: {f1:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), "best_V9_supcon.pt")

    # Final Visualization
    model.load_state_dict(torch.load("best_V9_supcon.pt"))
    model.eval(); y_t, y_p, y_probs = [], [], []
    with torch.no_grad():
        for b in val_loader:
            ids, mask, img, y = b["input_ids"].to(CFG.DEVICE), b["attention_mask"].to(CFG.DEVICE), b["image"].to(CFG.DEVICE), b["label"].to(CFG.DEVICE)
            logits, _ = model(ids, mask, img)
            y_t += y.cpu().tolist(); y_p += logits.argmax(1).cpu().tolist(); y_probs += torch.softmax(logits, 1).cpu().tolist()
    
    save_plots("V9_SupCon", train_losses, val_f1s, y_t, y_p, np.array(y_probs))
    print(f"🎯 V9 Hoàn tất! Best F1: {best_f1:.4f}")

if __name__ == "__main__": main()

