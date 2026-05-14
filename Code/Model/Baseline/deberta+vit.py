import os, torch, collections, gc
from tqdm import tqdm
from PIL import Image
from transformers import AutoTokenizer

# ==========================================
# 1. CONFIGURATION
# ==========================================
MAX_LEN = 128 
TOKENIZER_NAME = "microsoft/deberta-v3-base" 

DATA_DIR = "/kaggle/input/datasets/nhatnhat26/mvsa-dataset/MVSA"
WORK_DIR = "/kaggle/working/mvsa_clean"
os.makedirs(WORK_DIR, exist_ok=True)

def safe_read_text(path):
    for enc in ["utf-16", "utf-8", "latin-1"]:
        try:
            with open(path, "r", encoding=enc, errors="ignore") as f:
                content = f.read().strip()
                if content and len(content.split()) >= 3: 
                    return content
        except: pass
    return None

# 2. KHỞI TẠO TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

# 3. QUÉT VÀ LỌC DỮ LIỆU CHẶT CHẼ
valid_items = []
label_file = os.path.join(DATA_DIR, "labelResultAll.txt")
label_map = {"negative": 0, "neutral": 1, "positive": 2}

with open(label_file, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.read().splitlines()


for line in tqdm(lines[1:], desc="Filtering Data"):
    parts = line.strip().split()
    if len(parts) < 4: continue
    item_id = parts[0]
    
    try:

        votes = [parts[1].split(',')[0].lower(), 
                 parts[2].split(',')[0].lower(), 
                 parts[3].split(',')[0].lower()]
        label, count = collections.Counter(votes).most_common(1)[0]
        
        if count >= 2 and label in label_map:
            txt_path = os.path.join(DATA_DIR, "data", f"{item_id}.txt")
            img_path = os.path.join(DATA_DIR, "data", f"{item_id}.jpg")
            
            if os.path.exists(txt_path) and os.path.exists(img_path):
                
                try:
                    with Image.open(img_path) as img:
                        img.verify() 
                except:
                    continue
                
                text = safe_read_text(txt_path)
                if text:
                    enc = tokenizer(text, padding="max_length", truncation=True, max_length=MAX_LEN)
                    valid_items.append({
                        "ids": enc["input_ids"],
                        "masks": enc["attention_mask"],
                        "path": img_path,
                        "label": label_map[label]
                    })
    except Exception:
        continue

# 4. LƯU OFFLINE CACHE
if len(valid_items) > 0:
    torch.save({
        "ids": torch.tensor([x["ids"] for x in valid_items]),
        "masks": torch.tensor([x["masks"] for x in valid_items]),
        "paths": [x["path"] for x in valid_items],
        "labels": torch.tensor([x["label"] for x in valid_items]),
        "config": {"tokenizer": TOKENIZER_NAME, "max_len": MAX_LEN}
    }, f"{WORK_DIR}/mvsa_optimized_cache.pt")

    print(f"\n--- THỐNG KÊ KẾT QUẢ ---")
    print(f" Số lượng mẫu hợp lệ cuối cùng: {len(valid_items)}")
    print(f" Cache đã lưu tại: {WORK_DIR}/mvsa_optimized_cache.pt")
    
    # Tính toán tỷ lệ nhãn để bạn đưa vào Khóa luận
    label_counts = collections.Counter([x['label'] for x in valid_items])
    for lab, idx in label_map.items():
        print(f"   + {lab.capitalize()}: {label_counts[idx]} mẫu")
else:
    print(" Không tìm thấy mẫu nào hợp lệ sau khi lọc!")
##
import torch, gc, os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, get_linear_schedule_with_warmup
import torchvision.transforms as T
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report

# ==========================================
# 1. SETUP THIẾT BỊ (MULTI-GPU)
# ==========================================
NUM_GPUS = torch.cuda.device_count()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hệ thống phát hiện {NUM_GPUS} GPU(s). Sử dụng thiết bị: {DEVICE}")

# ==========================================
# 2. DATASET VỚI LAZY LOADING
# ==========================================
class OptimizedMVSADataset(Dataset):
    def __init__(self, data_dict, idx, is_train=False):
        self.ids = data_dict["ids"][idx]
        self.masks = data_dict["masks"][idx]
        self.labels = data_dict["labels"][idx]
        self.paths = [data_dict["paths"][i] for i in idx]
        
        # Chuẩn hóa chuẩn ViT (ImageNet)
        norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if is_train:
            self.transform = T.Compose([
                T.Resize((224, 224)), T.RandomHorizontalFlip(), 
                T.ColorJitter(0.1, 0.1, 0.1), T.ToTensor(), norm
            ])
        else:
            self.transform = T.Compose([T.Resize((224, 224)), T.ToTensor(), norm])

    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        try: img = Image.open(self.paths[i]).convert("RGB")
        except: img = Image.new("RGB", (224, 224), (0, 0, 0))
        return self.ids[i], self.masks[i], self.transform(img), self.labels[i]

# ==========================================
# 3. MÔ HÌNH DUAL-WAY CO-ATTENTION 
# ==========================================
class BestComboModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(cfg['tokenizer'])
        self.img_model = AutoModel.from_pretrained("google/vit-base-patch16-224-in21k")
        
        # Đóng băng 
        for p in self.text_model.parameters(): p.requires_grad = False
        for p in self.img_model.parameters(): p.requires_grad = False

        # Dual-Way Co-Attention Depth = 2
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "t2i": nn.MultiheadAttention(768, 8, batch_first=True),
                "i2t": nn.MultiheadAttention(768, 8, batch_first=True),
                "nT": nn.LayerNorm(768), "nI": nn.LayerNorm(768)
            }) for _ in range(2)
        ])
        
        self.gate = nn.Linear(1536, 768)
        self.classifier = nn.Sequential(
            nn.Linear(768, 512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512, 3)
        )

    def unfreeze_partial(self):
        for n, p in self.text_model.named_parameters():
            if "encoder.layer." in n:
                lid = int(n.split(".")[2])
                if lid >= 6: p.requires_grad = True

    def forward(self, ids, mask, img):
        # Ép kiểu dữ liệu sang Float32 để tránh lỗi Half != Float trên Multi-GPU
        img = img.to(torch.float32)
        
        # 1. Trích xuất đặc trưng thô
        HT = self.text_model(ids, attention_mask=mask).last_hidden_state.to(torch.float32)
        HI = self.img_model(pixel_values=img).last_hidden_state.to(torch.float32)
        
        km = (mask == 0)
        # 2. Cơ chế đối thoại chéo 2 chiều
        for l in self.layers:
            # Nhánh 1: Văn bản soi ảnh
            t2i, _ = l["t2i"](HT, HI, HI)
            HT = l["nT"](HT + t2i)
            # Nhánh 2: Ảnh soi văn bản
            i2t, _ = l["i2t"](HI, HT, HT, key_padding_mask=km)
            HI = l["nI"](HI + i2t)
            
        # 3. Pooling (Mean)
        m = mask.unsqueeze(-1).float()
        vT = (HT * m).sum(1) / torch.clamp(m.sum(1), 1e-9)
        vI = HI.mean(1)
        
        # 4. Gated Fusion
        combined = torch.cat([vT, vI], -1)
        gate_score = torch.sigmoid(self.gate(combined))
        v_fuse = gate_score * vT + (1 - gate_score) * vI
        
        return self.classifier(v_fuse)

# ==========================================
# Test
# ==========================================
def evaluate_full(model, loader):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for ids, mask, img, y in loader:
            out = model(ids.to(DEVICE), mask.to(DEVICE), img.to(DEVICE))
            preds.extend(out.argmax(1).cpu().tolist())
            truths.extend(y.tolist())
    
    return {
        "acc": accuracy_score(truths, preds),
        "pre": precision_score(truths, preds, average='macro', zero_division=0),
        "rec": recall_score(truths, preds, average='macro', zero_division=0),
        "f1_macro": f1_score(truths, preds, average='macro'),
        "f1_weight": f1_score(truths, preds, average='weighted'),
        "report": classification_report(truths, preds, target_names=['Neg', 'Neu', 'Pos'], digits=4)
    }

# ==========================================
# 5. EXECUTION (TRAIN 8:1:1)
# ==========================================

CACHE_PATH = "/kaggle/working/mvsa_clean/mvsa_optimized_cache.pt"
if not os.path.exists(CACHE_PATH):
    CACHE_PATH = "/kaggle/working/mvsa_optimized_cache.pt"

cache = torch.load(CACHE_PATH)
idx = torch.randperm(len(cache["labels"]), generator=torch.Generator().manual_seed(42))
tr_idx, vl_idx, ts_idx = idx[:int(0.8*len(idx))], idx[int(0.8*len(idx)):int(0.9*len(idx))], idx[int(0.9*len(idx)):]

# 
BS = 16 * NUM_GPUS if NUM_GPUS > 1 else 16
train_loader = DataLoader(OptimizedMVSADataset(cache, tr_idx, is_train=True), batch_size=BS, shuffle=True)
val_loader = DataLoader(OptimizedMVSADataset(cache, vl_idx), batch_size=BS)
test_loader = DataLoader(OptimizedMVSADataset(cache, ts_idx), batch_size=BS)

model = BestComboModel(cache['config']).float() 

if NUM_GPUS > 1:
    model = nn.DataParallel(model) 
model.to(DEVICE)

# --- STAGE 1: WARM-UP (5 Epochs) ---
print("\n" + "="*50 + "\nSTAGE 1: FROZEN BACKBONE (Warm-up)\n" + "="*50)
opt1 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)

for ep in range(10):
    model.train()
    total_loss = 0
    for ids, mask, img, y in tqdm(train_loader, desc=f"S1-E{ep+1}"):
        opt1.zero_grad()
        out = model(ids.to(DEVICE), mask.to(DEVICE), img.to(DEVICE))
        loss = nn.functional.cross_entropy(out, y.to(DEVICE))
        loss.backward()
        opt1.step()
        total_loss += loss.item()
    print(f"   => Average Loss: {total_loss/len(train_loader):.4f}")

# --- STAGE 2: PARTIAL FINE-TUNING (10 Epochs) ---
actual_model = model.module if NUM_GPUS > 1 else model
actual_model.unfreeze_partial()

print("\n" + "="*50 + "\n STAGE 2: PARTIAL FINE-TUNING (Last 6 Layers)\n" + "="*50)
opt2 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-6)

best_f1 = 0
for ep in range(20):
    model.train()
    for ids, mask, img, y in tqdm(train_loader, desc=f"S2-E{ep+1}"):
        opt2.zero_grad()
        out = model(ids.to(DEVICE), mask.to(DEVICE), img.to(DEVICE))
        loss = nn.functional.cross_entropy(out, y.to(DEVICE))
        loss.backward()
        opt2.step()
    
    m = evaluate_full(model, val_loader)
    print(f" Ep{ep+1} | F1-Mac: {m['f1_macro']:.4f} | F1-Wei: {m['f1_weight']:.4f} | Acc: {m['acc']:.4f}")
    
    if m["f1_macro"] > best_f1:
        best_f1 = m["f1_macro"]
        torch.save(actual_model.state_dict(), "best_model_combo.pt")
        print(f"Saved best checkpoint (F1: {best_f1:.4f})")

# --- FINAL EVALUATION ON TEST SET 
print("\n" + "="*50 + "\n FINAL TEST RESULTS (UNSEEN DATA)\n" + "="*50)
actual_model.load_state_dict(torch.load("best_model_combo.pt"))
res = evaluate_full(model, test_loader)

print(f"Accuracy     : {res['acc']:.4f}")
print(f"F1 Macro     : {res['f1_macro']:.4f}")
print(f"F1 Weighted  : {res['f1_weight']:.4f}")
print(f"Precision (M): {res['pre']:.4f}")
print(f"Recall (M)   : {res['rec']:.4f}")
print("\nDetailed Report:\n", res['report'])

