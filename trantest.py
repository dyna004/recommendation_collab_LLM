import sys
import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ==========================================
# 1. CẤU HÌNH
# ==========================================
CONFIG = {
    "max_len": 100,
    "hidden_size": 256,    
    "num_heads": 8,        
    "num_layers": 2,       
    "dropout": 0.1,
    "batch_size": 128,      
    "lr": 0.001,            
    "epochs": 100,           
    "mask_prob": 0.15,
    "model_init_seed": 42,
    "num_items": 0          
}

# Setup đường dẫn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Import Model
try:
    from models.bert import BERTModel
except ImportError:
    print("Lỗi: Không tìm thấy folder 'models'.")
    sys.exit()

# ==========================================
# 2. CLASS CONFIG "BẤT TỬ" (UPDATED)
# ==========================================
class RobustConfig:
    def __init__(self, d):
        # 1. Load key gốc
        for k, v in d.items(): 
            setattr(self, k, v)
        
        # 2. MAP CÁC BIẾN ĐỒNG NGHĨA (FULL COVERAGE)
        
        # --- HIDDEN SIZE ---
        val_hidden = d.get('hidden_size', 128)
        self.hidden = val_hidden
        self.hidden_size = val_hidden
        self.embed_size = val_hidden
        self.bert_hidden_units = val_hidden # <--- Fix 
        
        # --- LAYERS (BLOCKS) ---
        val_layers = d.get('num_layers', 2)
        self.n_layers = val_layers
        self.num_layers = val_layers
        self.transformer_blocks = val_layers
        self.bert_num_blocks = val_layers   # <--- FIX 
        
        # --- HEADS ---
        val_heads = d.get('num_heads', 4)
        self.attn_heads = val_heads
        self.num_heads = val_heads
        self.bert_num_heads = val_heads     # <--- Fix
        
        # --- LENGTH ---
        val_len = d.get('max_len', 50)
        self.max_len = val_len
        self.seq_len = val_len
        self.bert_max_len = val_len 
        
        # --- ITEMS / VOCAB ---
        val_items = d.get('num_items', 0)
        self.num_items = val_items
        self.vocab_size = val_items + 2 
        
        # --- DROPOUT ---
        val_drop = d.get('dropout', 0.1)
        self.dropout = val_drop
        self.bert_dropout = val_drop        # <--- Fix

# ==========================================
# 3. DATASET
# ==========================================
class BERT4RecDataset(Dataset):
    def __init__(self, user_seqs, num_items, max_len, mask_prob):
        self.user_seqs = user_seqs
        self.num_items = num_items
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = num_items + 1 

    def __len__(self):
        return len(self.user_seqs)

    def __getitem__(self, index):
        seq = self.user_seqs[index]
        if len(seq) > self.max_len:
            seq = seq[-self.max_len:]
        else:
            seq = [0] * (self.max_len - len(seq)) + seq

        tokens = []
        labels = []
        for item in seq:
            if item == 0:
                tokens.append(0)
                labels.append(0)
                continue
            prob = random.random()
            if prob < self.mask_prob:
                tokens.append(self.mask_token)
                labels.append(item)
            else:
                tokens.append(item)
                labels.append(0)
        return torch.tensor(tokens), torch.tensor(labels)

# ==========================================
# 4. TRAINING LOOP
# ==========================================
def train():
    print("[1/5] Đang đọc dữ liệu...")
    data_path = os.path.join(BASE_DIR, "data", "ml-1m.rating")
    
    try:
        df = pd.read_csv(data_path, sep="::", header=None, engine="python", names=['uid', 'mid', 'rating', 'timestamp'])
    except FileNotFoundError:
        print(f"Không tìm thấy file tại: {data_path}")
        sys.exit()

    # Re-map ID
    item_ids = df['mid'].unique()
    num_items = len(item_ids)
    item_ids.sort()
    val2idx = {item_id: i + 1 for i, item_id in enumerate(item_ids)}
    df['mid'] = df['mid'].map(val2idx)
    
    print(f"    -> Số lượng Items: {num_items}")
    CONFIG['num_items'] = num_items

    print("[2/5] Tạo chuỗi huấn luyện...")
    user_groups = df.sort_values('timestamp').groupby('uid')['mid'].apply(list)
    train_seqs = []
    for _, movies in user_groups.items():
        if len(movies) < 5: continue
        train_seqs.append(movies)

    dataset = BERT4RecDataset(train_seqs, num_items, CONFIG['max_len'], CONFIG['mask_prob'])
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)

    print("[3/5] Khởi tạo Mô hình...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    -> Running on: {device}")

    # Dùng class Config mới
    model_cfg = RobustConfig(CONFIG)
    
    try:
        model = BERTModel(model_cfg).to(device)
        print("    -> Model thành công!")
    except Exception as e:
        print("    ->  Model thất bại!")
        print(f"    ERROR: {e}")
        # Debug attributes
        print("    [DEBUG] Checking RobustConfig attributes:")
        print(f"    - bert_num_blocks: {getattr(model_cfg, 'bert_num_blocks', 'MISSING')}")
        print(f"    - bert_num_heads: {getattr(model_cfg, 'bert_num_heads', 'MISSING')}")
        print(f"    - bert_hidden_units: {getattr(model_cfg, 'bert_hidden_units', 'MISSING')}")
        sys.exit()

    criterion = nn.CrossEntropyLoss(ignore_index=0) 
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])

    print(f"[4/5] Bắt đầu Train ({CONFIG['epochs']} epochs)...")
    model.train()
    
    try:
        for epoch in range(CONFIG['epochs']):
            total_loss = 0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
            
            for tokens, labels in pbar:
                tokens, labels = tokens.to(device), labels.to(device)
                
                optimizer.zero_grad()
                logits = model(tokens) 
                
                logits = logits.view(-1, logits.size(-1))
                labels = labels.view(-1)
                
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f"{total_loss/(len(pbar)+1):.4f}"})
    except KeyboardInterrupt:
        print("\n[STOP] Đã dừng train thủ công.")

    print("[5/5] Lưu kết quả...")
    output_dir = os.path.join(BASE_DIR, "MyModel")
    os.makedirs(output_dir, exist_ok=True)
    
    model_save_path = os.path.join(output_dir, "my_best_model.pth")
    torch.save({"model_state_dict": model.state_dict()}, model_save_path)
    
    # Lưu config gốc
    config_save_path = os.path.join(output_dir, "config.json")
    with open(config_save_path, "w") as f:
        json.dump(CONFIG, f, indent=4)
        
    print("\n HUẤN LUYỆN HOÀN TẤT!")
    print(f"-> Model saved at: {model_save_path}")

if __name__ == "__main__":
    train()