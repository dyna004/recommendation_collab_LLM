import sys
import os
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. CẤU HÌNH PATH & LIBS
# ==========================================
BASE_PATH = "D:/Ngôn ngữ tự nhiên/BERT4Rec-VAE-Pytorch"
if BASE_PATH not in sys.path:
    sys.path.append(BASE_PATH)

try:
    from models.bert import BERTModel
except ImportError:
    print(f"Lỗi: Không tìm thấy source code BERT4Rec tại: {BASE_PATH}")
    sys.exit()

class ConfigNamespace:
    def __init__(self, cfg, num_items):
        # 1. Load thuộc tính gốc
        for k, v in cfg.items(): setattr(self, k, v)
        self.num_items = num_items
        
        # 2. Map các biến cần thiết cho BERTModel 
        self.bert_max_len = getattr(self, 'max_len', 100)
        self.bert_num_blocks = getattr(self, 'num_layers', 2)
        self.bert_num_heads = getattr(self, 'num_heads', 4)
        self.bert_hidden_units = getattr(self, 'hidden_size', 256)
        self.bert_dropout = getattr(self, 'dropout', 0.1)

# ==========================================
# 2. LOAD DATA & MODEL
# ==========================================
def load_resources():
    print("[1/4] Đang tải tài nguyên và Model...")
    
    # Định nghĩa đường dẫn theo cấu trúc của bạn
    data_path = f"{BASE_PATH}/data/ml-1m.rating"
    movies_path = f"{BASE_PATH}/data/movies.dat"
    # Đường dẫn config/model theo app.py
    config_path = f"{BASE_PATH}/Test/config.json"
    ckpt_path = f"{BASE_PATH}/Test/Model/best_acc_model.pth"

    # --- A. LOAD DATA & MAPPING ID ---
    try:
        df = pd.read_csv(data_path, sep="::", header=None, engine="python", names=['uid', 'mid', 'rating', 'timestamp'])
    except FileNotFoundError:
        print(f"Không tìm thấy file data tại: {data_path}")
        sys.exit()

    # Tạo mapping ID (Logic bắt buộc của BERT4Rec train)
    item_ids = df['mid'].unique()
    item_ids.sort()
    val2idx = {item_id: i + 1 for i, item_id in enumerate(item_ids)}
    idx2val = {i + 1: item_id for i, item_id in enumerate(item_ids)}
    num_items = len(item_ids)

    # --- B. LOAD MOVIE TEXT (TITLE + GENRE) ---
    try:
        movies_df = pd.read_csv(
            movies_path, sep="::", header=None, engine="python",
            names=["movie_id", "title", "genres"], encoding="latin-1"
        )
    except FileNotFoundError:
        print(f"Không tìm thấy file movies tại: {movies_path}")
        sys.exit()
    
    id2text = {}
    for _, row in movies_df.iterrows():
        # Kết hợp Tên + Thể loại để tạo ngữ cảnh semantic
        clean_genres = row['genres'].replace('|', ' ')
        text_repr = f"{row['title']} {clean_genres}"
        id2text[row['movie_id']] = text_repr

    # --- C. LOAD BERT MODEL ---
    try:
        with open(config_path) as f: cfg = json.load(f)
    except FileNotFoundError:
        print(f"Không tìm thấy config tại: {config_path}")
        sys.exit()

    config = ConfigNamespace(cfg, num_items)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    -> Running on: {device}")
    
    bert_model = BERTModel(config).to(device)
    
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        bert_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print("    -> Đã load weight model thành công.")
    else:
        print(f"Không tìm thấy model checkpoint tại: {ckpt_path}")
        sys.exit()
        
    bert_model.eval()

    # --- D. LOAD SEMANTIC MODEL ---
    print("    -> Đang tải model ngôn ngữ (all-MiniLM-L6-v2)...")
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

    return bert_model, semantic_model, config, df, val2idx, idx2val, id2text, device

# ==========================================
# 3. CHẠY ĐÁNH GIÁ
# ==========================================
def run_semantic_eval():
    bert_model, semantic_model, config, df, val2idx, idx2val, id2text, device = load_resources()
    
    print("-" * 60)
    print("BẮT ĐẦU ĐÁNH GIÁ SEMANTIC SIMILARITY")
    print("   Mục tiêu: Kiểm tra xem Model có gợi ý đúng 'GU' (nội dung) hay không.")
    print("-" * 60)

    # --- CHUẨN BỊ DỮ LIỆU TEST ---
    print("[2/4] Chuẩn bị dữ liệu Test (Leave-One-Out)...")
    user_groups = df.sort_values('timestamp').groupby('uid')['mid'].apply(list)
    
    test_data = []
    # Lấy mẫu 1000 user để test cho nhanh (chạy hết 6000 user sẽ lâu)
    sampled_uids = list(user_groups.keys())[:1000]
    
    for uid in sampled_uids:
        movies = user_groups[uid]
        if len(movies) < 5: continue
        
        # Convert ID gốc -> Model ID
        seq_ids = [val2idx[mid] for mid in movies if mid in val2idx]
        if len(seq_ids) < 2: continue

        # Input: Tất cả trừ cái cuối. Target: Cái cuối cùng.
        input_seq = seq_ids[:-1]
        target_item = seq_ids[-1]
        test_data.append((input_seq, target_item))

    # --- CHẠY INFERENCE ---
    print(f"[3/4] Đang tính toán trên {len(test_data)} mẫu...")
    
    sim_scores = []
    exact_matches = 0
    max_len = getattr(config, 'bert_max_len', 100)

    with torch.no_grad():
        for input_seq, target_idx in tqdm(test_data, desc="Processing"):
            # 1. Pad Sequence & Predict
            seq_len = len(input_seq)
            if seq_len > max_len:
                model_input = input_seq[-max_len:]
            else:
                model_input = input_seq + [0] * (max_len - seq_len)
            
            input_tensor = torch.tensor([model_input]).to(device)
            logits = bert_model(input_tensor)
            
            # Lấy vector dự đoán ở vị trí cuối cùng
            last_pos = min(seq_len, max_len) - 1
            preds = logits[0, last_pos, :]
            preds[0] = -float('inf') # Mask padding
            
            # Lấy Top 10 gợi ý
            _, topk_indices = torch.topk(preds, 10)
            topk_indices = topk_indices.cpu().numpy()
            
            # 2. Lấy nội dung text (Title + Genre)
            # - Text của phim mục tiêu (Target)
            target_raw_id = idx2val.get(target_idx)
            target_text = id2text.get(target_raw_id, "")
            
            if not target_text: continue 

            # - Text của 10 phim gợi ý
            rec_texts = []
            for item_idx in topk_indices:
                raw_id = idx2val.get(item_idx)
                text = id2text.get(raw_id, "")
                if text: rec_texts.append(text)
            
            if not rec_texts: continue

            # 3. Tính điểm tương đồng (Semantic Similarity)
            target_emb = semantic_model.encode([target_text]) # Vector Target
            rec_embs = semantic_model.encode(rec_texts)       # Vectors Gợi ý
            
            # So sánh Target với 10 gợi ý -> Lấy điểm cao nhất
            similarities = cosine_similarity(target_emb, rec_embs)[0]
            max_sim = np.max(similarities)
            sim_scores.append(max_sim)
            
            # Check Exact Match (Trúng phóc ID)
            if target_idx in topk_indices:
                exact_matches += 1

    # --- BÁO CÁO KẾT QUẢ ---
    avg_sim = np.mean(sim_scores)
    hit_ratio = exact_matches / len(test_data)
    
    print("\n" + "="*50)
    print("KẾT QUẢ ĐÁNH GIÁ CUỐI CÙNG")
    print("="*50)
    print(f" Số mẫu test           : {len(test_data)}")
    print(f"Hit Ratio (Exact ID)  : {hit_ratio:.4f} ({hit_ratio*100:.2f}%)")
    print(f"Avg Semantic Score    : {avg_sim:.4f} (Thang điểm -1 đến 1)")
    print("-" * 50)

if __name__ == "__main__":
    run_semantic_eval()