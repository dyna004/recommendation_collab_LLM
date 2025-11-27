import sys
import os
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# 1. SETUP & CONFIG
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

try:
    from models.bert import BERTModel
except ImportError:
    print("Lỗi: Không tìm thấy folder 'models'. Hãy đảm bảo bạn đang chạy file ở thư mục gốc dự án.")
    sys.exit()

class EvalConfig:
    def __init__(self, config_dict, num_items):
        # Load các key gốc từ json
        for k, v in config_dict.items():
            setattr(self, k, v)
        self.num_items = num_items
        
        # MAPPING: Đảm bảo tên biến khớp với BERTModel (quan trọng)
        self.hidden = self.hidden_size
        self.embed_size = self.hidden_size
        self.bert_hidden_units = self.hidden_size
        self.n_layers = self.num_layers
        self.transformer_blocks = self.num_layers
        self.bert_num_blocks = self.num_layers
        self.attn_heads = self.num_heads
        self.bert_num_heads = self.num_heads
        self.seq_len = self.max_len
        self.bert_max_len = self.max_len
        
        # Vocab size = Items + Padding (0) + Mask Token
        # Để an toàn, +2 để cover index của Mask Token
        self.vocab_size = self.num_items + 2 
        self.bert_dropout = getattr(self, 'dropout', 0.1)

# ==========================================
# 2. HÀM LOAD RESOURCES
# ==========================================
def load_resources():
    print("[1/4] Loading Resources...")
    data_path = os.path.join(BASE_DIR, "data", "ml-1m.rating")
    config_path = os.path.join(BASE_DIR, "MyModel", "config.json")
    ckpt_path = os.path.join(BASE_DIR, "MyModel", "my_best_model.pth") 

    try:
        df = pd.read_csv(data_path, sep="::", header=None, engine="python", names=['uid', 'mid', 'rating', 'timestamp'])
    except FileNotFoundError:
        print(f"Không tìm thấy data tại: {data_path}")
        sys.exit()
    
    # --- LOGIC MAPPING ID ---
    item_ids = df['mid'].unique()
    item_ids.sort() 
    num_items = len(item_ids)
    val2idx = {item_id: i + 1 for i, item_id in enumerate(item_ids)}
    df['mid'] = df['mid'].map(val2idx)
    
    # Load Config
    if not os.path.exists(config_path):
        print("Không tìm thấy config.json")
        sys.exit()

    with open(config_path) as f:
        raw_cfg = json.load(f)

    config = EvalConfig(raw_cfg, num_items)
    
    # Init Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    -> Device: {device}")
    
    model = BERTModel(config).to(device)
    
    # Load Weights
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print("    -> Model loaded successfully!")
    else:
        print(f"Không tìm thấy model tại {ckpt_path}")
        sys.exit()
        
    model.eval()
    return model, config, df, device

# ==========================================
# 3. HÀM TÍNH METRICS (FULL RANKING)
# ==========================================
def calculate_full_ranking_metrics(logits, targets, history_seqs, mask_token, k_list=[5, 10, 20]):
    """
    Tính R@K và NDCG@K cho toàn bộ Items.
    - logits: [batch_size, vocab_size] (Điểm số dự đoán cho mọi item)
    - targets: [batch_size] (Item thực tế user đã xem)
    - history_seqs: [batch_size, seq_len] (Lịch sử xem để lọc bỏ, không gợi ý lại)
    """
    results = {f'R@{k}': 0.0 for k in k_list}
    results.update({f'N@{k}': 0.0 for k in k_list})
    
    batch_size = logits.size(0)

    # 1. MASKING HISTORY (Lọc bỏ phim đã xem)
    # Clone để không ảnh hưởng dữ liệu gốc
    history_for_filter = history_seqs.clone()
    
    # [QUAN TRỌNG]: Thay thế Mask Token bằng 0 (Padding).
    # Vì Mask Token có ID lớn nhất, nếu dùng làm index cho scatter có thể gây lỗi hoặc sai logic.
    history_for_filter[history_for_filter == mask_token] = 0

    # Gán điểm -inf cho các index có trong lịch sử -> Không bao giờ lọt top
    # scatter_(dim, index, src)
    logits.scatter_(1, history_for_filter, float('-inf'))
    
    # Đảm bảo padding (index 0) luôn là -inf
    logits[:, 0] = float('-inf')

    # 2. SORTING (Lấy Ranking)
    # descending=True: Điểm cao nhất đứng đầu
    _, sorted_indices = torch.sort(logits, descending=True) # [B, V]
    
    # Chuyển về CPU để tính toán numpy cho nhanh & an toàn
    sorted_indices = sorted_indices.cpu().numpy()
    targets = targets.cpu().numpy()
    
    # 3. TÍNH METRICS
    for i in range(batch_size):
        target_item = targets[i]
        
        # Tìm vị trí (rank) của target trong danh sách sorted
        # np.where trả về tuple array, lấy phần tử đầu tiên
        # Cộng 1 vì index bắt đầu từ 0
        try:
            # Tìm index của target trong mảng đã sort
            rank = np.where(sorted_indices[i] == target_item)[0][0] + 1
        except IndexError:
            # Trường hợp hiếm: target bị filter mất (do data lỗi trùng lặp target vào history)
            continue
            
        for k in k_list:
            if rank <= k:
                results[f'R@{k}'] += 1
                # NDCG công thức: 1 / log2(rank + 1)
                results[f'N@{k}'] += 1.0 / np.log2(rank + 1)

    # Chia trung bình batch
    for k in results:
        results[k] /= batch_size
        
    return results

# ==========================================
# 4. MAIN EVALUATION LOOP
# ==========================================
def run_full_eval():
    model, config, df, device = load_resources()
    
    # Định nghĩa Mask Token ID (thường là item cuối cùng + 1)
    MASK_TOKEN = config.num_items + 1
    
    print("-" * 50)
    print(f"BẮT ĐẦU ĐÁNH GIÁ FULL RANKING")
    print(f"   - Total Items: {config.num_items}")
    print(f"   - Mask Token ID: {MASK_TOKEN}")
    print("-" * 50)
    
    # --- PREPROCESSING ---
    print("[2/4] Preprocessing Data...")
    user_groups = df.sort_values('timestamp').groupby('uid')['mid'].apply(list)
    
    test_seqs = []
    test_targets = []
    
    # Chiến thuật Leave-One-Out:
    # Input: [item_1, ..., item_k, MASK]
    # Target: item_next
    for _, movies in tqdm(user_groups.items(), desc="Building Sequences"):
        if len(movies) < 2: continue
        
        # Lấy lịch sử trừ item cuối
        history = movies[:-1]
        # Item cuối là target
        target = movies[-1]
        
        # Cắt lịch sử cho vừa max_len (chừa 1 chỗ cho Mask)
        max_history = config.max_len - 1
        if len(history) > max_history:
            history = history[-max_history:]
            
        # Thêm Mask vào cuối chuỗi input
        seq = history + [MASK_TOKEN]
        
        # Left Padding (thêm số 0 vào trước cho đủ độ dài)
        padding_len = config.max_len - len(seq)
        if padding_len > 0:
            seq = [0] * padding_len + seq
            
        test_seqs.append(seq)
        test_targets.append(target)
    
    # Convert to Tensor
    test_seqs = torch.tensor(test_seqs, dtype=torch.long)
    test_targets = torch.tensor(test_targets, dtype=torch.long)
    
    # DataLoader
    # Batch size vừa phải để tránh OOM khi tính Full Ranking matrix
    dataset = TensorDataset(test_seqs, test_targets)
    loader = DataLoader(dataset, batch_size=32, shuffle=False) 
    
    # Biến lưu tổng kết quả
    final_metrics = {'R@5': [], 'R@10': [], 'R@20': [], 
                     'N@5': [], 'N@10': [], 'N@20': []}
    
    print(f"[3/4] Running Inference & Ranking...")
    
    with torch.no_grad():
        for seq_batch, target_batch in tqdm(loader, desc="Evaluating"):
            seq_batch = seq_batch.to(device)
            target_batch = target_batch.to(device)
            
            # Forward pass
            logits = model(seq_batch) # Shape: [Batch, SeqLen, Vocab]
            
            # Lấy logits tại vị trí MASK
            # mask_indices: Ma trận boolean [Batch, SeqLen]
            mask_indices = (seq_batch == MASK_TOKEN)
            
            # Chọn ra vector dự đoán tương ứng với vị trí mask
            # Shape sau khi chọn: [Batch, Vocab]
            # Lưu ý: sum(mask_indices) phải bằng Batch size (mỗi seq có 1 mask)
            target_logits = logits[mask_indices] 
            
            # Check shape an toàn
            if target_logits.size(0) != seq_batch.size(0):
                # Trường hợp lỗi hiếm gặp nếu padding đè mất mask (logic trên đã handle rồi)
                continue
            
            # Tính Metrics
            batch_res = calculate_full_ranking_metrics(
                target_logits, 
                target_batch, 
                seq_batch, 
                mask_token=MASK_TOKEN,
                k_list=[5, 10, 20]
            )
            
            for k, v in batch_res.items():
                final_metrics[k].append(v)

    # --- REPORT ---
    print("\n" + "="*40)
    print(f"KẾT QUẢ ĐÁNH GIÁ (Toàn tập dữ liệu)")
    print("="*40)
    
    def fmt(metric_name):
        return f"{np.mean(final_metrics[metric_name]):.4f}"

    print(f"Recall@5 : {fmt('R@5')}")
    print(f"Recall@10: {fmt('R@10')}")
    print(f"Recall@20: {fmt('R@20')}")
    print("-" * 40)
    print(f"NDCG@5   : {fmt('N@5')}")
    print(f"NDCG@10  : {fmt('N@10')}")
    print(f"NDCG@20  : {fmt('N@20')}")
    print("="*40)
    print(" Hoàn tất.")

if __name__ == "__main__":
    run_full_eval()