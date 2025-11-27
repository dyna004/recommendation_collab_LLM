import streamlit as st
import torch
import json
import pandas as pd
import sys
import os
import google.generativeai as genai

# ==========================================
# 1. CẤU HÌNH & GIAO DIỆN
# ==========================================
st.set_page_config(page_title="Recommendation", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #121212; color: #FFF; }

    /* SECTION HEADER */
    .section-header {
        font-size: 1.8em;
        font-weight: bold;
        margin-top: 30px;
        margin-bottom: 20px;
        border-bottom: 2px solid #333;
        padding-bottom: 10px;
    }

    /* SERIES SPOTLIGHT CARD */
    .series-card {
        background: linear-gradient(135deg, #4A00E0, #8E2DE2);
        padding: 30px;
        border-radius: 12px;
        margin-bottom: 25px; /* Khoảng cách giữa các series nếu có nhiều */
        box-shadow: 0 4px 20px rgba(142, 45, 226, 0.5);
        border: 1px solid #A66CFF;
        position: relative;
        overflow: hidden;
    }
    
    .series-label { 
        font-size: 0.9em; 
        font-weight: bold; 
        color: #FFD700; 
        text-transform: uppercase; 
        letter-spacing: 2px; 
        margin-bottom: 10px; 
        display: flex; 
        align-items: center; 
        gap: 10px;
    }
    
    .series-title { 
        font-size: 2.5em; 
        font-weight: bold; 
        color: #FFF; 
        margin-bottom: 15px; 
        font-family: 'Arial Black', sans-serif; 
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5); 
    }
    
    .series-desc { 
        font-size: 1.1em; 
        line-height: 1.6; 
        color: #F0F0F0; 
        margin-bottom: 25px; 
        background: rgba(0,0,0,0.1);
        padding: 15px;
        border-radius: 8px;
    }
    
    /* Movie Tags inside Spotlight */
    .tag-area-label {
        font-size: 0.85em; 
        font-weight: bold; 
        color: #E0E0E0; 
        margin-bottom: 10px; 
        text-transform: uppercase;
        opacity: 0.8;
    }
    
    .tag-container { 
        display: flex; 
        flex-wrap: wrap; 
        gap: 8px; 
    }
    
    .movie-tag {
        background: rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.3);
        color: #FFF;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.9em;
        font-weight: 500;
        transition: all 0.2s;
    }
    .movie-tag:hover {
        background: #FFF;
        color: #4A00E0;
        cursor: default;
    }
    
    /* Non-series Intro */
    .non-series-box {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        border-left: 4px solid #00F0FF;
    }
    .non-series-title { font-weight: bold; color: #00F0FF; font-size: 1.2em; display: flex; align-items: center; gap: 10px;}

    /* CARD: KINH ĐIỂN */
    .classic-card {
        background-color: #2C2C2C;
        border-left: 5px solid #F1C40F;
        padding: 20px;
        margin-bottom: 15px;
        border-radius: 8px;
    }
    .classic-title { color: #F1C40F; font-size: 1.3em; font-weight: bold; font-family: 'Courier New', monospace; }
    .classic-rank { font-size: 1.5em; color: #FFF; font-weight: bold; margin-right: 10px; }
    .classic-reason { font-size: 0.95em; color: #DDD; margin-top: 8px; font-style: italic; line-height: 1.5; }
    .classic-meta { font-size: 0.8em; color: #888; margin-top: 5px; text-transform: uppercase; letter-spacing: 1px; }

    /* CARD: HIỆN ĐẠI */
    .modern-list-item {
        background: rgba(0, 240, 255, 0.05);
        border: 1px solid #00F0FF;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        transition: all 0.3s ease;
        height: 100%;
    }
    .modern-list-item:hover {
        background: rgba(0, 240, 255, 0.15);
        box-shadow: 0 0 15px rgba(0, 240, 255, 0.3);
        transform: translateY(-3px);
    }
    .modern-item-title { color: #00F0FF; font-weight: bold; font-size: 1.1em; margin-bottom: 5px; }
    .modern-item-desc { color: #CCC; font-size: 0.9em; font-style: italic; line-height: 1.4; }
    
    .divider-custom { margin: 40px 0; border-top: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

# --- API KEY & PATH ---
BASE_PATH = "D:/BERT4Rec-VAE-Pytorch"
if BASE_PATH not in sys.path:
    sys.path.append(BASE_PATH)

try:
    from models.bert import BERTModel
except ImportError:
    st.error(f" Lỗi: Không tìm thấy source code BERT4Rec tại: {BASE_PATH}")
    st.stop()

# API Key Gemini
api_key = "AIzaSyALZdaW-B2jDVrG6dHA6W1OwMw__OzT9Ho" 
genai.configure(api_key=api_key)

# ==========================================
# 2. LOAD DATA & MODEL
# ==========================================
@st.cache_resource
def load_model_and_data():
    base = BASE_PATH
    data_path = f"{base}/data/ml-1m.rating"
    movies_path = f"{base}/data/movies.dat"
    config_path = f"{base}/Test/config.json"
    ckpt_path = f"{base}/Test/Model/best_acc_model.pth"

    try:
        # Load Config
        df = pd.read_csv(data_path, sep="::", header=None, engine="python")
        item_ids = df[1].unique()
        num_items = len(item_ids)

        with open(config_path) as f: cfg = json.load(f)

        class ConfigNamespace:
            def __init__(self, cfg, num_items):
                for k, v in cfg.items(): setattr(self, k, v)
                self.num_items = num_items

        config = ConfigNamespace(cfg, num_items)
        
        # Load Model
        model = BERTModel(config)
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.eval()

        # Load Movies
        movies_df = pd.read_csv(
            movies_path, sep="::", header=None, engine="python",
            names=["movie_id", "title", "genres"], encoding="latin-1"
        )

        id2title = dict(zip(movies_df.movie_id, movies_df.title))
        id2genres = dict(zip(movies_df.movie_id, movies_df.genres))
        valid_movie_ids = list(id2title.keys())
        
        return model, config, id2title, id2genres, valid_movie_ids

    except Exception as e:
        st.error(f"Lỗi tải resources: {e}")
        st.stop()

try:
    model, config, id2title, id2genres, valid_movie_ids = load_model_and_data()
    title2id = {v: k for k, v in id2title.items()}
    all_titles = list(title2id.keys())
except Exception as e:
    st.error(f"Lỗi khởi tạo: {e}")
    st.stop()

# ==========================================
# 3. CORE LOGIC (BERT)
# ==========================================

def recommend_movies(model, config, id2title, id2genres, watched_movies):
    """Gợi ý phim từ BERT4Rec với Genre Boosting."""
    model.eval()
    max_len = 100
    watched_movies = [mid for mid in watched_movies if mid > 0 and mid < config.num_items]
    
    if not watched_movies: return "","", "", []
        
    pad_token = getattr(config, "pad_token", 0)
    seq = watched_movies + [pad_token] * (max_len - len(watched_movies))
    seq = torch.tensor([seq])

    with torch.no_grad():
        logits = model(seq)

    last_logits = logits[:, len(watched_movies) - 1, :]
    probs = torch.softmax(last_logits, dim=-1)

    for mid in watched_movies:
        if mid < probs.size(-1): probs[0, mid] = -1e9

    # Genre Boosting
    genre_count = {}
    for mid in watched_movies:
        for g in id2genres.get(mid, "").split("|"):
            genre_count[g] = genre_count.get(g, 0) + 1

    boost = torch.zeros_like(probs)
    for item_id in range(probs.size(-1)):
        movie_id = item_id + 1 
        g = id2genres.get(movie_id, "").split("|")
        bonus = sum(genre_count.get(gg, 0) for gg in g)
        boost[0, item_id] = bonus * 0.01

    probs = probs + boost

    rec_ids, scores = [], []
    probs_clone = probs.clone()

    while len(rec_ids) < 10:
        top_item = torch.argmax(probs_clone).item()
        movie_id = top_item + 1 
        title = id2title.get(movie_id)
        if title and movie_id not in watched_movies:
            rec_ids.append(movie_id)
            scores.append(probs_clone[0, top_item].item())
        probs_clone[0, top_item] = -1e9 
        if torch.max(probs_clone) < -1e8: break

    watched_text = "\n".join([f"- {id2title.get(mid)}" for mid in watched_movies])
    genres_info = ", ".join(set(",".join([id2genres.get(mid, "Unknown") for mid in watched_movies]).split("|")))
    rec_text = "\n".join([f"{i+1}. {id2title.get(rec_ids[i])} (score={scores[i]:.3f})" for i in range(len(rec_ids))])

    return watched_text, genres_info, rec_text, rec_ids

# ==========================================
# 4. AI LOGIC (GEMINI - MULTI-SERIES SUPPORT)
# ==========================================

def gemini_rerank_and_suggest(history_titles, candidates_top10):
    prompt = f"""
    CONTEXT:
    - User History: {json.dumps(history_titles)}
    - Candidates (Top 10 from BERT): {json.dumps(candidates_top10)}
    
    TASK 1 (MULTI-SERIES DETECTION): 
    - Analyze the "User History". Identify ALL distinct movie franchises/series present (e.g., if history has "Iron Man" and "Harry Potter", identify BOTH "MCU" and "Wizarding World").
    - Return a LIST of detected series in "series_list".
    - If a movie in history is NOT part of any big series, list it in "other_inputs_info".
    
    TASK 2 (RECOMMENDATIONS - STRICT EXCLUSION):
    - Select TOP 5 CLASSIC movies (pre-2000).
    - Select TOP 10 MODERN movies (post-2015).
    - CRITICAL RULE: Recommendations MUST NOT contain any movie that belongs to the detected series or the user history.

    OUTPUT JSON FORMAT (Strictly):
    {{
        "series_list": [
            {{
                "series_name": "Name of Series 1",
                "series_overview": "Intro about Series 1 (Vietnamese)",
                "all_movies_in_series": ["Movie 1", "Movie 2", ...]
            }},
            {{
                "series_name": "Name of Series 2",
                "series_overview": "Intro about Series 2 (Vietnamese)",
                "all_movies_in_series": ["Movie A", "Movie B", ...]
            }}
        ],
        "other_inputs_info": [
            {{ "title": "Non-series Movie Title", "brief": "Short intro (Vietnamese)" }}
        ],
        "reranked_classics": [
            {{ "title": "...", "rank": 1, "reason": "Reason (Vietnamese)" }}, ... (5 items)
        ],
        "modern_list": [
            {{ "title": "...", "overview": "Description (Vietnamese)" }}, ... (10 items)
        ]
    }}
    """
    return call_gemini(prompt)

def gemini_direct_recommendation(user_input_text):
    prompt = f"""
    CONTEXT: User likes: "{user_input_text}".
    
    TASK 1 (MULTI-SERIES DETECTION): 
    - Identify ALL distinct movie franchises/series in the input (e.g., input "Iron Man, Star Wars" -> detect both MCU and Star Wars).
    - Return a LIST of detected series in "series_list".
    - If a movie in input is NOT part of a big series, list it in "other_inputs_info".

    TASK 2 (RECOMMENDATIONS - STRICT EXCLUSION):
    - Select TOP 5 CLASSIC movies (pre-2000).
    - Select TOP 10 MODERN movies (post-2015).
    - CRITICAL RULE: Recommendations MUST NOT contain any movie that belongs to the detected series.

    OUTPUT JSON FORMAT (Strictly):
    {{
        "series_list": [
            {{
                "series_name": "Name of Series 1",
                "series_overview": "Intro about Series 1 (Vietnamese)",
                "all_movies_in_series": ["Movie 1", "Movie 2", ...]
            }},
             {{
                "series_name": "Name of Series 2",
                "series_overview": "Intro about Series 2 (Vietnamese)",
                "all_movies_in_series": ["Movie A", "Movie B", ...]
            }}
        ],
        "other_inputs_info": [
            {{ "title": "Non-series Movie Title", "brief": "Short intro (Vietnamese)" }}
        ],
        "reranked_classics": [
            {{ "title": "...", "rank": 1, "reason": "Reason (Vietnamese)" }}, ... (5 items)
        ],
        "modern_list": [
            {{ "title": "...", "overview": "Description (Vietnamese)" }}, ... (10 items)
        ]
    }}
    """
    return call_gemini(prompt)

def call_gemini(prompt):
    try:
        gemini = genai.GenerativeModel("gemini-2.5-flash")
        res = gemini.generate_content(prompt)
        clean_text = res.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_text)
    except Exception as e:
        return None

# ==========================================
# 5. UI LOGIC & FLOW
# ==========================================

st.title("Hybrid: BERT + Gemini Movie Recommender")
st.markdown("Hệ thống kết hợp sức mạnh phân tích hành vi cổ điển (**BERT**) và sự thấu hiểu nội dung hiện đại (**LLM**).")

# --- INPUT AREA ---
with st.container(border=True):
    col1, col2 = st.columns([1, 3])
    with col1:
        input_type = st.radio("Nguồn dữ liệu:", ["Chọn phim cũ (Database)", "Nhập phim mới bất kỳ"])

    final_input_ids = []
    history_titles = []
    direct_input_text = ""

    with col2:
        if input_type == "Chọn phim cũ (Database)":
            if not all_titles:
                st.error("Dữ liệu phim chưa được tải thành công.")
            else:
                default_val = ["Star Wars: Episode IV - A New Hope (1977)"]
                valid_defaults = [x for x in default_val if x in all_titles]
                
                selected = st.multiselect("Phim bạn đã xem:", all_titles, default=valid_defaults)
                for t in selected: 
                    final_input_ids.append(title2id[t])
                    history_titles.append(t)
        else:
            direct_input_text = st.text_input("Nhập tên phim (VD: Star Wars, Harry Potter, Titanic...):")

# --- EXECUTION ---
if st.button(" Phân Tích & Gợi Ý", type="primary"):
    
    ai_results = None
    
    # 1. PROCESS LOGIC
    if input_type == "Chọn phim cũ (Database)":
        if not final_input_ids:
            st.warning("Vui lòng chọn ít nhất 1 phim!")
        else:
            with st.spinner("Đang chạy mô hình AI..."):
                watched_text, genres_info, rec_text, rec_ids = recommend_movies(
                    model, config, id2title, id2genres, final_input_ids
                )
                bert_top10_titles = [id2title[mid] for mid in rec_ids if mid in id2title]
                
                with st.expander(" Chi tiết kỹ thuật (BERT Output)", expanded=False):
                    st.code(rec_text, language='text')
                
                ai_results = gemini_rerank_and_suggest(history_titles, bert_top10_titles)
    else:
        if not direct_input_text:
            st.warning("Vui lòng nhập tên phim!")
        else:
            with st.spinner("Gemini đang phân tích các vũ trụ điện ảnh..."):
                ai_results = gemini_direct_recommendation(direct_input_text)

    # 2. RENDER UI
    if ai_results:
        # --- PHẦN 0: SERIES SPOTLIGHT (Loop through all detected series) ---
        series_list = ai_results.get("series_list", [])
        
        if series_list:
            for series in series_list:
                st.markdown(f"""
                <div class="series-card">
                    <div class="series-label">✨ SPOTLIGHT: VŨ TRỤ PHIM / SERIES</div>
                    <div class="series-title">{series.get('series_name', 'Movie Series')}</div>
                    <div class="series-desc">"{series.get('series_overview', '')}"</div>               
                    <div class="tag-area-label">DANH SÁCH PHIM THUỘC SERIES NÀY:</div>
                    <div class="tag-container">
                """, unsafe_allow_html=True)
                
                # Render Tags INSIDE the purple card
                series_movies = series.get('all_movies_in_series', [])
                tag_html = ""
                for mov in series_movies:
                    tag_html += f'<span class="movie-tag">{mov}</span>'
                st.markdown(f'{tag_html}</div></div>', unsafe_allow_html=True)
        
        # --- PHẦN 0.5: NON-SERIES INFO ---
        other_inputs = ai_results.get('other_inputs_info', [])
        if other_inputs:
             for item in other_inputs:
                st.markdown(f"""
                <div class="non-series-box">
                    <div class="non-series-title"> Ngoài ra: {item.get('title')}</div>
                    <div style="font-size:0.9em; color:#DDD; margin-top:5px;">{item.get('brief')}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # --- PHẦN 1: TOP 5 KINH ĐIỂN ---
        st.markdown('<div class="section-header">Top 5 Kinh Điển </div>', unsafe_allow_html=True)
        
        reranked_classics = ai_results.get("reranked_classics", [])
        if not reranked_classics:
             st.info("Không có dữ liệu đề xuất kinh điển.")
        
        for item in reranked_classics:
            mid = title2id.get(item['title']) if input_type == "Chọn phim cũ (Database)" else None
            genre_str = id2genres.get(mid, "Classic Cinema") if mid else "Classic Cinema"
            
            st.markdown(f"""
            <div class="classic-card">
                <div style="display:flex; align-items:center;">
                    <span class="classic-rank">#{item['rank']}</span>
                    <div>
                        <div class="classic-title">{item['title']}</div>
                        <div class="classic-meta">{genre_str}</div>
                    </div>
                </div>
                <div class="classic-reason">"{item['reason']}"</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="divider-custom"></div>', unsafe_allow_html=True)

        # --- PHẦN 2: TOP 10 HIỆN ĐẠI ---
        st.markdown('<div class="section-header">Top 10 Hiện Đại </div>', unsafe_allow_html=True)
        
        modern_list = ai_results.get("modern_list", [])
        col_m1, col_m2 = st.columns(2)
        
        for i, item in enumerate(modern_list):
            with (col_m1 if i % 2 == 0 else col_m2):
                st.markdown(f"""
                <div class="modern-list-item">
                    <div class="modern-item-title"> {item.get('title', 'Unknown')}</div>
                    <div class="modern-item-desc">"{item.get('overview', '')}"</div>
                </div>
                """, unsafe_allow_html=True)

    elif ai_results is None and (final_input_ids or direct_input_text):
         st.error("Không nhận được phản hồi từ AI. Vui lòng thử lại.")