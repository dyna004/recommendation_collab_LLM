# Giới thiệu

Bài làm được triển khai các mô hình từ hai bài báo sau:

> **BERT4Rec: Sequential Recommendation with BERT (Sun et al.)**  

> **Variational Autoencoders for Collaborative Filtering (Liang et al.)**  

và cho phép bạn đào tạo chúng trên MovieLens-1m và MovieLens-20m.

# Cách sử dụng
1. Câu lệnhBERT4Rec sử dụng ML-1m 
   ```bash
   printf '1\ny\n' | python main.py --template train_bert
   ```
2. Chạy giao diện
```bash
    streamlit run đường dẫn:\..
```
3. chạy đánh giá 
```bash
    python đường dẫn:\...
```

# Đánh giá 
<img src=ảnh\semantic.png>
<img src=ảnh\evaluate.png>

# Giao diện
<img src=ảnh\Streamlit.png>