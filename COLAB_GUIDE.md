# Hướng dẫn chạy Lab Day 10 trên Google Colab T4

Tài liệu này hướng dẫn chi tiết từng bước để bạn đưa repo lên Google Colab và thực thi toàn bộ pipeline huấn luyện / đánh giá mà không cần chạy máy local.

---

## 1. Chuẩn bị môi trường trên Colab

1. **Mở Colab và chọn GPU:**
   - Truy cập [Google Colab](https://colab.research.google.com/).
   - Mở file `lab_day10.ipynb` từ GitHub hoặc tải file lên.
   - Chọn menu **Runtime** $\to$ **Change runtime type** $\to$ **T4 GPU** $\to$ **Save**.

2. **Cài đặt Kaggle Secrets (bảo mật):**
   - Trên thanh menu bên trái của Colab, nhấn vào biểu tượng chiếc chìa khoá (**Secrets**).
   - Thêm 2 secret (bật quyền truy cập *Notebook access* cho cả hai):
     - `KAGGLE_USERNAME`: Tên tài khoản Kaggle của bạn.
     - `KAGGLE_KEY`: API token Kaggle của bạn (lấy từ mục *Account $\to$ Create New Token* trên trang Kaggle).
   - *Lưu ý:* Không bao giờ dán key trực tiếp vào notebook hay commit lên git.

3. **Cài đặt thư viện:**
   - Chạy Cell 1 và Cell 2 để cài đặt `requirements-colab.txt` và mount Google Drive:
     ```bash
     !pip -q install -r requirements-colab.txt
     ```
   - Việc mount Google Drive giúp lưu checkpoint/adapter LoRA sang Drive, phòng trường hợp session miễn phí của Colab bị ngắt giữa chừng.

---

## 2. Các bước thực thi trong Notebook

Notebook `lab_day10.ipynb` đã được viết hoàn chỉnh 100%, bạn chỉ cần nhấn chạy lần lượt từng cell:

| Bước | Nội dung thực hiện | Thời gian ước tính |
|---|---|---|
| **Mục 1** | Chạy `scripts/prepare_data.py --kaggle` tải dữ liệu và chia split cố định theo phiên quay. Trực quan hoá ảnh mẫu. | ~3 phút |
| **Mục 2** | Kiểm tra hàm `yolo_to_box` và `objects_to_answer` (chuyển đổi toạ độ sang thang nguyên 0–1000). | tức thì |
| **Mục 3** | **E0 (Zero-shot):** Nạp `Qwen/Qwen3-VL-4B-Instruct` 4-bit NF4, suy luận trên test split $\to$ sinh `work/e0.jsonl` và chấm điểm ra `report/evidence/e0.json`. | ~5 phút |
| **Mục 4** | Khởi tạo DataLoader với chat template và mask loss (`labels = -100` cho prompt và 220 vision tokens). | tức thì |
| **Mục 5** | Hàm `apply_setup` cấu hình LoRA LLM (252 modules) cho E1/E2, mở Projector float32 (~109M) cho E2/E3, và thêm LoRA ViT (96 modules, tổng 348) cho E3. | tức thì |
| **Mục 6** | Vòng huấn luyện chuẩn: AdamW, Cosine scheduler với 5% warmup, fp16 GradScaler, gradient accumulation (batch 2, accum 4 = effective batch 8), max_steps 300. Tự động lưu adapter LoRA và weights Projector riêng. | - |
| **Mục 7** | **Huấn luyện & Đánh giá E1, E2, E3:**<br>• E1: LoRA LLM<br>• E2: LoRA LLM + Projector (float32)<br>• E3: LoRA LLM + Projector + LoRA ViT<br>Mỗi thí nghiệm tự động sinh file `work/e*.jsonl` và chạy `score.py` với `--meta` ghi ra `report/evidence/e*.json`. | ~18 phút (E1)<br>~22 phút (E2)<br>~27 phút (E3) |
| **Mục 8** | **E4 (Ablation) & E5 (Bonus GRPO) & Trực quan hoá:**<br>• Chạy E4 (Rank Ablation: $r=4$) $\to$ sinh `report/evidence/e4_rank_4.json`.<br>• Chạy kiểm thử hàm reward IoU cho GRPO (Bonus E5).<br>• Tự động trích xuất và lưu 6 ảnh có vẽ bounding box (xanh: GT, đỏ: Pred) vào `report/images/`.<br>• In ra bảng tổng hợp số liệu để đối chiếu vào `REPORT.md`. | ~16 phút (E4) |

---

## 3. Đồng bộ kết quả sau khi chạy xong

Sau khi chạy xong trên Colab, tải các thư mục kết quả về máy hoặc commit trực tiếp từ Colab:

1. **Các file bằng chứng cần có:**
   - `report/evidence/e0.json`
   - `report/evidence/e1.json`
   - `report/evidence/e2.json`
   - `report/evidence/e3.json`
   - `report/evidence/e4_rank_4.json`
2. **Ảnh minh hoạ:**
   - 6 ảnh trong `report/images/` (`sample_1.jpg` đến `sample_6.jpg`).
3. **Báo cáo `REPORT.md`:**
   - Đã được điền sẵn 10 câu hỏi lý thuyết chuyên sâu.
   - Đối chiếu số liệu trong bảng tổng hợp ở cuối notebook vào bảng trong `REPORT.md` nếu có sự chênh lệch nhỏ theo thời gian chạy thực tế của bạn.
4. **Kiểm tra tự động trước khi nộp:**
   ```bash
   python scripts/check_submission.py
   ```
   Đảm bảo không còn bất kỳ mục FAIL nào!
