Họ tên: Võ Đức Tài
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | - | - | 0.042 | [0.025, 0.061] | 0.85 | 0.051 | 0.028 | 0.065 |
| E1 | LoRA LLM | 20,447,232 | 8.15 | 18.2 | 0.468 | [0.428, 0.509] | 0.38 | 0.485 | 0.412 | 0.562 |
| E2 | + projector | 129,628,800 | 10.42 | 22.5 | 0.504 | [0.462, 0.545] | 0.32 | 0.521 | 0.448 | 0.605 |
| E3 | + LoRA ViT | 137,427,584 | 12.80 | 27.1 | 0.512 | [0.470, 0.553] | 0.31 | 0.530 | 0.455 | 0.612 |
| E4 | rank r=4 (alpha=8) | 5,111,808 | 7.90 | 16.5 | 0.452 | [0.411, 0.493] | 0.41 | 0.468 | 0.395 | 0.548 |

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
- **Số lượng tham số từng khối trong Qwen3-VL-4B (tổng cộng ≈ 4.44B tham số):**
  - **ViT (Vision Transformer):** ≈ 415M tham số (24 visual blocks xử lý các patch ảnh kích thước 14×14).
  - **Projector:** ≈ 109M tham số (gồm `visual.merger` ≈ 27M tham số và `visual.deepstack_merger_list` ≈ 82M tham số, kết hợp đặc trưng đa tầng ở các lớp 5, 11, 17 của ViT đưa vào LLM).
  - **LLM (Language Model):** ≈ 4.0B tham số (36 transformer layers + embeddings + lm_head).
- **Khối được cập nhật trong từng thí nghiệm:**
  - **E1 (LoRA LLM):** Chỉ cập nhật các adapter LoRA gắn vào 252 module của LLM (`self_attn` q, k, v, o và `mlp` gate, up, down trên 36 layers với rank r=16, alpha=32). ViT và Projector đóng băng hoàn toàn.
  - **E2 (+ Projector):** Cập nhật LoRA trên LLM (như E1) + mở cập nhật đầy đủ (full fine-tune) toàn bộ Projector (`visual.merger` và `visual.deepstack_merger_list`, ≈ 109M tham số) ở định dạng float32. ViT vẫn đóng băng.
  - **E3 (+ LoRA ViT):** Cập nhật LoRA trên LLM + mở Projector (như E2) + gắn thêm LoRA adapter trên các khối Attention và MLP của ViT (`attn.qkv, attn.proj, mlp.linear_fc1, mlp.linear_fc2` trên 24 blocks, thêm 96 modules LoRA, tổng cộng 348 modules LoRA).

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
- **`parse_valid_rate` của E0:** Đạt khoảng 0.65 (65%), do mô hình base chưa được tinh chỉnh theo format JSON grounding chuyên biệt.
- **Hai lỗi phổ biến nhất của model gốc:**
  1. *Lỗi định dạng cú pháp JSON và sinh văn bản tự do:* Model thường sinh kèm lời giải thích dài dòng, thẻ code markdown không đóng (` ```json `), hoặc sinh lặp lại văn bản mô tả xung quanh thay vì trả về duy nhất một JSON list hợp lệ.
  2. *Lỗi sai thang toạ độ hoặc ảo giác (hallucination):* Model trả toạ độ theo pixel ảnh thực tế (ví dụ [120, 240, 250, 310] trên ảnh 640×360) hoặc toạ độ chuẩn hoá [0.0, 1.0] thay vì thang toạ độ nguyên 0–1000 theo quy ước, hoặc gán nhãn ngoài 3 lớp cho phép (ví dụ nhãn `asphalt`, `road`, `car`).
- **Ví dụ output thô từ E0:**
  `Based on the image, I detected the following road damage: ```json\n[{"label": "crack", "bbox_2d": [145.2, 210.5, 320.1, 280.4]}]\n``` The cracks are visible in the middle of the road.` (Bị lỗi do bọc markdown và toạ độ số thực).

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
- **Hậu quả lên F1:** Nếu mô hình dự đoán theo thang toạ độ 0–1000 nhưng chấm theo toạ độ pixel (hoặc ngược lại):
  - Giá trị F1 ngay lập tức sập về **0.000** (Precision = 0.0, Recall = 0.0).
- **Nguyên nhân:**
  - Ảnh có kích thước 640×360. Một box ở giữa ảnh có toạ độ thang 1000 là [400, 400, 600, 600]. Nếu chấm theo pixel, toạ độ y=400 đã vượt ra ngoài chiều cao khung hình (H=360), diện tích giao nhau giữa box dự đoán và ground-truth pixel là 0.
  - Tương tự, nếu mô hình xuất box theo pixel [100, 50, 200, 100], khi coi đó là thang 1000 thì box đó chỉ chiếm 10% × 5% góc trên cùng bên trái khung hình, dẫn tới IoU với box thật luôn < 0.1 (dưới ngưỡng IoU >= 0.5).
- **Cách kiểm chứng:**
  - Chạy `score.py` với cờ `--coord pixel` trên output của `work/e1.jsonl` (vốn ở thang 1000), kết quả F1 đo được lập tức giảm về 0.000.
  - Test case `test_pixel_vs_norm1000_scale_trap` trong file `tests/test_score.py` cũng chứng minh rõ hiện tượng này: khi tráo đổi thang toạ độ, số lượng TP giảm về 0 và toàn bộ trở thành FP/FN.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
- **Vì sao chỉ tính loss trên phần trả lời:**
  - Nhiệm vụ sinh ngôn ngữ có điều kiện (conditional generation) yêu cầu mô hình học xác suất P(Answer | Prompt, Image).
  - Bằng cách gán `labels = -100` cho toàn bộ các token thuộc về prompt và hình ảnh, hàm mất mát `CrossEntropyLoss` (với `ignore_index = -100`) sẽ hoàn toàn bỏ qua các token này và chỉ tính gradient trên các token do assistant sinh ra (cấu trúc JSON, nhãn lớp và toạ độ box).
- **Điều gì xảy ra nếu tính loss trên cả prompt và token ảnh:**
  1. *Lãng phí dung lượng mô hình:* Prompt là chuỗi câu hỏi cố định và ảnh chiếm tới 220 vision tokens (áp đảo độ dài câu trả lời). Nếu tính loss trên cả prompt và ảnh, gradient sẽ bị chi phối bởi việc học tái tạo lại câu hỏi rập khuôn và các đặc trưng hình ảnh, làm loãng tín hiệu gradient học toạ độ box.
  2. *Gây overfitting và suy giảm khả năng tổng quát hoá:* Mô hình sẽ bị overfit vào câu từ cụ thể của prompt thay vì học cách căn chỉnh thị giác - không gian (visual grounding), khiến mô hình hoạt động kém khi gặp ảnh mới hoặc prompt thay đổi nhẹ.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
- **Vì sao không thể huấn luyện nếu ViT bị lượng tử hoá 4-bit:**
  - Khi một module bị lượng tử hoá 4-bit qua `bitsandbytes`, các trọng số được lưu dưới dạng `Params4bit` (bảng mã khối NF4). Không thể trực tiếp cập nhật gradient và thay đổi trọng số gốc (full fine-tuning) trên kiểu dữ liệu lượng tử hoá này nếu không dùng adapter LoRA.
  - Nếu toàn bộ `visual` bị lượng tử hoá 4-bit, các module của projector (`visual.merger` và `visual.deepstack_merger_list`) cũng trở thành trọng số lượng tử 4-bit và không thể gán `requires_grad = True` để tối ưu trực tiếp bằng AdamW.
- **Cách xử lý:**
  - Khi khởi tạo `BitsAndBytesConfig`, truyền cờ `llm_int8_skip_modules=["visual", "lm_head"]`. Tùy chọn này chỉ thị thư viện không lượng tử hoá phần thị giác, giữ nguyên ViT và Projector ở định dạng số thực (fp16/fp32) để cho phép mở cập nhật ở E2/E3.
- **Vì sao projector phải ở float32:**
  - Projector làm nhiệm vụ nén và ánh xạ không gian vector đặc trưng từ ViT sang không gian ẩn của LLM. Đây là cầu nối trung gian chịu sự biến động gradient rất lớn từ cả hai phía.
  - Định dạng `float16` có dải động hẹp (dynamic range nhỏ), rất dễ bị hiện tượng tràn số (overflow -> Inf) hoặc triệt tiêu dưới (underflow -> 0). Đưa projector về `float32` giúp duy trì độ chính xác số học cao, ổn định gradient và ngăn ngừa bùng nổ loss `NaN`.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
- **Vì sao chia ngẫu nhiên sẽ cho F1 cao giả:**
  - Tập dữ liệu gồm các frame cắt liên tiếp từ các video hành trình dashcam. Hai frame cách nhau 1–2 giây trên cùng một đoạn đường có background, điều kiện ánh sáng, thời tiết và cùng các ổ gà/vết nứt giống hệt nhau.
  - Nếu chia ngẫu nhiên (random split), các frame liền kề của cùng một đoạn đường sẽ bị phân tán vào cả train và test (gây rò rỉ dữ liệu - data leakage nghiêm trọng). Mô hình sẽ ghi nhớ background của đoạn đường đó và đạt F1 cao giả tạo. Nhưng khi áp dụng thực tế trên một cung đường lạ, F1 sẽ sụt giảm nghiêm trọng.
  - Chia theo phiên quay (`session_key` dựa trên ngày, giờ và tiền tố tệp) đảm bảo toàn bộ một hành trình chỉ nằm trọn vẹn trong một split duy nhất.
- **Số ảnh và số box từng lớp ở mỗi split (từ `data/splits/meta.json`):**
  - **Train:** 1.422 ảnh | 1.074 pothole, 1.750 crack, 691 manhole (tổng 3.515 box).
  - **Val:** 276 ảnh | 66 pothole, 470 crack, 103 manhole (tổng 639 box).
  - **Test:** 311 ảnh | 121 pothole, 299 crack, 163 manhole (tổng 583 box).

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
- **So sánh định lượng và tài nguyên:**
  - **E1 (LoRA LLM):** Chi phí tài nguyên thấp nhất (~8.15 GB VRAM, ~18.2 phút), F1 tăng vọt so với E0 (từ zero-shot 0.042 lên 0.468).
  - **E2 (+ Projector):** Bổ sung ~109M tham số trainable, VRAM đỉnh tăng lên (10.42 GB), thời gian huấn luyện tăng nhẹ (22.5 phút). F1 tiếp tục cải thiện lên 0.504 nhờ projector học cách căn chỉnh tốt hơn giữa đặc trưng thị giác và toạ độ grounding.
  - **E3 (+ LoRA ViT):** Bổ sung 96 module LoRA trên ViT, VRAM đỉnh cao nhất (12.80 GB, chạm ngưỡng an toàn của GPU T4), thời gian train lâu nhất (27.1 phút). Tuy nhiên F1 chỉ tăng nhẹ lên 0.512 và dao động trong khoảng CI95 so với E2 [0.470, 0.553] vs [0.462, 0.545], do backbone ViT vốn đã được huấn luyện rất mạnh, việc mở ViT trên tập dữ liệu nhỏ (1.422 ảnh) dễ bị bão hoà hoặc overfit nhẹ.
- **Đánh giá "đáng đồng tiền":**
  - E2 là cấu hình tối ưu nhất (sweet spot) khi cân bằng giữa hiệu quả cải thiện F1 (+0.036 so với E1) và chi phí tính toán VRAM/thời gian.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
- **Độ rộng của khoảng tin cậy CI95:**
  - Tập test gồm 311 ảnh với 583 box là kích thước mẫu tương đối nhỏ. Khoảng tin cậy bootstrap 95% của F1 có biên độ dao động khoảng ±0.040 đến ±0.045 (độ rộng khoảng tin cậy lên tới 0.081 – 0.083).
- **Giới hạn kết luận:**
  - Khi hai setup có mức chênh lệch F1 nhỏ hơn độ rộng của CI95 (ví dụ E2 đạt 0.504 và E3 đạt 0.512, chênh nhau 0.008 nhưng CI95 của chúng chồng lấn mạnh lên nhau), ta không thể khẳng định một cách có ý nghĩa thống kê (statistical significance) rằng E3 thực sự vượt trội hơn E2.
  - Chỉ những bước nhảy vọt vượt ra ngoài khoảng tin cậy (như từ E0 lên E1 với chênh lệch F1 > 0.40) mới cho phép rút ra kết luận chắc chắn.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
- **Lớp có kết quả kém nhất:** Lớp `crack` (vết nứt, F1 ≈ 0.412–0.455).
- **Nguyên nhân chi tiết:**
  1. *Kích thước và hình dạng box:* Vết nứt có hình dạng thanh mảnh, kéo dài ngoằn ngoèo, diện tích trung vị rất nhỏ (chỉ chiếm ~0.7% diện tích ảnh). Khi ảnh 640×360 bị nén thành 220 vision tokens, thông tin chi tiết biên dạng ở mức điểm ảnh bị mất mát đáng kể, khiến mô hình khó xác định chính xác 4 toạ độ bao quanh vết nứt để đạt IoU >= 0.5.
  2. *Độ nhập nhằng ngữ nghĩa (Semantic Ambiguity):* Các vết nứt rất dễ bị nhầm lẫn với vệt sơn đường mờ, khe co giãn bê tông, hoặc bóng đổ của cây cối/cột điện trên mặt đường, dẫn đến tỷ lệ False Positives cao.
  3. *So sánh với các lớp khác:* Ngược lại, lớp `manhole` (nắp cống, F1 ≈ 0.562–0.612) có hình dạng hình học tròn/chữ nhật rõ ràng, tương phản cao với mặt đường, nên mô hình nhận diện chính xác nhất.
- **Ảnh minh hoạ:** Đã lưu 6 ảnh phân tích dự đoán đúng/sai vào `report/images/` (các mẫu với box xanh lá Ground Truth `GT:` và box đỏ Prediction `Pred:`):
  1. `report/images/sample_1.jpg`: Nhận diện ổ gà (pothole) chính xác cao (True Positive, IoU > 0.85, hai box GT và Pred gần như trùng khớp).
  2. `report/images/sample_2.jpg`: Nhận diện vết nứt (crack) - Vết nứt bên trái phát hiện đúng (TP), nhưng bóng đổ của cây bên lề đường bị đoán nhầm thành crack (False Positive).
  3. `report/images/sample_3.jpg`: Nhận diện nắp cống (manhole) - Dạng hình học tròn/bầu dục rõ ràng và độ tương phản cao giúp mô hình định vị chính xác hoàn hảo (True Positive).
  4. `report/images/sample_4.jpg`: Bối cảnh hỗn hợp - Ổ gà to ở cự ly gần được nhận diện tốt (TP), nhưng vết nứt nhỏ li ti ở xa bị bỏ sót hoàn toàn (False Negative).
  5. `report/images/sample_5.jpg`: Thách thức vết nứt ở khoảng cách xa - Độ phân giải thấp khi bị nén vào 220 visual tokens khiến box dự đoán bị lệch biên dạng (IoU < 0.5).
  6. `report/images/sample_6.jpg`: Đa đối tượng (Multi-class) - Cả nắp cống (manhole) và ổ gà (pothole) xuất hiện đồng thời, mô hình phân loại và định vị chính xác cả 2 nhãn.

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
- **Biến ablation lựa chọn:** Khảo sát ảnh hưởng của **LoRA Rank** (rank r=4, alpha=8 so với baseline r=16, alpha=32 trên setup E1).
- **Kết quả thu được:**
  - Khi giảm rank từ r=16 xuống r=4: Số tham số trainable giảm từ 20.4M xuống còn 5.1M (~giảm 75%), VRAM giảm xuống 7.90 GB và thời gian huấn luyện giảm còn 16.5 phút.
  - Điểm F1 ở rank r=4 đạt 0.452 (giảm 0.016 so với r=16 là 0.468), nhưng vẫn duy trì ở mức cao và vượt trội hoàn toàn so với E0 (0.042).
- **Độ tin cậy của kết luận:**
  - Khoảng tin cậy CI95 của r=4 [0.411, 0.493] và r=16 [0.428, 0.509] có sự giao thoa rất lớn (>70% diện tích chồng lấn). Điều này cho thấy với bài toán định vị hư hại trên tập dữ liệu này, việc tăng rank từ 4 lên 16 đem lại sự cải thiện nhỏ nhưng không mang tính đột phá tuyệt đối, chứng minh rằng rank nhỏ r=4 vẫn đủ năng lực biểu diễn cho tác vụ thích ứng định dạng JSON grounding.

## Nhìn lại

- Tỷ lệ kích thước đối tượng trong ảnh dashcam có phương sai rất lớn (từ ổ gà khổng lồ ngay trước mũi xe đến vết nứt nhỏ li ti ở cuối tầm nhìn). Cơ chế nén ảnh tĩnh cố định 220 tokens của Qwen3-VL gặp thách thức lớn với các đối tượng vi mô (small objects).
- Trong thực tế, việc kết hợp giữa VLM để phát hiện vùng quan tâm (ROI) và các mô hình detector chuyên biệt (như YOLOv8/v11) ở độ phân giải gốc có thể là hướng giải quyết toàn diện hơn.
