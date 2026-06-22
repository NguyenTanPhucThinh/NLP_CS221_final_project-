# NLP Demo 3 — Multi-Task Dialogue Analysis

Phân loại đồng thời **cảm xúc** (Emotion Recognition) và **hành động hội thoại** (Dialogue Act Classification) trên từng câu thoại, sử dụng một mô hình BERT dùng chung (multi-task learning).

---

## Bài toán

Cho một câu thoại trong hội thoại hàng ngày, mô hình cần dự đoán **đồng thời** hai nhãn:

| Task | Mô tả | Số lớp |
|---|---|---|
| **Dialogue Act** | Mục đích ngôn ngữ của câu thoại | 4 lớp |
| **Emotion** | Cảm xúc người nói biểu đạt | 7 lớp |

### Nhãn Dialogue Act

| ID | Nhãn |
|---|---|
| 1 | Inform |
| 2 | Question |
| 3 | Directive |
| 4 | Commissive |

### Nhãn Emotion

| ID | Nhãn |
|---|---|
| 0 | No emotion / Neutral |
| 1 | Anger |
| 2 | Disgust |
| 3 | Fear |
| 4 | Happiness |
| 5 | Sadness |
| 6 | Surprise |

---

## Dataset

Dataset được lấy từ **DailyDialog** — một bộ dữ liệu hội thoại tiếng Anh hàng ngày chất lượng cao, được gán nhãn thủ công cho cả Dialogue Act và Emotion trên từng lượt thoại.

- **Nguồn Kaggle:** [DailyDialog — Unlock the Conversation Potential](https://www.kaggle.com/datasets/thedevastator/dailydialog-unlock-the-conversation-potential-in)
- **Bài báo gốc:** Li et al., *DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset*, IJCNLP 2017.

| Tập | Số mẫu (gốc) | Số mẫu (sau cân bằng) | File |
|---|---|---|---|
| Train | 87,170 | 30,027 | `data/train.csv` |
| Validation | — | — | `data/validation.csv` |
| Test | — | 7,740 | `data/test.csv` |

**Cấu trúc file CSV:**

```
text,act,emotion
"Say , Jim , how about going for a few beers after dinner ?",3,0
"What do you mean ? It will help us to relax .",2,0
```

**Thống kê văn bản (tập train sau cân bằng):**
- Độ dài trung bình: ~59 ký tự / ~13 từ
- Độ dài tối đa: 1,412 ký tự (278 từ)
- Median: 46 ký tự / 11 từ

**Xử lý mất cân bằng nhãn:** Nhãn `Neutral` (ID 0) chiếm ~83% tập train gốc (72,143 / 87,170 mẫu). Để giảm mất cân bằng, chỉ giữ lại 15,000 mẫu neutral, giảm tổng tập train xuống 30,027 mẫu.

---

## Pipeline

```
Raw CSV
   │
   ▼
[Tiền xử lý]
 • Map nhãn nguyên thủy → ID liên tục
 • Cân bằng lớp neutral (downsampling về 15,000)
 • Tính class weights (balanced) để dùng trong loss
   │
   ▼
[Tokenization]
 • BERT Tokenizer (bert-base-uncased)
 • Max length = 128 token, padding + truncation
   │
   ▼
[MultiTaskBert]
 • Backbone: BertModel (bert-base-uncased, frozen init → fine-tune)
 • Dropout(p = hidden_dropout_prob)
 • Đầu ra Dialogue Act: Linear(hidden_size → 4)
 • Đầu ra Emotion: Linear(hidden_size → 7)
   │
   ▼
[Loss]
 • CrossEntropyLoss(weight=class_weights) cho mỗi task
 • Tổng loss = loss_act + loss_emotion
   │
   ▼
[Huấn luyện]
 • Optimizer: AdamW (lr = 2e-5)
 • Scheduler: ReduceLROnPlateau (factor=0.1, patience=1)
 • Epochs: tối đa 5, early stopping patience = 3
   │
   ▼
[Đánh giá]
 • Accuracy, Macro F1, Weighted F1 trên tập test
 • Classification report + Confusion matrix cho mỗi task
```

---

## Kiến trúc mô hình

```
MultiTaskBert(nn.Module)
├── bert: BertModel (bert-base-uncased)   [110M params]
├── dropout: Dropout
├── act_classifier: Linear(768 → 4)
└── emotion_classifier: Linear(768 → 7)
```

Cả hai classifier dùng chung vector `[CLS]` từ encoder BERT. Tổng số tham số có thể huấn luyện: ~110M.

**Baseline so sánh:** TF-IDF (bigrams, max_features=10,000) + LinearSVC với class_weight='balanced'.

---

## Kết quả

### BERT Multi-Task (trên tập test)

| Metric | Dialogue Act | Emotion |
|---|---|---|
| **Accuracy** | **82.20%** | **78.76%** |
| **Macro F1** | **0.741** | **0.466** |
| **Weighted F1** | **0.820** | **0.810** |

**Chi tiết Dialogue Act (BERT):**

| Lớp | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Inform | 0.84 | 0.87 | 0.85 | 3,534 |
| Question | 0.93 | 0.94 | 0.93 | 2,210 |
| Directive | 0.67 | 0.72 | 0.70 | 1,278 |
| Commissive | 0.63 | 0.39 | 0.48 | 718 |

**Chi tiết Emotion (BERT):**

| Lớp | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Neutral | 0.94 | 0.81 | 0.87 | 6,321 |
| Anger | 0.34 | 0.44 | 0.39 | 118 |
| Disgust | 0.40 | 0.30 | 0.34 | 47 |
| Fear | 0.31 | 0.29 | 0.30 | 17 |
| Happiness | 0.47 | 0.79 | 0.59 | 1,019 |
| Sadness | 0.24 | 0.42 | 0.31 | 102 |
| Surprise | 0.35 | 0.71 | 0.46 | 116 |

**Lịch sử training (BERT, best epoch = 1):**

| Epoch | Train Loss | Train Act Acc | Train Emo Acc | Val Loss | Val Act Acc | Val Emo Acc |
|---|---|---|---|---|---|---|
| 1 | 0.9039 | 84.23% | 81.77% | 1.0042 | 79.01% | 83.27% |
| 2 | 0.7098 | 87.50% | 86.30% | 1.1479 | 78.57% | 80.18% |
| 3 | 0.5303 | 90.55% | 90.01% | 1.2019 | 78.18% | 82.67% |
| 4 | 0.3378 | 94.33% | 94.06% | 1.3675 | 78.84% | 80.58% |

*Early stopping kích hoạt sau epoch 4.*

---

### Baseline TF-IDF + SVM (trên tập test)

| Metric | Dialogue Act | Emotion |
|---|---|---|
| **Accuracy** | 75.97% | 79.95% |
| **Macro F1** | 0.697 | 0.418 |
| **Weighted F1** | 0.760 | 0.810 |

**BERT cải thiện so với SVM baseline:** +6.2 pp accuracy (Act), +2.5 pp Macro F1 (Emotion).

---

## Cấu trúc dự án

```
NLP_demo_3/
├── data/
│   ├── train.csv          # Tập huấn luyện (~87K mẫu gốc)
│   ├── validation.csv     # Tập kiểm định
│   └── test.csv           # Tập kiểm tra (~7,740 mẫu)
├── main.ipynb             # Notebook chính: EDA, training, evaluation
├── main.py                # Entry point (placeholder)
├── id_to_label.txt        # Ánh xạ ID → nhãn cho cả hai task
├── pyproject.toml         # Cấu hình dự án và dependencies
├── uv.lock                # Lock file của uv
├── .python-version        # Python version (3.12)
└── README.md
```

---

## Yêu cầu môi trường

**Python:** 3.12+

**Cài đặt dependencies (dùng [uv](https://docs.astral.sh/uv/)):**

```bash
uv sync
```

**Dependencies chính:**

| Thư viện | Phiên bản tối thiểu | Mục đích |
|---|---|---|
| `torch` | ≥ 2.12.1 | Deep learning framework |
| `transformers` | ≥ 5.12.1 | BERT model và tokenizer |
| `scikit-learn` | ≥ 1.9.0 | SVM baseline, metrics |
| `pandas` | ≥ 3.0.3 | Xử lý dữ liệu |
| `numpy` | ≥ 2.5.0 | Tính toán số học |
| `matplotlib` | ≥ 3.11.0 | Vẽ biểu đồ |
| `seaborn` | ≥ 0.13.2 | Visualization |
| `tqdm` | ≥ 4.68.3 | Progress bar |

**Lưu ý:** Notebook `main.ipynb` được chạy trên **Google Colab** với GPU. Để chạy lại cần:
1. Mount Google Drive và cập nhật đường dẫn dữ liệu.
2. GPU được khuyến nghị (BERT fine-tuning tốn nhiều bộ nhớ và thời gian).
3. Model checkpoint được lưu tại `best_model.pt` trong quá trình training.

---

## Chạy nhanh

```bash
# Cài môi trường
uv sync

# Mở notebook (cần Jupyter)
jupyter notebook main.ipynb
```

Hoặc mở trực tiếp trên Google Colab và cập nhật đường dẫn Google Drive.
