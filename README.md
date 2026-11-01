# Adversarial IDS XAI

Dự án nghiên cứu khả năng chống tấn công adversarial của mô hình phát hiện xâm nhập mạng (IDS), kết hợp với SHAP để phân tích sự thay đổi trong lời giải thích của mô hình.

Mục tiêu là tái hiện hướng tiếp cận SHAP trong bài báo, sau đó mở rộng cho dữ liệu chuỗi thời gian IDS bằng detector dựa trên các đặc trưng thay đổi SHAP, entropy và Gini.

## Nội dung chính

* Tiền xử lý dữ liệu IDS dạng chuỗi thời gian.
* Tạo mẫu bằng cửa sổ trượt độ dài 5.
* Huấn luyện mô hình LSTM và GRU.
* Sinh mẫu adversarial bằng FGSM, BIM và PGD.
* Tính SHAP bằng Kernel SHAP và Gradient SHAP.
* Xây dựng detector theo hai chế độ: `pair` và `state`.
* So sánh detector luật, Logistic Regression và Random Forest.
* Phân tích mở rộng bằng entropy và Gini của vector SHAP.

## Kiến trúc xử lý

```text
Raw IDS data
    │
    ▼
Làm sạch, chọn đặc trưng, chuẩn hóa
    │
    ▼
Tạo chuỗi cửa sổ trượt (window = 5)
    │
    ▼
Train LSTM / GRU
    │
    ├── FGSM
    └── PGD
            │
            ▼
       Mẫu adversarial
            │
            ▼
     SHAP clean / adversarial
            │
            ├── Pair detector
            ├── State detector
            └── Entropy / Gini ablation
```

## Cấu trúc thư mục

```text
configs/                 Cấu hình LSTM, GRU và SHAP
data/                    Dữ liệu đầu vào và dữ liệu đã xử lý
logs/                    Log huấn luyện, attack và SHAP
models/                  Mô hình đã huấn luyện
results/                 Feature detector, dự đoán và metrics
scripts/data/            Tiền xử lý và kiểm tra dữ liệu
scripts/training/        Huấn luyện LSTM và GRU
scripts/attack/          FGSM, BIM và PGD
scripts/xai/             Kernel SHAP và Gradient SHAP
scripts/evaluation/      Đánh giá, detector và ablation
scripts/visualization/   Vẽ biểu đồ phân tích SHAP
```

## Dataset

Dự án sử dụng bộ dữ liệu TON-IoT Network Train/Test:

* [TON-IoT Train/Test Network Dataset trên Kaggle](https://www.kaggle.com/datasets/fadiabuzwayed/ton-iot-train-test-network)

## Cài đặt

Khuyến nghị sử dụng Python 3.11 hoặc môi trường tương thích với TensorFlow, PyTorch và SHAP.

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
```



### Sử dụng Anaconda hoặc Miniconda

Tạo một môi trường riêng cho dự án:

```powershell
conda create -n adversarial\_ids\_xai python=3.11 -y
conda activate adversarial\_ids\_xai
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Kiểm tra môi trường đang được sử dụng:

```powershell
conda env list
python --version
where python
```

Mỗi lần làm việc với dự án, kích hoạt lại môi trường:

```powershell
conda activate adversarial\_ids\_xai
```

## Quy trình chạy

### 1\. Chuẩn bị dữ liệu

```powershell
python scripts/data/01\_prepare\_a2\_train\_only.py
python scripts/data/04\_test\_dataloader.py
python scripts/data/05\_audit\_sequences.py
```

`scripts/data/06\_prepare\_v2\_temporal\_split.py` là pipeline tách theo thời gian. Pipeline này chia train/validation/test trước khi tạo cửa sổ và fit preprocessing trên train, phù hợp hơn khi cần kiểm soát rò rỉ giữa các tập.

### 2\. Huấn luyện mô hình IDS

```powershell
python scripts/training/01\_train\_lstm.py
python scripts/training/02\_train\_gru.py
```

### 3\. Đánh giá mô hình sạch

```powershell
python scripts/evaluation/01\_evaluate\_lstm.py
python scripts/evaluation/02\_evaluate\_gru.py
```

### 4\. Sinh mẫu adversarial

```powershell
python scripts/attack/01\_fgsm\_lstm.py
python scripts/attack/01\_fgsm\_gru.py
python scripts/attack/02\_pgd\_lstm.py
python scripts/attack/02\_pgd\_gru.py
```

Nên đánh giá các mức `epsilon = 0.01, 0.05, 0.10, 0.20`. Với PGD cần ghi rõ `epsilon`, `alpha` và số bước `steps` trong kết quả.

### 5\. Tính SHAP

```powershell
python scripts/xai/04\_shap\_kernel\_all.py --help
python scripts/xai/05\_shap\_gradient\_all.py --help
```

SHAP được tính cho mẫu clean và adversarial, sau đó dùng làm đầu vào cho detector.

### 6\. Xây dựng detector features

```powershell
python scripts/evaluation/06\_build\_detector\_features.py --help
```

Ở `pair mode`, mỗi mẫu là một cặp clean–adversarial với các đặc trưng:

```text
drift\_mean, shift\_std, shift\_max, shift\_l2,
top5\_overlap, top10\_overlap, large\_shift\_fraction
```

Ở `state mode`, mỗi vector SHAP là một mẫu độc lập. Đầu vào gồm SHAP attribution đã chuẩn hóa và các đặc trưng tổng hợp như entropy, top-k mass và max mass; nhãn là `0 = clean`, `1 = adversarial`.

`pair` phù hợp để phân tích thay đổi SHAP giữa hai trạng thái. `state` phù hợp hơn cho detector online khi không có mẫu clean tương ứng.

### 7\. Detector luật và detector học máy

```powershell
python scripts/evaluation/07\_rule\_based\_detector.py --help
python scripts/evaluation/08\_train\_learned\_detector.py `
  --detector-dir <detector-directory> `
  --mode pair `
  --classifier logistic
```

Classifier có thể là:

```text
logistic
random\_forest
```

Với `--mode pair`, thư mục cần có `pair\_X.npy`, nhãn pair và các file split tương ứng. Với `--mode state`, cần có `state\_X.npy`, `state\_y.npy` và các file split state.

### 8\. Entropy và Gini SHAP

```powershell
python scripts/evaluation/09\_compute\_shap\_entropy\_gini.py `
  --run-dir <shap-run-directory>
```

Script tính entropy/Gini của vector SHAP clean và adversarial, cùng các giá trị chênh lệch giữa hai trạng thái.

Tạo các bộ đặc trưng ablation:

```powershell
python scripts/evaluation/10\_build\_entropy\_gini\_ablation.py `
  --detector-dir <detector-directory> `
  --entropy-gini-dir <entropy-gini-directory>
```

Các biến thể gồm:

```text
baseline
entropy\_only
gini\_only
entropy\_gini\_only
baseline\_plus\_entropy
baseline\_plus\_gini
extended
```

`baseline` giữ 7 đặc trưng ban đầu. `extended` gồm 7 đặc trưng baseline và 4 đặc trưng entropy/Gini:

```text
entropy\_delta
entropy\_abs\_delta
gini\_delta
gini\_abs\_delta
```

## Metrics

Các kết quả được lưu trong `metrics.json`. Các kết quả:

* ROC-AUC
* Accuracy
* Precision
* Recall
* F1-score
* Specificity
* Attack success rate
* Norm perturbation, đặc biệt là `L∞` và `L2`

Khi so sánh entropy/Gini, cần giữ nguyên attack, epsilon, mô hình IDS, SHAP run, train/validation/test split, classifier và random seed. So sánh chính:

```text
baseline vs entropy\_only
baseline vs gini\_only
baseline vs baseline\_plus\_entropy
baseline vs baseline\_plus\_gini
baseline vs extended
```

## Trích dẫn

Đề tài phát triển và tham khảo từ:

* [Reference paper: Information Security Journal article](https://link.springer.com/article/10.1007/s10207-026-01299-x)

## Figures và kết quả trực quan

Các figures tổng hợp kết quả được lưu trên Google Sheets:

* [Project figures trên Google Drive](https://docs.google.com/spreadsheets/d/1BXJiXE232AF8mbMURToujwBSp3ytzeMq/edit?usp=drive_link&ouid=108932263966024454491&rtpof=true&sd=true)

