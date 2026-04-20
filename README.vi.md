# Giám Sát Hành Vi Trong Lớp Học

[English](README.md)

Repo này dùng để giám sát gần thời gian thực các hành vi kéo dài trong lớp học:

- `sleeping`
- `using_phone`

Pipeline runtime hiện tại:

`PicoDet -> OC_SORT -> PPLCNet_x1_0 -> Temporal Buffer -> Behavior State Machine -> Telegram Alert`

Mục tiêu của hệ thống là cảnh báo ổn định theo từng `track_id`, không phải gán nhãn hành vi độc lập cho từng frame.

## Phạm Vi Repo

Repo này giữ:

- mã runtime
- script triển khai
- công cụ benchmark và vẽ biểu đồ
- tài liệu public

Repo này không lưu trong Git các tài nguyên nặng:

- model Paddle inference
- video mẫu
- output sinh ra khi chạy
- file tạm cục bộ

Tải các tài nguyên đó tại [docs/ARTIFACTS.md](docs/ARTIFACTS.md).

## Môi Trường Đã Kiểm Tra

Cấu hình public hiện tại đã được kiểm tra trên:

- Windows
- Python `3.9.25`
- PaddlePaddle GPU `3.2.2`
- NumPy `1.23.5`
- OpenCV `4.5.5`

Lệnh `paddle.utils.run_check()` đã chạy thành công trong env `paddle_det` trên máy tác giả.

## Cài Đặt

Repo này hỗ trợ 2 cách cài:

1. dùng Conda theo môi trường đã xác minh cho Windows + GPU
2. dùng `venv` + `pip` nếu bạn muốn setup nhẹ hơn

`requirements.txt` cố ý không chứa PaddlePaddle vì gói CPU và GPU phụ thuộc máy của người cài.

### Cách 1: Conda Theo Môi Trường Đã Xác Minh

Đây là cách khuyến nghị nếu bạn muốn bám sát môi trường chạy thực tế của repo.

```powershell
conda env create -f deploy_bundle/environment/environment.demo.yml
conda activate paddle_det
```

Danh sách version đã xác minh nằm tại [deploy_bundle/environment/verified_versions.txt](deploy_bundle/environment/verified_versions.txt).

### Cách 2: `venv` + `pip`

Tạo và kích hoạt virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Cài đúng một gói PaddlePaddle trước:

Nhánh Windows GPU đã được kiểm tra với repo này:

```powershell
python -m pip install paddlepaddle-gpu==3.2.2
```

Nếu chỉ chạy CPU:

```powershell
python -m pip install paddlepaddle==3.2.2
```

Sau đó cài các dependency còn lại:

```powershell
python -m pip install -r requirements.txt
```

Nếu máy của bạn dùng CUDA hoặc nền tảng khác, hãy chọn đúng wheel PaddlePaddle từ tài liệu cài đặt chính thức rồi mới tiếp tục.

### Kiểm Tra Nhanh Sau Khi Cài

```powershell
python -c "import cv2, numpy, paddle, yaml, scipy, PIL, numba; print('Imports OK')"
python -c "import paddle; paddle.utils.run_check()"
```

## Khôi Phục Runtime Artifacts

Repo sẽ chưa chạy được nếu chưa tải các artifact ngoài repo được mô tả trong [docs/ARTIFACTS.md](docs/ARTIFACTS.md).

Tối thiểu cần có:

- `output_inference/picodet_m_416_classroom`
- `output_inference/pplcnet_behavior`
- tùy chọn `test_data/data.mp4` nếu muốn chạy demo mặc định

`output_inference/` là thư mục artifact chuẩn của repo này.

## Chạy Nhanh

### 1. Dùng Deploy Bundle

Demo video mặc định:

```powershell
.\deploy_bundle\run_demo.ps1
```

Launcher CMD thuần:

```bat
deploy_bundle\run_demo.cmd video
```

Webcam:

```powershell
.\deploy_bundle\run_demo.ps1 -SourceType camera -CameraId 0
```

RTSP:

```powershell
.\deploy_bundle\run_demo.ps1 -SourceType rtsp -RtspUrl "rtsp://<user>:<password>@<camera-ip>/Streaming/Channels/101"
```

Chạy ở chế độ chỉ in command:

```powershell
.\deploy_bundle\run_demo.ps1 -DryRun
```

### 2. Chạy Trực Tiếp Script Runtime

Chạy với file video:

```powershell
python deploy/pipeline/pipeline_product.py `
  --config deploy/pipeline/config/infer_cfg_pphuman.yml `
  --video_file test_data/data.mp4 `
  --device gpu
```

Xem trước RTSP:

```powershell
python deploy/pipeline/pipeline_product.py `
  --config deploy/pipeline/config/infer_cfg_pphuman.yml `
  --rtsp "rtsp://<user>:<password>@<camera-ip>/Streaming/Channels/101" `
  --device gpu
```

Republish RTSP qua mediaMTX:

```powershell
python deploy/pipeline/pipeline_product.py `
  --config deploy/pipeline/config/infer_cfg_pphuman.yml `
  --rtsp "rtsp://<user>:<password>@<camera-ip>/Streaming/Channels/101" `
  --device gpu `
  --pushurl "rtsp://127.0.0.1:8554/output"
```

Ghi chú chi tiết hơn về live stream nằm ở [RTSP_DEMO_GUIDE.md](RTSP_DEMO_GUIDE.md).

## Bật Telegram

Telegram đang tắt mặc định.

Để bật:

```powershell
Copy-Item deploy_bundle\secrets\telegram.env.example deploy_bundle\secrets\telegram.env
```

Điền 2 biến trong `deploy_bundle/secrets/telegram.env`:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

Sau đó chạy:

```powershell
.\deploy_bundle\run_demo.ps1 -UseTelegram
```

Không commit `deploy_bundle/secrets/telegram.env`.

## Điểm Vào Chính

- script runtime: [deploy/pipeline/pipeline_product.py](deploy/pipeline/pipeline_product.py)
- launcher public: [deploy_bundle/run_demo.ps1](deploy_bundle/run_demo.ps1)
- config runtime chính: [deploy/pipeline/config/infer_cfg_pphuman.yml](deploy/pipeline/config/infer_cfg_pphuman.yml)
- config deploy bundle: [deploy_bundle/config/demo_final.yml](deploy_bundle/config/demo_final.yml)

Giá trị mặc định quan trọng:

- `MOT.skip_frame_num = 2`
- `ID_BASED_CLSACTION.skip_frame_num = 2`
- `ID_BASED_CLSACTION.crop_mode = full`
- `preview_local = auto`
- `save_visual_output = auto`
- `sleep_warn_seconds = 5.0`
- `sleep_alert_seconds = 12.0`
- `phone_warn_seconds = 6.0`
- `phone_alert_seconds = 15.0`
- `TELEGRAM_ALERT.enable = False`

## Cấu Trúc Repo

- `deploy/`: mã runtime và các module PaddleDetection, PP-Tracking được vendor vào repo
- `deploy_bundle/`: launcher demo sạch và ghi chú môi trường
- `docs/`: tài liệu public như link tải artifact
- `tools/`: công cụ benchmark và vẽ biểu đồ
- `output_inference/`: artifact runtime tải ngoài Git
- `test_data/`: video demo tải ngoài Git
- `output/`: output sinh ra khi chạy, bị loại khỏi Git
- `tmp/`: vùng scratch, bị loại khỏi Git

## Ghi Chú

- Repo public này tập trung vào runtime lớp học, không tập trung vào mã train.
- Có thể vẫn còn mirror model cục bộ ở `models/`, nhưng `output_inference/` mới là nguồn chuẩn khi triển khai.
- Launcher PowerShell ưu tiên cho Windows. Nếu dùng nền tảng khác, hãy chạy `pipeline_product.py` trực tiếp.
