# RTSP Demo Guide

## 1. Mục tiêu của guide này

Guide này mô tả cách demo hệ thống trong bối cảnh thực tế:

- kéo luồng RTSP từ camera
- chạy suy luận trực tiếp trên thiết bị biên
- hiển thị overlay gồm bounding box, `track_id`, nhãn hành vi và trạng thái cảnh báo
- tùy chọn republish kết quả sang mediaMTX

Guide này dùng runtime chuẩn:

- [pipeline_product.py](deploy/pipeline/pipeline_product.py)

Không khuyến nghị dùng:

- [pipeline.py](deploy/pipeline/pipeline.py)

cho demo RTSP mới.

## 2. Khi nào nên dùng `pipeline_product.py`

`pipeline_product.py` được viết để giữ nguyên logic AI của pipeline cũ nhưng cải thiện đường chạy live.

### Logic được giữ nguyên

- detector: PicoDet
- tracker: OC_SORT
- classifier: PPLCNet_x1_0
- `Temporal Buffer`
- `Behavior State Machine`
- logic cảnh báo Telegram
- kiểu vẽ `minimal`

### Phần được thay đổi để phù hợp demo/live

1. RTSP được nhận diện đúng là live source.
2. Reader thread cho live dùng queue nhỏ và ưu tiên frame mới.
3. Có reconnect khi nguồn RTSP chập chờn.
4. Timestamp cảnh báo của RTSP/camera bám theo thời gian thực.
5. Có cleanup cache/state khi mất track hoặc reconnect.
6. Có xử lý an toàn hơn cho crop alignment và output shutdown.
7. RTSP hiện có thể preview trực tiếp trên máy local mà không cần republish.

Điều này làm `pipeline_product.py` có độ trễ tốt hơn cho demo mà không thay đổi lõi nghiệp vụ.

## 3. Khuyến nghị demo

### Chế độ A: RTSP local preview

Đây là chế độ nên dùng nếu:

- máy chạy AI cũng là máy trình chiếu demo
- mục tiêu là thấy hệ thống đang xử lý trực tiếp
- muốn giảm độ trễ tối đa

Ưu điểm:

- không cần mediaMTX
- không cần encode lại RTSP output
- không thêm buffer ở media server hoặc player
- logic AI giữ nguyên

Nhược điểm:

- chỉ xem trực tiếp trên máy chạy pipeline

### Chế độ B: RTSP republish qua mediaMTX

Chỉ nên dùng nếu:

- cần xem từ máy khác
- cần đưa vào VLC, ffplay hoặc dashboard khác

Ưu điểm:

- linh hoạt khi demo từ xa
- dễ chia sẻ luồng overlay cho người khác xem

Nhược điểm:

- tăng độ trễ
- tăng tải encode và I/O
- dễ tích backlog nếu tổng pipeline chậm hơn tốc độ camera

## 4. Kết luận thực dụng cho demo

Nếu demo diễn ra ngay trên máy chạy AI:

- **nên bỏ `--pushurl`**

Vì:

- bỏ `pushurl` không làm thay đổi mô hình
- không làm thay đổi logic cảnh báo
- không làm thay đổi kết quả nhận dạng
- chỉ loại bỏ phần republish video, là phần làm tăng trễ nhưng không tăng độ chính xác

Nói ngắn gọn:

- `push stream` là thành phần **truyền tải**
- không phải thành phần **suy luận**

Do đó, nếu mục tiêu demo là “mô hình đang xử lý đúng”, bỏ push stream là lựa chọn hợp lý.

## 5. Vì sao RTSP republish thường trễ hơn benchmark video file

Benchmark video file chủ yếu đo:

- decode file cục bộ
- detector
- tracker
- classifier
- visualize

Demo RTSP republish đo thêm:

- buffer camera
- decode RTSP đầu vào
- visualize
- encode output H.264
- push RTSP sang mediaMTX
- buffer của mediaMTX
- buffer của player

Vì vậy, dù model là lightweight, tổng hệ thống vẫn có thể trễ nhiều nếu:

- tốc độ xử lý nhỏ hơn FPS nguồn vào
- live queue không được kiểm soát
- output stream phải encode lại
- player giữ buffer mặc định quá lớn

## 6. Chuẩn bị môi trường

### 6.1. Kích hoạt môi trường

```powershell
conda activate paddle_det
```

### 6.2. Kiểm tra model runtime

Phải tồn tại:

- [picodet_m_416_classroom](output_inference/README.md)
- [pplcnet_behavior](output_inference/README.md)

Nếu repo vừa được clone từ GitHub, hãy khôi phục artifacts theo:

- [docs/ARTIFACTS.md](docs/ARTIFACTS.md)

### 6.3. Kiểm tra config

Config runtime:

- [infer_cfg_pphuman.yml](deploy/pipeline/config/infer_cfg_pphuman.yml)

## 7. Chạy demo RTSP local preview

Đây là lệnh khuyến nghị nhất cho demo tại chỗ:

```powershell
python deploy/pipeline/pipeline_product.py `
  --config deploy/pipeline/config/infer_cfg_pphuman.yml `
  --rtsp "rtsp://<user>:<password>@<camera-ip>/Streaming/Channels/101" `
  --device gpu `
  -o TELEGRAM_ALERT.enable=False
```

Kết quả mong đợi:

- pipeline mở cửa sổ preview local
- mỗi người được vẽ bounding box
- có `ID`, nhãn hành vi và màu trạng thái
- không republish nên độ trễ thấp hơn

Thoát demo:

- nhấn `q` trên cửa sổ preview

## 8. Chạy demo RTSP republish qua mediaMTX

### 8.1. Khởi động mediaMTX

Ví dụ nếu mediaMTX đang lắng nghe ở `8554`:

```powershell
mediamtx.exe
```

### 8.2. Chạy pipeline

```powershell
python deploy/pipeline/pipeline_product.py `
  --config deploy/pipeline/config/infer_cfg_pphuman.yml `
  --rtsp "rtsp://<user>:<password>@<camera-ip>/Streaming/Channels/101" `
  --device gpu `
  --pushurl "rtsp://127.0.0.1:8554/output"
```

Pipeline sẽ in ra đường dẫn output thực tế ở console.

Với `pipeline_product.py`, tên stream không còn là `101` như bản cũ, mà được tạo từ source RTSP theo dạng an toàn. Hãy lấy đúng URL được in ra ở console.

### 8.3. Mở stream output

Ví dụ bằng ffplay:

```powershell
ffplay -fflags nobuffer -flags low_delay -framedrop -rtsp_transport tcp "rtsp://127.0.0.1:8554/output/..."
```

Hoặc VLC:

- mở Network Stream
- dán đúng URL output mà pipeline đã in ra

## 9. Khuyến nghị cấu hình cho demo

### Muốn giữ logic AI đầy đủ nhưng giảm trễ tối đa

Khuyến nghị:

- dùng `pipeline_product.py`
- giữ `visual=True`
- bỏ `--pushurl`
- tắt Telegram nếu không cần trình diễn cảnh báo ra điện thoại:

```powershell
-o TELEGRAM_ALERT.enable=False
```

### Muốn demo đầy đủ cả Telegram

Giữ:

- `visual=True`
- `TELEGRAM_ALERT.enable=True`

Nhưng vẫn nên:

- bỏ `--pushurl` nếu không cần xem trên máy khác

### Chỉ dùng `--pushurl` khi

- cần xem từ VLC ở máy khác
- cần record hoặc phát lại qua media server
- cần trình diễn “kết quả AI dưới dạng RTSP output”

## 10. Giải thích ngắn về độ trễ

Nếu camera là `12 FPS`, mỗi frame đến sau khoảng:

- `83.3 ms`

Nếu tổng pipeline thực tế chỉ xử lý được `7 FPS`, mỗi frame mất khoảng:

- `142.9 ms`

Chênh lệch khoảng:

- `59.6 ms/frame`

Khoảng chênh này sẽ cộng dồn thành backlog. Vì vậy republish RTSP có thể trễ rất lớn dù model riêng lẻ khá nhẹ.

## 11. Checklist xử lý trễ lớn

Nếu stream output trễ quá nhiều, kiểm tra theo thứ tự:

1. Có đang dùng `pipeline_product.py` hay không.
2. Có thật sự cần `--pushurl` hay không.
3. Có đang bật Telegram trong lúc demo hay không.
4. Player có đang mở với low-buffer mode hay không.
5. mediaMTX có đang buffer quá nhiều hay không.
6. Máy có đang encode bằng CPU hay không.
7. Camera RTSP có phải main stream quá nặng hay không.

## 12. Gợi ý demo thực tế

### Kịch bản 1: demo tại chỗ, ít trễ nhất

```powershell
python deploy/pipeline/pipeline_product.py `
  --config deploy/pipeline/config/infer_cfg_pphuman.yml `
  --rtsp "rtsp://<user>:<password>@<camera-ip>/Streaming/Channels/101" `
  --device gpu `
  -o TELEGRAM_ALERT.enable=False
```

### Kịch bản 2: demo tại chỗ, vẫn có Telegram

```powershell
python deploy/pipeline/pipeline_product.py `
  --config deploy/pipeline/config/infer_cfg_pphuman.yml `
  --rtsp "rtsp://<user>:<password>@<camera-ip>/Streaming/Channels/101" `
  --device gpu
```

### Kịch bản 3: demo từ xa qua mediaMTX

```powershell
python deploy/pipeline/pipeline_product.py `
  --config deploy/pipeline/config/infer_cfg_pphuman.yml `
  --rtsp "rtsp://<user>:<password>@<camera-ip>/Streaming/Channels/101" `
  --device gpu `
  --pushurl "rtsp://127.0.0.1:8554/output"
```

## 13. Khi nào vẫn nên giữ push stream

Không nên bỏ `pushurl` nếu mục tiêu demo là:

- camera AI chạy ở một máy
- người xem đứng ở máy khác
- cần ghi nhận output dưới dạng RTSP chuẩn
- cần tích hợp với hệ thống xem live khác

Trong trường hợp đó, republish là hợp lý, nhưng phải chấp nhận:

- độ trễ cao hơn
- độ phức tạp hệ thống lớn hơn

## 14. Kết luận vận hành

Đối với demo trên thiết bị biên, cách vận hành hợp lý nhất thường là:

- giữ full pipeline AI
- bỏ bớt các khâu truyền tải không cần thiết
- ưu tiên hình ảnh đang xử lý tại chỗ hơn là republish RTSP

Với repo này, điều đó đồng nghĩa:

- dùng [pipeline_product.py](deploy/pipeline/pipeline_product.py)
- ưu tiên RTSP local preview
- chỉ dùng mediaMTX khi thật sự cần remote display
