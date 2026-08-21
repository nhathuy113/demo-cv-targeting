# Level 2 — Detect "tank địch" (custom class) — Thiết kế chi tiết

## 0. Mục tiêu & ràng buộc

- Input: vẫn là frame từ `capture.py` (không đổi).
- Output: bounding box gán class `tank_enemy` (và `tank_ally` nếu tách được), thay vì proxy "person" của Level 1.
- Ràng buộc quan trọng nhất: **COCO không có class "tank"** → không có shortcut, bắt buộc phải tự thu thập + label + train. Đây là phần tốn thời gian nhất trong cả 3 level, không phải phần code.
- Interface không đổi: `Detector.detect()` giữ nguyên chữ ký, `geometry.py`/`overlay.py`/`main.py` không cần sửa gì — chỉ đổi model file + class list + `--target-classes`.

## 1. Data collection — chi tiết

### 1.1 Nguồn dữ liệu
- Capture từ chính game bạn chơi ở chế độ **single-player/campaign/replay** (không phải multiplayer online).
- Cách lấy: chạy `capture.py` độc lập (không cần detector) để lưu frame ra ổ đĩa mỗi N giây trong lúc chơi bình thường, hoặc trích frame từ replay/video đã quay sẵn bằng `ffmpeg -i replay.mp4 -vf fps=1 frames/%04d.jpg`.
- **Tránh trùng lặp gần giống nhau**: nếu lấy từ video liên tục, dùng fps thấp (0.5-1 frame/giây) thay vì lấy mọi frame — nhiều frame liên tiếp gần như giống hệt nhau không thêm thông tin, chỉ làm dataset phình to vô ích và risk model học "thuộc lòng" 1 vài cảnh thay vì tổng quát hoá.

### 1.2 Độ đa dạng cần có (quan trọng hơn số lượng tuyệt đối)
Một dataset 300 ảnh đa dạng tốt hơn 1000 ảnh na ná nhau. Checklist đa dạng:
- **Khoảng cách**: tank ở gần (chiếm > 30% khung hình), trung bình, xa (< 5% khung hình).
- **Góc nhìn**: từ trước, từ sau, từ bên hông, chéo 45°.
- **Ánh sáng**: ban ngày, hoàng hôn, đêm, trong nhà/hầm nếu game có map indoor.
- **Trạng thái che khuất một phần**: tank nấp sau vật cản, chỉ lộ tháp pháo — **cố ý thu thập cả case này ngay từ Level 2** dù occlusion-robust là mục tiêu chính của Level 3, vì nếu Level 2 dataset chỉ toàn ảnh tank lộ rõ hoàn toàn, model sẽ không bao giờ học được cách nhận diện partial view, và Level 3 augmentation (che nhân tạo) sẽ hiệu quả hơn nếu có sẵn 1 ít case thật làm đối chiếu.
- **Background đa dạng**: nhiều map/địa hình khác nhau nếu game có, tránh model học nhầm "cỏ xanh + tank" thay vì học đặc trưng hình dạng tank.
- **Số lượng tối thiểu đề xuất**: ~300-500 ảnh có annotation cho bản đầu (transfer learning từ pretrained COCO cần ít data hơn train from scratch rất nhiều) — đủ để có baseline, sau đó đánh giá rồi quyết định có cần thu thêm không dựa trên chỗ model sai nhiều nhất (active learning kiểu thủ công).

## 2. Labeling — quy trình cụ thể

### 2.1 Công cụ
- **CVAT** (self-host hoặc cloud) hoặc **Roboflow** (có free tier, export thẳng ra format YOLO) — Roboflow tiện hơn nếu muốn nhanh vì có augmentation + train tích hợp sẵn, CVAT tốt hơn nếu muốn kiểm soát chi tiết/label offline.
- Format annotation: YOLO txt format (`class_id x_center y_center width height`, normalized 0-1) — khớp thẳng với pipeline train của ultralytics, không cần convert.

### 2.2 Class taxonomy — quyết định trước khi label, đổi sau sẽ phải label lại
Có 2 phương án, chọn 1 trước khi bắt đầu:

**Phương án A — 2 class riêng biệt (`tank_enemy`, `tank_ally`)**
- Chỉ khả thi nếu game có khác biệt trực quan rõ ràng (màu sơn, phù hiệu, kiểu dáng khác giữa phe) — nếu không, model sẽ không học được, chỉ đoán ngẫu nhiên giữa 2 class trông giống hệt nhau.
- Ưu điểm: output trực tiếp dùng được, không cần logic phân loại thêm.

**Phương án B — 1 class chung (`tank`) + không cố phân biệt phe bằng CV**
- Dùng khi 2 phe trông giống nhau về hình dáng (chỉ khác texture/skin nhỏ).
- Nhược điểm: overlay sẽ hiện "Mục tiêu" chung chung, không biết địch hay quân mình — cần quyết định có chấp nhận được không cho mục đích demo/portfolio.
- **Tránh phương án dựa vào team-ID đọc từ HUD/UI game** để bù cho hạn chế này — cách đó lệch sang đọc dữ liệu nội bộ (gần overlay-reading/memory-adjacent), không còn thuần computer vision, sẽ làm loãng câu chuyện "đây là CV project" khi trình bày.

→ **Khuyến nghị**: quan sát thử 10-20 ảnh mẫu trước, nếu phân biệt được bằng mắt trong < 1 giây thì chọn Phương án A, nếu phải nhìn kỹ mới phân biệt được thì chọn Phương án B (an toàn hơn, tránh model học nhiễu).

### 2.3 Edge case cần quy ước rõ khi label (tránh label không nhất quán giữa các ảnh)
- Tank chỉ lộ < 20% diện tích (gần như hoàn toàn bị che): có label hay bỏ qua? → khuyến nghị: **vẫn label** nếu bằng mắt người vẫn nhận ra được đó là tank, đây chính là data quý cho Level 3.
- Nhiều tank chồng lên nhau trong 1 khung hình: label riêng từng box, không gộp.
- Tank ở rìa khung hình, bị cắt một phần bởi mép ảnh: label phần nhìn thấy được, không đoán phần ngoài khung hình.

## 3. Dataset structure

```
dataset/
  images/
    train/   (70%)
    val/     (20%)
    test/    (10%)
  labels/
    train/
    val/
    test/
  data.yaml   # trỏ đường dẫn + class names, format chuẩn ultralytics
```

- Split theo **map/session**, không split ngẫu nhiên từng ảnh riêng lẻ — nếu lấy nhiều frame liên tiếp từ cùng 1 đoạn gameplay, chia ngẫu nhiên dễ khiến train và test có ảnh gần giống hệt nhau (leakage), làm mAP báo cáo cao giả tạo, không phản ánh khả năng tổng quát hoá thật.

## 4. Training — chi tiết kỹ thuật

- **Base**: `yolov8n.pt` (giữ nano để tốc độ inference thấp, phù hợp real-time overlay — không đổi sang model to hơn dù có thể tăng accuracy, vì latency là ưu tiên đã chốt ở Level 1).
- **Transfer learning**: 
  - Dataset nhỏ (~300-500 ảnh) → freeze backbone, chỉ train detection head vài epoch đầu, sau đó unfreeze toàn bộ train tiếp với learning rate thấp hơn — giảm risk overfit khi data ít.
  - Augmentation: dùng augmentation mặc định của ultralytics (mosaic, flip, hsv jitter) — không cần custom thêm ở bước này, occlusion augmentation để dành riêng cho Level 3.
- **Export**: dùng đúng script đã dùng ở Level 1 (`model.export(format='onnx', imgsz=640, simplify=True)`) — không đổi pipeline export, chỉ đổi file weight đầu vào.

## 5. Evaluation

- **mAP theo từng class** (không chỉ overall) — nếu chọn Phương án A (2 class), quan trọng để biết class nào yếu hơn (thường class ít data hơn sẽ yếu hơn rõ rệt).
- **Confusion giữa 2 class** (nếu Phương án A): confusion matrix riêng cho enemy/ally — nếu nhầm lẫn cao, đó là tín hiệu nên chuyển sang Phương án B thay vì cố train thêm.
- **So sánh với Level 1 baseline**: recall trên chính tập test có tank, so với recall của model Level 1 (COCO pretrained, class "person" không áp dụng được cho tank) — về bản chất Level 1 sẽ có recall gần 0 với tank, nên so sánh này chủ yếu để chứng minh giá trị của việc fine-tune, dùng trong phần trình bày/portfolio.

## 6. Rủi ro & cách giảm thiểu

| Rủi ro | Dấu hiệu | Cách xử lý |
|---|---|---|
| Data quá ít, model overfit | train loss giảm nhưng val mAP không tăng hoặc giảm | Thu thêm data đa dạng hơn (không phải nhiều hơn), giảm epoch, tăng augmentation |
| Model học nhầm background thay vì tank | detect sai trên map mới chưa từng thấy trong train | Đảm bảo dataset có nhiều map/địa hình khác nhau (mục 1.2) |
| 2 class enemy/ally nhầm lẫn cao | confusion matrix chéo lớn | Chuyển Phương án A → B, hoặc thu thêm data tập trung riêng case dễ nhầm |
| Label không nhất quán (người khác nhau label khác quy ước) | mAP thấp bất thường dù data đủ | Áp dụng đúng edge-case convention ở mục 2.3, review lại 1 lượt trước khi train |

## 7. Việc cần bạn chốt trước khi bắt tay thu thập data

- [ ] Game cụ thể là game nào? (ảnh hưởng cách capture — nếu là game có replay file thì trích frame từ đó dễ hơn chơi live)
- [ ] Phương án A (2 class enemy/ally) hay B (1 class chung)? — quan sát vài ảnh mẫu trước theo gợi ý ở mục 2.2 rồi quyết định.
- [ ] Có sẵn hardware để train không (GPU local hay cần dùng Colab/cloud)? — ảnh hưởng tới việc chọn epoch/batch size thực tế.