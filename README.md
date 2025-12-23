###Multi-camera Human Tracking with DeepSORT & Tracklet association

## **Description**

This project tracks human movement across two camera. It is a naive implementation of Multi-camera Multi-object Tracking problem using [this paper](www.sciencedirect.com/science/article/pii/S0925231223006811) as a reference.
It detects humans using YOLOv8, tracks them within each camera using DeepSORT tracklets, and performs cross-camera re-identification (ReID) to maintain consistent global IDs.

---

## **Installation**

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/multi-camera-tracking.git
cd multi-camera-tracking
```

2. **Set up Python environment**

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Optional: Install GPU support for PyTorch**

```bash
# Check your CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## **Pipeline**

1. **Detection**

   * Each camera feed is processed by a YOLO detector configured to detect humans only.
   * Detection outputs bounding boxes and confidence scores.

2. **Per-Camera Tracking**

   * DeepSORT tracker tracks objects within each camera feed.
   * Tracklets are maintained with **EMA embeddings** for temporal smoothing.

3. **Global Identity Assignment**

   * Tracklets are matched across cameras using **embedding similarity**.
   * A global ID store assigns and maintains consistent IDs across all camera feeds.
   * Relative superiority logic ensures IDs are only assigned when matches are clearly better than alternatives.

4. **Visualization**

   * Bounding boxes and global IDs (with similarity scores) are drawn on each frame.
   * Frames are optionally stored for **post-processing or MongoDB export**.

---

## **Usage**

### **Configuration**

* Video sources or webcams can be specified in `main.py`:

```python
cameras = [
    (0, "Camera 1"),   # First webcam
    (1, "Camera 2"),   # Second webcam
]
```

* Global ID parameters and confidence thresholds are configurable in `param.py`:

```python
similarity_threshold = 0.75
tracklet_ema_alpha = 0.9
confidence_threshold = 0.6
```

### **Export Processed Data**

* Specify your DB param in `param.py`
* All processed frame data (bounding boxes, tracklets, global IDs, embeddings) can be exported to **MongoDB** after processing:

```python
from mongo_exporter import bulk_insert_frames

bulk_insert_frames(all_frames, mongo_uri="mongodb+srv://user:pass@cluster.mongodb.net/mydb")
```

---

