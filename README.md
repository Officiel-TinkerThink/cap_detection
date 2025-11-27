### Overview
Computer Vision Project that is self-explained. Intended to train to detect the bounding box and the classification of the bottlecap. The objective of this project is to get the model with high metric score (mAP), and also fast inference. This is would be intended to be deployed on the edge devices. However due to my hardware limitation (I don’t have raspberry pi 5). I only test the running on the cpu.

**Objective:** Detecting Bounding Box and classifying bottle-caps into 3 classes:
  - Light Blue --> 0
  - Dark Blue --> 1
  - Others --> 2

**Constraint & Limitation**
- Very Small data, only contains 12 images (8 train, 4 val)
- Latency time target (5 - 10 ms) --> equivalent to (100-200) fps
- Class Bias --> Number of dataset are leaning toward others

---

```
**Best Model:** `runs/detect/train2/weights/best.pt` → **mAP@0.5 = 0.72**, **Latency (NCNN format): (11 +- 2) ms**
```

### Best Perfomance Model Detail

| Metric                  | Value              |
|-------------------------|--------------------|
| Model                   | YOLOv11n           |
| Image Size              | 320×320            |
| mAP@0.5 (test set)      | **0.72**           |
| mAP@0.5:0.95            | 0.48               |
| Latency (NCNN)    | **(11 +-2) ms** (CPU)    |

---


### Installation

```bash
git clone 
cd cap_detection
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```


### CLI Usage
```bash
# Training
uv run bsort train --config configs/settings.yaml

# Inference
uv run bsort infer --config configs/settings.yaml \
  --image image_path (file or dir)
```
---

### Edge Deployment Inference --> suggested on Raspberry Pi 5
```bash
uv run python tools/cpu_inference.py \
  --model-dir runs/detect/train2/weights/best_ncnn_model \
  --image image.jpg --runs num_runs --num_warmup num_warmup
```
---

**Edge Format Inference Benchmark**
```
Elapsed time --> (11 +- 2) miliseconds      # using NCNN format
```
---

### Model Training & Tracking

All experiments are tracked publicly on **Weights & Biases**:

**Public Dashboard:** https://wandb.ai/spidiebot-personal/bottle_caps  
**Best Run:** `train2_final`  
**Key Observations:**
- Heavy mosaic + HSV augmentation critical for small dataset
- `imgsz=320` gives best speed/accuracy balance
- `lr0=0.001`

---

**Future Improvement or Optimzation**
```
- Using more data would likely to improve performance (mAP or confidence level)
- Size optimization & Latency time could be optimized using pruning, knowledge distillation, and quantization 
```

**Insight**
A model can achieve a high mAP score, but don’t be misled—during real inference, it may still produce very low confidence scores for detected objects. In my case, extending the training for more epochs helped. Even though the mAP curve appeared to plateau, the additional training improved the model’s confidence (i.e., its soft labeling) for each detected object.


