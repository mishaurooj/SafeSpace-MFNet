# SafeSpace MFNet: Precise & Efficient Multi-Feature Drone Detection

> Implementation README for the SafeSpace MFNet / MFNet‑FA project described in the IEEE TVT paper “SafeSpace MFNet: Precise and Efficient MultiFeature Drone Detection Network”.

---

## Overview

SafeSpace **MultiFeatureNet (MFNet)** and **MFNet‑FA** are real‑time, camera‑based drone vs. bird detection models designed for challenging, cluttered backgrounds and tiny targets. MFNet adds a **Focus** module to emphasize fine‑grained spatial details, while MFNet‑FA further introduces a **Feature Attention (FA)** module to adaptively weight channels. Three model scales are provided — **S/M/L** — each supporting **multi‑scale detection** (P3/P4/P5).

**Highlights**

- Real‑time UAV & bird detection with strong accuracy in adverse backgrounds (clouds, fog, rain, forest, mountains).
- **Focus module** to recover “unseen” tiny features; **FA module** to emphasize informative channels.
- **Multi‑scale head** (strides 8/16/32) with anchors tuned for small, medium, large targets.
- **AutoML‑driven input size** (optimal found: **320×320**) and **Dynamic Batch Size Adjustment (DBSA)**.
- **Scaled Weight Decay Factor (SWDF)** for better generalization.
- YOLOv5‑style training losses (objectness / classification / localization).

> Reference: IEEE TVT, DOI: 10.1109/TVT.2023.3323313
> Dataset: Roboflow,  https://universe.roboflow.com/detection-axsgy/uav-ce0zg/dataset/11
---

## Results (from the paper)

- **MFNet‑L (Ablation Study 2)** — average **Precision 98.4%**, **Recall 96.6%**, **mAP 98.3%**, **IoU 72.8%**.
- **Bird detection** — **MFNet‑M (Ablation Study 2)** best **Precision 99.8%**.
- **UAV detection** — **MFNet‑L (Ablation Study 2)** best **Precision 97.2%**.
- **Efficiency option** — **MFNet‑FA‑S (Ablation Study 3)** offers the best resource efficiency and fast FPS.

*(Numbers above summarize the paper’s reported metrics; see the citation section for details.)*

---

## Repository Structure (suggested)

```
.
├── README.md
├── requirements.txt
├── configs/
│   ├── mfnet_s.yaml
│   ├── mfnet_m.yaml
│   └── mfnet_l.yaml
├── data/
│   ├── datasets.yaml          # paths & class names
│   └── samples/               # small sample images (optional)
├── src/
│   ├── models/
│   │   ├── focus.py
│   │   ├── attention.py       # FA module
│   │   ├── backbone.py        # CSP + SPP/SPPF
│   │   ├── neck.py            # FPN + PAN
│   │   └── head.py
│   ├── train.py
│   ├── val.py
│   └── infer.py
└── weights/
    └── (optional pretrained checkpoints)
```

You can adapt the folder names to your codebase — this README assumes a common PyTorch layout.

---

## Installation

1) **Python & CUDA**  
   - Python 3.9–3.11, PyTorch (CUDA recommended).

2) **Clone & install**  
   ```bash
   git clone <your-repo-url>.git
   cd <your-repo-folder>
   pip install -r requirements.txt
   ```

**Minimal `requirements.txt` (example):**
```
torch>=2.1
torchvision>=0.16
opencv-python
numpy
pyyaml
tqdm
matplotlib
```

---

## Data Preparation

- Detection labels follow **YOLO Darknet TXT** (one txt per image).  
- The paper uses **5,105** images (birds + UAVs) from public sources with diverse backgrounds (clear, cloudy, sunny, fog, rain, water, mountains, forest); split **85%/10%/5%** for train/val/test.  
- Class names: `["bird", "uav"]` (order consistent across train/val/test).

Example `data/datasets.yaml`:
```yaml
train: /path/to/train/images
val:   /path/to/val/images
test:  /path/to/test/images

nc: 2
names: [bird, uav]
```

> Tip: If migrating from Roboflow, export in “YOLO Darknet TXT” format and keep folder names stable.

---

## Configs

Each model scale can be configured via YAML:

```yaml
# configs/mfnet_m.yaml (example)
model:
  img_size: 320      # paper’s optimal image size
  anchors: [[8,8],[16,16],[32,32]]   # P3/P4/P5
  backbone:
    csp_blocks: true
    spp: true
    focus: true
  neck:
    fpn: true
    pan: true
  head:
    num_classes: 2

train:
  optimizer: "adam"  # or "sgd"
  lr0: 0.01
  momentum: 0.937
  weight_decay: 0.0005   # SWDF can adjust
  warmup_epochs: 3.0
  batch_size: "auto"     # DBSA logic
  epochs: 300
```

---

## Training

```bash
python -m src.train   --cfg configs/mfnet_m.yaml   --data data/datasets.yaml   --project runs/mfnet_m   --device 0
```

**Notes**

- **DBSA**: If your code supports gradient accumulation, set `--batch_size auto` (or leave empty) to probe the largest feasible batch for your GPU.
- **SWDF**: Implement a per‑layer decay scaler or a schedule (e.g., cosine) to realize the paper’s scaled decay.
- Use **320×320** as a strong default; you can allow AutoML logic to retune if desired.

---

## Evaluation

```bash
python -m src.val   --weights weights/mfnet_m.pt   --data data/datasets.yaml   --img 320
```

Outputs: Precision / Recall / mAP / IoU, plus PR curves and confusion matrices.

---

## Inference

```bash
python -m src.infer   --weights weights/mfnet_l.pt   --source path/to/images_or_video   --img 320   --conf 0.25
```

The head runs detections at three scales (strides **8/16/32**) with NMS for final boxes.

---

## Model Notes

- **Backbone**: CSP blocks + SPP/SPPF; **Focus** module (MFNet) enhances tiny‑feature preservation.
- **Attention**: **FA module** (MFNet‑FA) applies global‑pooling + 2‑layer MLP (ratio=16) with sigmoid to re‑weight channels.
- **Loss**: YOLOv5‑style objectness / cls / box losses; Adam/SGD supported.
- **Anchors**: representative anchors (P3/P4/P5) around **8/16/32** for small/medium/large targets.
- **Scales**: S/M/L variants provide accuracy‑speed trade‑offs.

---

## Reproducing Paper Metrics

To align with the paper’s settings:

- Input size **320×320** (AutoML‑selected).
- Apply **DBSA** to find max batch size within memory.
- Use **SWDF** for regularization scaling.
- Evaluate on the same split and backgrounds (clear, cloudy, sunny, fog, rain, water, mountains, forest).

> Expect best average results with **MFNet‑L**; for resource‑constrained or multi‑target scenes, **MFNet‑FA‑S** is a strong choice.

---

## Frequently Asked Questions

**Q: Do I need both MFNet and MFNet‑FA?**  
A: Start with MFNet‑L for best overall accuracy. Use MFNet‑FA‑S for lightweight, multi‑object real‑time scenarios.

**Q: Can I use other input sizes?**  
A: Yes. 320×320 was optimal in the paper; you can retune via AutoML.

**Q: What datasets are supported?**  
A: Any YOLO‑style dataset (two classes: bird, uav). The paper aggregates several public sources.

---

## Citation

If you use this repository, please cite the paper:

```
@article{Khan2024SafeSpaceMFNet,
  title   = {SafeSpace MFNet: Precise and Efficient MultiFeature Drone Detection Network},
  author  = {Misha Urooj Khan and Mahnoor Dil and Muhammad Zeshan Alam and Farooq Alam Orakazi and Abdullah M. Almasoud and Zeeshan Kaleem and Chau Yuen},
  journal = {IEEE Transactions on Vehicular Technology},
  year    = {2024},
  volume  = {73},
  number  = {3},
  pages   = {3106-3117},
  doi     = {10.1109/TVT.2023.3323313}
}
```

**Related link**: https://github.com/ZeeshanKaleem/MultiFeatureNet

---

## Acknowledgments

We acknowledge the open‑source community for foundational components (e.g., CSPNet, FPN, PAN, SPP/SPPF, YOLO losses).
