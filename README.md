Decoupled Spatiotemporal Transformer with Graph Convolution for Enhanced 3D Human Pose Estimation

This is the official repository for the paper:

> **Decoupled Spatiotemporal Transformer with Graph Convolution for Enhanced 3D Human Pose Estimation**  
> Guowei Zhong, Chenglong Li, Xinyan Gao, Jinxiao Zhang  
> *Shandong Jianzhu University & Shandong Huayun 3D Technology Co., Ltd.*  

## 🎯 Overview

DSGTFormer is a novel Transformer-based architecture for 3D human pose estimation from monocular videos. It introduces:

- **Temporal-Graph Decoupled SpatioTemporal Attention (DSGT)**: A dual-branch architecture that uses GCN for spatial modeling and self-attention for temporal modeling.
- **Spatiotemporal Interaction Attention (STA)**: A cross-attention module for deep fusion of spatiotemporal features.
- **State-of-the-art performance** on Human3.6M and MPI-INF-3DHP datasets.

---

## 📁 Repository Structure

```
DSGTFormer/
├── checkpoint/                 # Pre-trained model weights
├── dataset/                   # Data loaders for Human3.6M and MPI-INF-3DHP
├── common/                    # Utility functions and basic operations
├── model/                     # TGSTFormer network architecture
│   ├── tgst_pe_3dhp.py        # Main model  
│   └── DSGTFormer.py         # Main model
├── run.py               # Training and evaluation script
├── demo/                     # Demo for in-the-wild videos
└── README.md
```

---

## ⚙️ Dependencies

- Python 3.10+
- PyTorch ≥ 1.8
- NumPy
- Matplotlib ≥ 3.1.0

Install with:
```bash
pip install torch numpy matplotlib
```

---

## 📊 Datasets

### Human3.6M
Download and set up following [VideoPose3D](https://github.com/facebookresearch/VideoPose3D/blob/master/DATASETS.md).

### MPI-INF-3DHP
Set up as in [P-STMO](https://github.com/paTRICK-swk/P-STMO).

---

## 🚀 Training & Evaluation

### Human3.6M

**Training:**
```bash
python run.py -f 27 -b 128 --train 1 --layers 6 -s 3
```

**Evaluation (CPN-detected 2D poses):**

```bash
# 27 frames
python run.py -f 27 -b 128 --train 0 --layers 6 -s 1 -k 'cpn_ft_h36m_dbb' --reload 1 --previous_dir ./checkpoint/tgstformer_27.pth

# 81 frames
python run.py -f 81 -b 128 --train 0 --layers 6 -s 1 -k 'cpn_ft_h36m_dbb' --reload 1 --previous_dir ./checkpoint/tgstformer_81.pth

# 243 frames
python run.py -f 243 -b 128 --train 0 --layers 6 -s 1 -k 'cpn_ft_h36m_dbb' --reload 1 --previous_dir ./checkpoint/tgstformer_243.pth
```

**Performance on Human3.6M (CPN input):**

| Frames | MPJPE (P1) ↓ | P-MPJPE (P2) ↓ |
| ------ | ------------ | -------------- |
| 27     | 43.4 mm      | 34.3 mm        |
| 81     | 41.7 mm      | 32.9 mm        |
| 243    | **39.8 mm**  | **32.0 mm**    |

### MPI-INF-3DHP

**Evaluation:**
```bash
python run_3dhp.py --train 0 --frames 81 -b 128 -s 1 --reload 1 --previous_dir ./checkpoint/tgstformer_3dhp_81.pth
```

**Performance on MPI-INF-3DHP:**

| Frames | PCK ↑    | AUC ↑    | MPJPE ↓     |
| ------ | -------- | -------- | ----------- |
| 9      | 98.5     | 82.6     | 24.3 mm     |
| 27     | 98.7     | 83.8     | 22.1 mm     |
| 81     | **98.8** | **84.5** | **20.7 mm** |

---

## 🧪 Demo on Custom Videos

1. Download YOLOv3 and HRNet weights from [here](https://drive.google.com/drive/folders/1_ENAMOsPM7FXmdYRbkwbFHgzQq_B_NQA) into `./demo/lib/checkpoint/`.
2. Place your video in `./demo/video/`.
3. Run:
```bash
python demo/vis.py --video your_video.mp4
```

---

## 🙏 Acknowledgement

This codebase refers to the following repositories:

- [VideoPose3D](https://github.com/facebookresearch/VideoPose3D)
- [STCFormer](https://github.com/...)
- [MHFormer](https://github.com/Vegetebird/MHFormer)
- [MixSTE](https://github.com/JinluZhang1126/MixSTE)

We thank the authors for their contributions.

---

## 📧 Contact

For questions, please contact:  

Guowei Zhong –2024115012@stu.sdjzu.edu.cn 

or open an issue in this repository.
