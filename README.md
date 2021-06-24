## LaST: Large-Scale Spatio-Temporal Person Re-identification

![](last.jpg)

[[Project]](https://sites.google.com/view/personreid) [[Paper]](https://arxiv.org/pdf/2105.15076.pdf)

This repository contains the source code for loading the **LaST** dataset and evaluating its generalization. 

## Details
**LaST** is a large-scale dataset with more than **228k** pedestrian images. It is used to study the scenario that pedestrians have a large activity scope and time span. Although collected from movies, we have selected suitable frames and labeled them as carefully as possible. Besides the identity label, we also labeled the clothes of pedestrians in training set.

* **Train**: **5000** identities and **71,248** images.
* **Val**:   **56** identities and **21,379** images.
* **Test**:  **5806** identities and **135,529** images.

**Note**: You can download LaST from this link: [LaST]() with passward:.

## Prerequisites

- Python 3.7
- PyTorch 1.6
- Torchvision 0.7.0
- Cuda10.2

## Experiments

#### Direct Transfer

| Training Set   | PRCC  | Celeb-reID |
|                | R1 | mAP  | R1 | mAP |
|----------|----------|----------|----------|----------|
| ImageNet | 94.8% | 86.6% | 77.2% | 65.6% |
| Market1501    | 86.0% | 74.8% | 52.3% | 61.1% |











