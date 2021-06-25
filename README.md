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

| Training Set   | PRCC |     | Celeb-reID |   |
|----------|----------|----------|----------|----------|
|                | R1 | mAP  | R1 | mAP |
| ImageNet      | 24.7% | 13.5% | 28.7% | 3.0% |
| Market1501    | 29.0% | 24.3% | 36.7% | 3.7% |
| DukeMTMC      | 28.3% | 24.1% | 40.9% | 4.6% |
| MSMT17        | 26.2% | 24.6% | 43.4% | 5.0% |
| LaST          | 39.3% | 32.6% | 47.0% | 7.0% |

  
    
    




