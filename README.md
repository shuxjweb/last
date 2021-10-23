## LaST: Large-Scale Spatio-Temporal Person Re-identification

![](last.jpg)

[[Project]](https://sites.google.com/view/personreid) [[Paper]](https://arxiv.org/abs/2105.15076)

This repository contains the source code for loading the **LaST** dataset and evaluating its generalization. 

## Details
**LaST** is a large-scale dataset with more than **228k** pedestrian images. It is used to study the scenario that pedestrians have a large activity scope and time span. Although collected from movies, we have selected suitable frames and labeled them as carefully as possible. Besides the identity label, we also labeled the clothes of pedestrians in the training set.

* **Train**: **5000** identities and **71,248** images.
* **Val**:   **56** identities and **21,379** images.
* **Test**:  **5806** identities and **135,529** images.

**Note**: You can download LaST from these links: [BaiduPan](https://pan.baidu.com/s/1uwT4XkH9TGzJ2ovgZ23fyA) with passward: **vvfe**. or [Googledrive](https://drive.google.com/file/d/1ksVfVrkvCLXHbz3KXK1N4c0gEGQoc0IB/view?usp=sharing).

## Prerequisites

- Python 3.7
- PyTorch 1.6
- Torchvision 0.7.0
- Cuda10.2

## Experiments
#### Train LaST with BoT
```
python last_train_bot.py --train 1 --data_dir /data/last/ --logs_dir ./20210407_last_bot_base
```


#### Direct Transfer

| Training Set   | PRCC |     | Celeb-reID |   |
|----------|----------|----------|----------|----------|
|                | R1 | mAP  | R1 | mAP |
| ImageNet      | 24.7% | 13.5% | 28.7% | 3.0% |
| Market1501    | 29.0% | 24.3% | 36.7% | 3.7% |
| DukeMTMC      | 28.3% | 24.1% | 40.9% | 4.6% |
| MSMT17        | 26.2% | 24.6% | 43.4% | 5.0% |
| LaST          | 39.3% | 32.6% | 47.0% | 7.0% |

1. Put the pre-trained model in the folder "pre_feat". For example, last_ini_imagenet.pth.
```
./pre_feat/last_ini_imagenet.pth
```
2. Modify the loaded model name as follows:
```
last_model_wts = torch.load(os.path.join('pre_feat', 'last_ini_imagenet.pth'))
```
3. Start Testing
```
python prcc_train_base_last.py --train 0 --data_dir /data/prcc/ --logs_dir ./pre_feat
```  
    
#### Domain Adaptation

| Pre-Training   | PRCC |     | Celeb-reID |   |
|----------|----------|----------|----------|----------|
|                | R1 | mAP  | R1 | mAP |
| ImageNet      | 43.1% | 41.3% | 49.2% | 8.7% |
| Market1501    | 44.3% | 43.1% | 49.3% | 8.7% |
| DukeMTMC      | 43.9% | 44.2% | 49.8% | 8.9% |
| MSMT17        | 43.7% | 44.1% | 51.0% | 9.0% |
| LaST          | 54.4% | 54.3% | 56.1% | 11.7% |    

1. Put the pre-trained model in the folder "pre_feat". For example, last_ini_imagenet.pth.
```
./pre_feat/last_ini_imagenet.pth
```
2. Start Training
```
python prcc_train_base_last.py --train 1 --data_dir /data/prcc/ --logs_dir ./20210205_prcc_base_last_sgd
```  
    
## Citation
Please kindly cite this paper in your publications if it helps your research:
```bibtex
@article{shu2021large,
  title={Large-Scale Spatio-Temporal Person Re-identification: Algorithm and Benchmark},
  author={Shu, Xiujun and Wang, Xiao and Zang, Xianghao and Zhang, Shiliang and Chen, Yuanqi and Li, Ge and Tian, Qi},
  journal={arXiv preprint arXiv:2105.15076},
  year={2021}
}
```
 
## Related Work
We forked the projects in [Person_reID_baseline_pytorch](https://github.com/layumi/Person_reID_baseline_pytorch), [fast-reid](https://github.com/JDAI-CV/fast-reid), [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) and [reid-strong-baseline](https://github.com/michuanhaohao/reid-strong-baseline). Thank the authors for their great work.

## License
The dataset and code are released for academic research use only. If you have questions, please contact [shuxj@mail.ioa.ac.cn](shuxj@mail.ioa.ac.cn)












