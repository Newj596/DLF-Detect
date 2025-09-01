# 🌫️ DLF-Detect: Boosting  Object Detection Performance in Foggy Days by Unsupervised Image Restoration
## 🛠️ Overall Framework
This work presents a novel **Dual-stream Learning Framework (DLF)** that incorporates a **Zero-referenced Image Dehazing subnet (ZIDnet)** into a SOTA detector. By designing image restoration losses based on the atmosphere scattering model, ZIDnet can work in an unsupervised manner. The unsupervised design significantly expands the applicability of multi-task learning from synthetic image pairs to real-world scenarios where only hazy images are available. Additionally, a feature selection module is newly introduced to emphasize informative features while suppressing noisy responses under foggy conditions. Furthermore, to mitigate conflicts between the image restoration and object detection losses during joint training, we introduce a dynamic task balancing method to automatically adjust the contributions of the image restoration and object detection tasks.
![](https://raw.githubusercontent.com/Newj596/DLF-Detect/main/ovf2.png)
## 🧠 Contributions
1) We present a multi-task learning network that improves the accuracy of foggy object detection by unsupervised image restoration.

2) We design an unsupervised restoration subnet based on ASM. Three prior-based loss functions are adopted to ensure that DLF learns haze-free features.

3) We propose a novel feature selection module for robust object detection, which retains informative features and suppresses noisy ones.

## 🌁 Visual Results on Foggy-Cityscapes Dataset (Synthetic Fog)
![](https://raw.githubusercontent.com/Newj596/DLF-Detect/main/fogcity.png)
## 🌙 Visual Results on RTTS Dataset (Real Fog)
![](https://raw.githubusercontent.com/Newj596/DLF-Detect/main/rtts.png)

## ⚙️ Installation
Following the installation instructions as YOLO v7 [link](https://github.com/WongKinYiu/yolov7) 
```
cd DLF-Detect\
pip install -r requirements.txt
```
## 📦 Datasets

| RTTS      | Foggy-Cityscapes      |
|------------|------------|
| [link](https://pan.baidu.com/s/1IYkX2B31rSkji55-12TZVg?pwd=yba2) | [link](https://pan.baidu.com/s/1yXBVsci0IVGf78p6mA7Rlw?pwd=a56q) |

## 📦 Weights

| RTTS      | Foggy-Cityscapes      |
|------------|------------|
| [link](https://pan.baidu.com/s/1vB4A07L45sEcGuV5-GDQGA?pwd=s3wx) | [link](https://pan.baidu.com/s/1vry4VtzvK1ec82IpmteYjQ?pwd=6ebg) |

## 🚀 Training DFL on RTTS/Foggy-CityScapes Dataset
```
python train.py --weights yolov7_training.pt --cfg yolov7.yaml --data hybrid_rtts.yaml/hybrid_fogcity.yaml --epochs 30 --hyp data/hyp.srach.custom.yaml --device 0
```

## 🎯 Validating DFL on RTTS/Foggy-CityScapes Dataset
```
python val.py --weights best.pt --cfg yolov7.yaml --data hybrid_rtts.yaml/hybrid_fogcity.yaml  --device 0
```

## 🔍 Image Restoration and Detecting Objects with DFL
```
python detect.py --weights best.pt --cfg yolov7.yaml --data hybrid_rtts.yaml/hybrid_fogcity.yaml --source \your_path --device 0 
```

## ⚙️ Implementation Details
The backbone network for DLF is a modified YOLOv7 with FSM. During training, images are resized to a fixed size of 640 for RTTS. Since Foggy-Cityscapes contains images that have a higher resolution, they are resized to 1024 for training and testing. Various data augmentations are employed, e.g., horizontal flipping, multi-scale cropping, image transformation, and image mosaic. Parameters of DLF are initialized with pre-trained weights on the COCO dataset and trained with Adam in 30 epochs. For DLF tested on RTTS, the learning rate starts with 0.0001 and linearly reduces to 0.00001. The batch size is set as 24. While tested on Foggy-Cityscapes, DLF has an initial learning rate of 0.0005 and a batch size of 8. We use PyTorch for our experiments and run it on a single Nvidia RTX 4090 card.
