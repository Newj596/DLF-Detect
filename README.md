# 🌫️ DLF-Detect: Boosting  Object Detection Performance in Foggy Days by Unsupervised Image Restoration
## 🛠️ Overall Framework
This work presents a novel Dual-stream Learning Framework (DLF) that incorporates a Zero-referenced Image Dehazing subnet (ZIDnet) into a detector. By designing image restoration losses based on the atmosphere scattering model, ZIDnet can work in an unsupervised manner. The unsupervised design significantly expands the applicability of multi-task learning from synthetic image pairs to real-world scenarios where only hazy images are available. Additionally, a feature selection module is newly introduced to emphasize informative features while suppressing noisy responses under foggy conditions. Furthermore, to mitigate conflicts between the image restoration and object detection losses during joint training, we introduce a dynamic task balancing method to automatically adjust the contributions of the image restoration and object detection tasks.
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
## 🚀 Training FiLMN-S/FiLMN-X with C2F on RTTS/ExDark Dataset
### Coarse Training
```
python train.py --weights yolov5s.pt/yolov5x.pt --cfg yolov5s.yaml/yolov5x.yaml --data fog.yaml/light.yaml --epochs 30 --freeze [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23] --device 0
```
### Fine Training
```
python train.py --weights best.pt --cfg yolov5s.yaml/yolov5x.yaml --data fog.yaml/light.yaml --epochs 10 --device 0
```

## 🎯 Validating FiLMN-S/FiLMN-X with DTS on RTTS/ExDark Dataset
```
python val.py --weights best.pt --cfg yolov5s.yaml/yolov5x.yaml --data fog.yaml/light.yaml --task golden_search --device 0
```

## 🔍 Detecting Objects with FiLMN-S/FiLMN-X
```
python detect.py --weights best.pt --cfg yolov5s.yaml/yolov5x.yaml --data fog.yaml/light.yaml --source \your_path --device 0 --conf-thres [confidence determined by DTS]
```

## ⚙️ Implementation Details
The proposed FiLMN is written in PyTorch and based on the YOLO v5 branch of the ultralytics repository. Backbones are pre-trained on COCO dataset. Unless otherwise specified, YOLO v5x is employed as the backbone network throughout this work. To address the trade-off between computational precision and processing latency, we adopt FiLMN with five specialized attention networks. The modulation factor $\gamma$ in focal localization loss is empirically set to 1.2. Since FiLMN is trained in a two-step manner offered by C2F, we first train the attention blocks and detection head with a learning rate of 0.01. Then we fine-tune the overall framework with a lower learning rate of 0.0001. We employ the SGD optimizer to train the proposed network. The training procedure is conducted on an Nvidia RTX4070 GPU with a batch size of 8. In the testing phase, a, b, and \epsilon in DTS are empirically set as 0.01, 0.5, and 0.001.
