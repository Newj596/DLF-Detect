# 🌫️ DLF-Detect: Boosting  Object Detection Performance in Foggy Days by Unsupervised Image Restoration
## 🛠️ Overall Framework
![](https://raw.githubusercontent.com/Newj596/DLF-Detect/main/ovf2.png)
This work presents a **Fine-grained Latent-domain Mixture-of-expert Network (FiLMN)** to achieve robust object detection in dynamic weather conditions.  We dynamically allocate attention to degradation-specific features through a gating network. A theoretical error bound for domain adaptation is provided. Additionally, a focal localization loss is newly proposed to mine hard examples in extreme weather. Moreover, a two-phase training strategy is introduced to resolve learning rate conflicts between fixed backbones and adaptive experts. Furthermore, we propose a novel dynamic threshold search algorithm for post-processing, which eliminates manual parameter tuning limitations. Evaluations on RTTS and ExDark benchmarks demonstrate marked improvements with gains of **3.8% mAP** in dense fog and **2.0%** in extremely dark environments over state-of-the-art methods. Extensive real-world validation on unmanned surface vehicles in diverse environments demonstrates FiLMN’s generalization capability in the wild.
## 🧠 Methods 
![](https://raw.githubusercontent.com/Newj596/FiLMN/main/imgs/methods.png)
a) We propose FiLMN, a novel architecture that dynamically partitions weather-degraded inputs into latent sub-domains and employs specialized expert modules to handle each sub-domain. A meta-gating mechanism adaptively fuses expert outputs, enabling precise feature recalibration for diverse degradation patterns. The domain adaptation error bound of FiLMN is theoretically given. In addition, a focal distillation loss is formulated to handle hard examples in dynamic weather conditions.

b) To resolve learning rate conflicts between pre-trained backbones and domain-specific components, we introduce a two-stage training paradigm, called C2F for short. In the coarse stage, domain-aware modules (experts and gating networks) are trained at high learning rates to rapidly capture degradation-specific features. In the fine stage, the full network is learned at a reduced learning rate, ensuring stable convergence while preserving global contextual coherence. 

c) We propose a multi-iterative optimization method called DTS on the basis of the golden section selection mechanism. It replaces empirically predefined thresholds through a multi-stage iterative optimization framework to search the best confidence threshold for post-processing. This approach dynamically adapts to varying degradation patterns in adverse weather conditions, enhancing detection performance without manual parameter tuning and additional task-special subnet design required by existing methods.

## 🌁 Visual Results on RTTS Dataset (Foggy)
![](https://raw.githubusercontent.com/Newj596/FiLMN/main/imgs/fog_result.png)
## 🌙 Visual Results on RTTS Dataset (Low-light)
![](https://raw.githubusercontent.com/Newj596/FiLMN/main/imgs/dark_result.png)

## 🌐 Real-World Applications
![](https://raw.githubusercontent.com/Newj596/FiLMN/main/imgs/usv_res.png)
Clear Daytime
Fig. (a) illustrates our method's effectiveness in tracking surface vessels under ideal visibility. The domain-mixed expert system maintains stable detection confidences across varying boat orientations and wave patterns. The detection efficacy in this scenario originates from the pronounced chromatic and textural saliency of objects, which establishes robust feature discriminability for reliable classification. Key advantages include reduced sensitivity to sun glares and wake interference compared to conventional approaches, as evidenced by the consistent bounding box accuracy throughout the temporal sequence.

Foggy Conditions
As shown in Fig. (b), our architecture achieves reliable detection in reduced visibility conditions. In such scenarios, objects are subject to intense backlighting and haze occlusion, resulting in silhouette-like appearances with severely degraded color and texture information, thereby leading to the breakdown of conventional detection algorithms.  Our proposed FiLMN addresses the problems by preserving critical edge information while suppressing fog-induced artifacts, enabling accurate target localization even when vessel contours become partially obscured.  

Low-Light Night
Detection results in Fig. (c) demonstrate FiLMN's capability in nighttime operations. Under nighttime conditions, objects suffer from significant feature distribution shifts caused by background light interference, thereby rendering domain-specific training weights inadequate for such dynamically varying scenarios. In contrast, FiLMN adopts a domain-agnostic framework that dynamically selects attention mechanisms based on evolving feature patterns to recalibrate critical representations, demonstrating its enhanced robustness in these challenging environments. The framework successfully handles illumination variations and dark channel noise, maintaining detection consistency across different light levels. 

## ⚙️ Installation
Following the installation instructions as YOLO v5 [link](https://github.com/ultralytics/yolov5) 
```
cd FiLMN\
pip install -r requirements.txt
```
## 📦 Datasets

| RTTS      | ExDark      |
|------------|------------|
| [link](https://pan.baidu.com/s/1IYkX2B31rSkji55-12TZVg?pwd=yba2) | [link](https://pan.baidu.com/s/1alIMr8ReBvQStX8Mk3VCsg?pwd=7wit) |

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
