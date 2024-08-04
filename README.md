# EfficientMFD

## The code is released at https://github.com/icey-zhang/E2E-MFD.

EfficientMFD: Towards More Efficient Multimodal Synchronous Fusion Detection

Multimodal image fusion and object detection play a vital role in autonomous driving. Current joint learning methods have made significant progress in the multi-modal fusion detection task combining the texture detail and objective semantic information. However, the tedious training steps have limited its applications to wider real-world industrial deployment. To address this limitation, we propose a novel end-to-end multi-modal fusion detection algorithm, named EfficientMFD, to simplify models that exhibit decent performance with only one training step. Synchronous joint optimization is utilized in an end-to-end manner between two components, thus not being affected by the local optimal solution of the individual task. Besides, a comprehensive optimization is established in the gradient matrix between the shared parameters for both tasks. It can converge to an optimal point with fusion detection weights. We extensively test it on several public datasets, demonstrating superior performance on not only visually appealing fusion but also favorable detection performance (e.g.,~6.6\% AP) over other state-of-the-art approaches. 

![object_visualize_APPENDIX](https://github.com/icey-zhang/EfficientMFD/assets/74139077/00068d4b-9bc2-4fd8-b82c-93f061b50cdd)

### 1. Prepare the dataset M3FD

```python
EfficientMFD
├── datasets
│   ├── VOC2007
│   │   ├── ImageSets
│   │   │   ├── trainval.txt
│   │   │   ├── test.txt
│   │   ├── Annotations
│   │   │   ├── 00000.xml
│   │   │   ├── 00001.xml
│   │   │   ├── ......
│   │   ├── JPEGImages
│   │   │   ├── 00000.mat
│   │   │   ├── 00001.mat
│   │   │   ├── ......
```

### 2. Begin to train
```python
python train_net.py
```


The code will come soon. If you have any questions, please contact mingxiangcao@stu.xidian.edu.cn.

