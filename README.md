# EHN

<!-- <img src= "https://github.com/Chengzhi-Cao/SC-Net/blob/main/pic/network.png" width="100%"> -->
<img src= "pic/results.jpg" width="100%">

This repository provides the official PyTorch implementation of the following paper:

**Event-driven Heterogeneous Network for Video Deraining**


<!-- IEEE Transactions on Neural Networks and Learning Systems -->

<!-- [Paper Link](https://ieeexplore.ieee.org/abSCNetct/document/10314003) -->


## Dependencies

- Python
- scikit-image
- opencv-python
- pytorch == 1.8.0+cu111
- torchvision == 0.9.0+cu111
- numpy == 1.19.2
- h5py == 3.3.0
- Win10 or Ubuntu18.04


## Dataset


- Download Derain dataset：
  - Synlight: 
  - Synheavy:
  - NTU: 




- Unzip files ```dataset``` folder.


After preparing data set, the data folder should be like the format below:

```
GOPRO
├─ train
│ ├─ blur    % 2103 image pairs
│ │ ├─ xxxx.png
│ │ ├─ ......
│ │
│ ├─ sharp
│ │ ├─ xxxx.png
│ │ ├─ ......
│
├─ test    % 1111 image pairs
│ ├─ ...... (same as train)

```

- Utilize [V2E](https://github.com/SensorsINI/v2e) to generate the corresponding event sequence.

- Preprocess events by running the command below:

  ``` python data/dataset_event.py```



## Train

To train SCNet , run the command below:

``` python main.py --model_name "SCNet" --mode "train_event_Temporal" --data_dir "dataset/GOPRO" ```

Model weights will be saved in ``` results/model_name/weights``` folder.


## Test

To test SCNet , run the command below:

``` python main.py --model_name "SCNet" --mode "test" --data_dir "dataset/GOPRO" --test_model "xxx.pkl" ```

Output images will be saved in ``` results/model_name/result_image``` folder.



## Performance

<img src= "pic/result1.jpg" width="100%">

<img src= "pic/result2.jpg" width="100%">



## Citation

<!-- ```
@ARTICLE{10314003,
  author={Cao, Chengzhi and Fu, Xueyang and Zhu, Yurui and Sun, Zhijing and Zha, Zheng-Jun},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Event-Driven Video Restoration With Spiking-Convolutional Architecture}, 
  year={2023},
  volume={},
  number={},
  pages={1-15},
  keywords={Image restoration;Feature extraction;Correlation;Computer architecture;Cameras;Task analysis;Superresolution;Convolutional neural networks (CNNs);event camera;spiking neural networks (SNNs);video restoration},
  doi={10.1109/TNNLS.2023.3329741}}
``` -->

## Notes and references
The  code is based on the paper:

'Rethinking Coarse-to-Fine Approach in Single Image Deblurring'(https://arxiv.org/abs/2108.05054)

'Event-driven Video Deblurring via Spatio-Temporal Relation-Aware Network'(https://www.ijcai.org/proceedings/2022/112)


## Contact
Should you have any question, please contact chengzhicao@mail.ustc.edu.cn.

