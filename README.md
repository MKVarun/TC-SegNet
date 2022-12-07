# TC-SegNet
A robust deep learning framework for segmentation of 2D echocardiography.

## Description

We used the Cardiac Acquisitions for Multi-structure Ultrasound Segmentation (CAMUS) dataset available [here](https://www.creatis.insa-lyon.fr/Challenge/camus/databases.html).

**Models used**
- UNet(UNet)
- [Recurrent U-Net](https://doi.org/10.1109/ICCV.2019.00223) (R2UNet)
- [Attention U-Net](https://arxiv.org/abs/1804.03999) (Attn_Unet)
- [Residual Unet](https://doi.org/10.1109/LGRS.2018.2802944) (ResUnet)
- [ASPP UNet](https://www.sciencedirect.com/science/article/abs/pii/S0925231220303374?via%3Dihub)
- [HMEDN](https://doi.org/10.1109/TIP.2019.2919937)
- **TC-SegNet** (model = ResUnetPlusPlus_Path)

## Results
**Pre Processing**<br>
<img src="imgs/preprocess.png" alt="drawing" width="400"/><br>


**Segmentation Output**<br>
<img src="imgs/final_overlay.png" alt="drawing" width="600"/><br>
