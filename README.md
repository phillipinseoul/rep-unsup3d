# Reproducing Unsup3D for CS492(A): Machine Learning for 3D Data (22' Spring)

By Yuseung Lee & Inhee Lee

## Environment Setting
We recommend you to use this model 

## How to run the code
You need to modify configs file first before train or test the model. We recommend you to use bfm_template.yml in configs/ablation/ as template.

We provide dataloader for BFM datasets and CelebA. 
Here is the link for both datasets. (This link will be closed after evaluation of CS492(A))

dataset gdrive : https://drive.google.com/drive/folders/1AzvmOGHv-4xLKotcAaP0pkeALRoMkGNp?usp=sharing
pretrained gdrive : https://drive.google.com/drive/folders/17101Jj5PcCqmb1ywRzmcriONTGOHZCDj?usp=sharing


codes to be runned
```bash
$ python run.py --configs configs/bfm_train_v0.yaml
$ tensorboard --logdir /logs/ --port 6001
```

* exmaples on BFM and CelebA
![image](https://user-images.githubusercontent.com/65122489/172181746-95db1bf6-a59f-41de-ace2-4067cad181a6.png)
* Pipdline of Unsup3D
![image](https://user-images.githubusercontent.com/65122489/172181610-a4b4ea31-a425-4751-b01f-ba0104d558cb.png)


## Reference
- Official Implementation
https://github.com/elliottwu/unsup3d
- Modified Neural Renderer
https://github.com/adambielski/neural_renderer
- Unsup3D[Wu et al.]
https://arxiv.org/abs/1911.11130


