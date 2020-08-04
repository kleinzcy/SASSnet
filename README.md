# SASSnet
Code for paper: Shape-aware Semi-supervised 3D Semantic Segmentation for Medical Images(MICCAI 2020)

Our code is origin from [UA-MT](https://github.com/yulequan/UA-MT)

You can find paper in [Arxiv](https://arxiv.org/abs/2007.10732).

# Usage

1. Clone the repo:
```
git clone https://github.com/kleinzcy/SASSnet.git 
cd SASSnet
```
2. Put the data in `data/2018LA_Seg_Training Set`.

3. Train the model
```
cd code
# for 16 label
python train_gan_sdfloss.py --gpu 0 --label 16 --consistency 0.01 --exp model_name
# for 8 label
python train_gan_sdfloss.py --gpu 0 --label 8 --consistency 0.015 --exp model_name
```

Params are the best setting in our experiment.

4. Test the model
```
python test_LA.py --model model_name --gpu 0 --iter 6000
```
Our best model are saved in model dir.

# Citation

If you find our work is useful for you, please cite us.
