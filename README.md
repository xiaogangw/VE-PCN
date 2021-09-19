## [Voxel-based Network for Shape Completion by Leveraging Edge Generation](https://arxiv.org/pdf/2108.09936.pdf)

This is the PyTorch implementation for the paper "Voxel-based Network for Shape Completion by Leveraging Edge Generation (ICCV 2021, oral)"

## Getting Started
python version: python-3.6;  cuda version: cuda-10;  PyTorch version: 1.5

## Compile Customized Operators
Build operators under ops by using python setup.py install.

## Datasets
[Our dataset](https://drive.google.com/file/d/1w0max7KksZQtlYsZN9WQpWIACZrVCR5t/view?usp=sharing)   [PCN's dataset](https://github.com/wentaoyuan/pcn)  [TopNet's dataset](https://github.com/lynetcha/completion3d)
    
## Train the model
To train the models on pcn dataset: python train_edge.py  
    --train_pcn;  
    --loss_type: pcn;  
    --train_path: the training data;  
    --eval_path: the validation data;  
    --n_gt_points: 16384;  
    --n_out_points: 16384;  
    --density_weight:1e11;   
    --dense_cls_weight:1000;   
    --p_norm_weight:0;  
    --dist_regularize_weight:0;  
    --chamfer_weight:1e6;  
    --lr 0.0007.
    
To train the models on topnet dataset: python train_edge.py  
    --train_pcn;  
    --loss_type: topnet;  
    --train_path: the training data;  
    --eval_path: the validation data;  
    --n_gt_points: 2048;  
    --n_out_points: 2048;  
    --density_weight:1e10;   
    --dense_cls_weight:100;   
    --p_norm_weight:300;  
    --dist_regularize_weight:0.3;  
    --chamfer_weight:1e4;  
    --augment;  
    --lr 0.0007.
    
To train the models on our dataset: python train_edge.py   
    --train_seen;  
    --loss_type: topnet;  
    --h5_train: the training data;  
    --h5_val: the validation data;  
    --n_gt_points: 2048;  
    --n_out_points: 2048;  
    --density_weight:1e10;   
    --dense_cls_weight:100;   
    --p_norm_weight:300;  
    --dist_regularize_weight:0.3;  
    --chamfer_weight:1e4;  
    --lr 0.0007.
    

## Evaluate the models
The pre-trained models can be downloaded here: [Models](https://drive.google.com/file/d/1U8csGt588IV9setytFjqK6VIpGFM6GhJ/view?usp=sharing), unzip and put them in the root directory.  
To evaluate models: python test_edge.py  
    --loss_type: topnet or pcn;  
    --eval_path: the test data from different cases;   
    --checkpoint: the pre-trained models;   
    --num_gt_points: the resolution of ground truth point clouds.
    
## Citation
@inproceedings{wang2021voxel,  
&nbsp;&nbsp;&nbsp;&nbsp;      author    = {Wang, Xiaogang and , Marcelo H. Ang Jr. and Lee, Gim Hee},  
&nbsp;&nbsp;&nbsp;&nbsp;      title     = {Voxel-based Network for Shape Completion by Leveraging Edge Generation},  
&nbsp;&nbsp;&nbsp;&nbsp;      booktitle = {ICCV)},         
&nbsp;&nbsp;&nbsp;&nbsp;      year      = {2021},  
}

## Acknowledgements 
Our implementations use the code from the following repository:  
[Chamferdistance](https://github.com/krrish94/chamferdist/tree/488b2d6b9f62014a04109dfe8ee3a68189c44f4d)        
[PointNet++](https://github.com/erikwijmans/Pointnet2_PyTorch)  
[convolutional_point_cloud_decoder](https://gitlab.vci.rwth-aachen.de:9000/lim/convolutional_point_cloud_decoder)
