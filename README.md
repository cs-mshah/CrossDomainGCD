# OpenLDN: Learning to Discover Novel Classes for Open-World Semi-Supervised Learning

Implementation of [OpenLDN: Learning to Discover Novel Classes for Open-World Semi-Supervised Learning](https://arxiv.org/abs/2207.02261).

Semi-supervised learning (SSL) is one of the dominant approaches to address the annotation bottleneck of supervised learning. Recent SSL methods can effectively leverage a large repository of unlabeled data to improve performance while relying on a small set of labeled data. One common assumption in most SSL methods is that the labeled and unlabeled data are from the same underlying data distribution. However, this is hardly the case in many real-world scenarios, which limits their applicability. In this work, instead, we attempt to solve the recently proposed challenging open-world SSL problem that does not make such an assumption. In the open-world SSL problem, the objective is to recognize samples of known classes, and simultaneously detect and cluster samples belonging to novel classes present in unlabeled data. This work introduces OpenLDN that utilizes a pairwise similarity loss to discover novel classes. Using a bi-level optimization rule this pairwise similarity loss exploits the information available in the labeled set to implicitly cluster novel class samples, while simultaneously recognizing samples from known classes. After discovering novel classes, OpenLDN transforms the open-world SSL problem into a standard SSL problem to achieve additional performance gains using existing SSL methods. Our extensive experiments demonstrate that OpenLDN outperforms the current state-of-the-art methods on multiple popular classification benchmarks while providing a better accuracy/training time trade-off.


## Training
```shell
# For CIFAR10 50% Labels and 50% Novel Classes 
python3 train.py --dataset cifar10 --lbl-percent 50 --novel-percent 50 --arch resnet18

# For CIFAR100 50% Labels and 50% Novel Classes 
python3 train.py --dataset cifar100 --lbl-percent 50 --novel-percent 50 --arch resnet18

For training on the other datasets, please download the dataset and put under the "name_of_the_dataset" folder and put the train and validation/test images under "train" and "test" folder. After that, please set the value of data_root argument as "name_of_the_dataset".

# For Tiny ImageNet 50% Labels and 50% Novel Classes
python3 train.py --dataset tinyimagenet --lbl-percent 50 --novel-percent 50 --arch resnet18

# For ImageNet-100 50% Labels and 50% Novel Classes
python3 train.py --dataset imagenet100 --lbl-percent 50 --novel-percent 50 --arch resnet50

# For Oxford-IIIT Pet 50% Labels and 50% Novel Classes
python3 train.py --dataset oxfordpets --lbl-percent 50 --novel-percent 50 --arch resnet18

# For FGVC-Aircraft 50% Labels and 50% Novel Classes
python3 train.py --dataset aircraft --lbl-percent 50 --novel-percent 50 --arch resnet18

# For Stanford-Cars 50% Labels and 50% Novel Classes
python3 train.py --dataset stanfordcars --lbl-percent 50 --novel-percent 50 --arch resnet18

# For Herbarium19 50% Labels and 50% Novel Classes
python3 train.py --dataset herbarium --lbl-percent 50 --novel-percent 50 --arch resnet18

# For SVHN 10% Labels and 50% Novel Classes
python3 train.py --dataset svhn --lbl-percent 10 --novel-percent 50 --arch resnet18
```

## Our Documentation

Environment
```shell
conda activate openldn
```

Add new datasets to `../datasets` folder and only symlink as required. Example:
```shell
ln -sf ~/Mainak/datasets/pacs_dataset ~/Mainak/CrossDomainNCD/OpenLDN/data/
```

Changing arguments:  
- `base/train-base-new.py` (new contains our implementation, train-base contains original OpenLDN)  
- `closed_world_ssl/train_mixmatch.py`, `closed_world_ssl/train_uda.py`

For reproducibility, set `args.seed=0`. Verified that this creates the same split indexs.  
To resume training use the same random splits as in `random_splits/`.

For training on **PACS** dataset use:
```shell
python train.py --dataset pacs --lbl-percent 50 --novel-percent 50 --arch resnet18
```

Experimental **PACS** `Dataset` class is in `base/utils/multi_domain.py`. We can add different `Dataset` classes and run this file to create the data in the generic format of `data/name_of_dataset/train` and `data/name_of_dataset/test` folders.

For training on **officehome** dataset use:
```shell
python train.py --dataset officehome --lbl-percent 60 --novel-percent 30 --arch resnet18
```
The train and test sets are full sets of a particular domain and not split further. Change the train and test domains in the `elif args.dataset == 'officehome'` condition in `datasets.py`, `datasets_mixmatch.py`, `datasets_uda.py` files.

### wandb upload

To upload tensorboard log dir to wandb use:
```shell
wandb sync -p IITB-MBZUAI -e cv-exp --include-globs '*.txt' --exclude-globs '*.tar,*.pt,*.pth' output_dir/
```  
Default wandb logging has been added.  

### t-SNE plots
Edit arguments in `base/visualization/tsne.py` and then run:  
```shell
cd base
python visualization/tsne.py
```

## Citation
```
@inproceedings{rizve2022openldn,
  title={Openldn: Learning to discover novel classes for open-world semi-supervised learning},
  author={Rizve, Mamshad Nayeem and Kardan, Navid and Khan, Salman and Shahbaz Khan, Fahad and Shah, Mubarak},
  booktitle={European Conference on Computer Vision},
  pages={382--401},
  year={2022},
  organization={Springer}
}
```
