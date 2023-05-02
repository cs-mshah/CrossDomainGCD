# Cross Domain GCD

## Problem setup
- 2 domains: *Source Domain* and *Target Domain* with a common label set.
- Samples of only few classes from the source are known (labeled). We assume no knownledge of unknown class samples from the source domain.
- Target Domain contains samples from both known and unknown classes and are unlabeled. They are available for training.
- **Task**: perform classification for the known class samples in the target domain and clustering for the unkown class samples in the target domain.

## Environment
It is better to use a separate folder to store all datasets and symlink from there. Set your `DATASETS_ROOT` env variable.  

```shell
export DATASETS_ROOT=~/user/datasets/
```

```shell
conda env create -f environment.yml
conda activate openldn
pip install git+https://github.com/thuml/Transfer-Learning-Library.git
```

## Datasets

Add new datasets to `DATASETS_ROOT` folder and only symlink as required. Example:
```shell
ln -sf ~/Mainak/datasets/pacs_dataset ~/Mainak/CrossDomainNCD/OpenLDN/data/
```

Set `args.lbl_percent=100` and `args.novel_percent=30`

```python
def create_dataset(args):
    """ creates ImageDataset into the following structure:
    domain1/
        train/ ..................(known labeled classes)
            class1/
            class2/
            ...
        val/ ..................(empty)
            class1/
            class2/

    domain2--
        train/ ..................(empty)
            class1/
            class2/
            ...
        val/ ..................(known + unknown classes ;use as unlabeled)
            class1/
            class2/
            ...
             """
```

In our setup, we have few classes from the source domain (domain1/train). Domain1/val remains empty as all examples are used in training.  
For target domain, all data appears in the domain2/val/ split as it is used for training and testing both.

## Training

Change arguments in `argparser.py` for training. Training command:

```shell
python train-base-new.py
```

Default wandb logging has been added.  

### t-SNE plots

Handled during training. t-SNE embeddings for
- self supervised approach: _normalized_ embeddings from the head
- backbones without heads: embeddings from the last layer (before fc layer).  

To independently get tsne plots run:
```shell
python visualization/tsne.py
```

## Remote runs
A good way to run experiments on a remote DGX server is to use `rsync` to transfer the repository.
Always do a **dry run** first when using `--delete`. `cd ../` : i.e get outside main repository directory then:
```shell
rsync -avz --delete --dry-run --exclude-from='OpenLDN/rsync_exclude.txt' -e "ssh" OpenLDN dgxadmin@10.107.111.21:/home/dgxadmin/Manan/CrossDomainGCD
```
If everything seems fine,
```shell
rsync -avz --delete --exclude-from='OpenLDN/rsync_exclude.txt' -e "ssh" OpenLDN dgxadmin@10.107.111.21:/home/dgxadmin/Manan/CrossDomainGCD
```