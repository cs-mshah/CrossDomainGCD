# Cross Domain GCD

## Problem setup
- 2 domains: *Source Domain* and *Target Domain* with a common label set.
- Samples of only few classes from the source are known (labeled). We assume no knownledge of unknown class samples from the source domain.
- Target Domain contains samples from both known and unknown classes and are unlabeled. They are available for training.
- **Task**: perform classification for the known class samples in the target domain and clustering for the unkown class samples in the target domain.

## Environment

```shell
conda env create -f environment.yml
conda activate openldn
```

## Datasets

Add new datasets to `../datasets` folder and only symlink as required. Example:
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

Handled during training. To independently get tsne plots run:
```shell
python visualization/tsne.py
```