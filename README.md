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
ln -sf ~/<path_to_dataset>/pacs_dataset ~/<path_to_project>/CrossDomainGCD/data/
```
The above is automatically handled with `download_dataset()` called in `argparser.py`  

## Training

Change arguments in `argparser.py` for training. Training command:

```shell
python train.py
```

Default wandb logging has been added.  

### t-SNE plots

Automatically handled during training. t-SNE embeddings are obtained for the latent space (before fc layer). 

To independently get tsne plots run:
```shell
python visualization/tsne.py
```

## Remote runs
A good way to run experiments on a remote DGX server is to use `rsync` to transfer the repository.
Always do a **dry run** first when using `--delete`. `cd ../` (i.e get outside main repository directory) then:
```shell
rsync -avz --delete --dry-run --exclude-from='OpenLDN/rsync_exclude.txt' -e "ssh" OpenLDN dgxadmin@10.107.111.21:/home/dgxadmin/Manan/CrossDomainGCD
```
If everything seems fine,
```shell
rsync -avz --delete --exclude-from='OpenLDN/rsync_exclude.txt' -e "ssh" OpenLDN dgxadmin@10.107.111.21:/home/dgxadmin/Manan/CrossDomainGCD
```

## Cleanup
- To disable wandb logging, prefix the run command with `WANDB_MODE=disabled`.  
- To clean local experiments in `outputs/`, which are not there on wandb, run `wandb_utils.py`.
- To clean local wandb/ logs which are synced online, run `wandb sync --clean`.

## References

Self Supervised Pretrained weights: [swav weights](https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar): [facebookresearch/swav](https://github.com/facebookresearch/swav)  

Domain adaptation: [thuml/Transfer-Learning-Library](https://github.com/thuml/Transfer-Learning-Library/tree/master/examples/domain_adaptation/image_classification)  

Some implementation: [OpenLDN](https://github.com/nayeemrizve/OpenLDN)  

Self Supervised pretraining code (SimCLR): [SupContrast](https://github.com/HobbitLong/SupContrast)

