import numpy as np
from PIL import Image, ImageFilter, ImageOps
import random
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from skimage import io
import pickle
import os
from .randaugment import RandAugmentMC
from .multi_domain import create_dataset
import math
from utils.utils import set_seed
import tllib.vision.datasets


# normalization parameters (mean, std)
# use imagenet if using an imagenet pretrained model
normalize_dict = {
    'cifar10': [(0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)],
    'cifar100': [(0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)],
    'normal': [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)],
    'tinyimagenet': [(0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)],
    'imagenet': [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
    'pacs': [(0.7659, 0.7463, 0.7173), (0.3089, 0.3181, 0.3470)]
}


def get_dataset(args):
    return get_dataset224(args)


def get_transforms(args, split: str):
    """returns transform for a given train/test split"""
    normalize = transforms.Normalize(mean=normalize_dict['imagenet'][0], std=normalize_dict['imagenet'][1])
    if args.dataset in normalize_dict.keys():
        normalize = transforms.Normalize(mean=normalize_dict[args.dataset][0], std=normalize_dict[args.dataset][1])
    
    if split == 'train':
        T = transforms.Compose([
            transforms.RandomResizedCrop(size=args.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
        
        if args.contrastive:
            """generate 2 transforms for ssl"""
            class TwoCropTransform:
                """Create two crops of the same image"""
                def __init__(self, transform):
                    self.transform = transform

                def __call__(self, x):
                    return [self.transform(x), self.transform(x)]

            T = transforms.Compose([
                transforms.RandomResizedCrop(size=args.img_size, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize,
            ])
            T = TwoCropTransform(T)
    
    else:
        T = transforms.Compose([
            transforms.Resize(args.img_size + 32), # 256 = 224 + 32
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            normalize])
    return T


def get_dataset224(args):
    # augmentations
    transform_labeled = get_transforms(args, 'train')
    transform_cross = transform_labeled
    transform_val = get_transforms(args, 'test')
    
    if args.dataset == 'pacs':
        # train_root = os.path.join(args.data_root, 'train', args.train_domain) # old
        # test_root = os.path.join(args.data_root, 'val', args.test_domain)
        train_root = os.path.join(args.data_root, args.train_domain)
        test_root = os.path.join(args.data_root, args.test_domain)
    elif args.dataset == 'office31':
        train_root = os.path.join(args.data_root, args.train_domain, 'images')
        test_root = os.path.join(args.data_root, args.test_domain, 'images')
    elif args.dataset == 'officehome':
        # train_root = os.path.join(args.data_root, args.train_domain, 'train') # old
        # test_root = os.path.join(args.data_root, args.test_domain, 'val')
        train_root = os.path.join(args.data_root, args.train_domain)
        test_root = os.path.join(args.data_root, args.test_domain)
    elif args.dataset == 'visda17':
        train_root = os.path.join(args.data_root, 'train')
        test_root = os.path.join(args.data_root, 'validation')
    else:
        train_root = os.path.join(args.data_root, 'train')
        test_root = os.path.join(args.data_root, 'test')
    
    
    base_dataset = datasets.ImageFolder(train_root)
    base_dataset_targets = np.array(base_dataset.imgs)
    base_dataset_targets = base_dataset_targets[:,1]
    base_dataset_targets= list(map(int, base_dataset_targets.tolist()))
    set_seed(args) # VERY IMPORTANT to keep same sets of classes when calling get_dataset224 multiple times
    train_labeled_idxs, train_unlabeled_idxs, train_val_idxs = x_u_split_known_novel(base_dataset_targets, args.lbl_percent, args.no_class, list(range(0,args.no_known)), list(range(args.no_known, args.no_class)))

    # balance the labeled and unlabeled data
    if len(train_unlabeled_idxs) > len(train_labeled_idxs):
        exapand_labeled = len(train_unlabeled_idxs) // len(train_labeled_idxs)
        train_labeled_idxs = np.hstack([train_labeled_idxs for _ in range(exapand_labeled)])

        if len(train_labeled_idxs) < len(train_unlabeled_idxs):
            diff = len(train_unlabeled_idxs) - len(train_labeled_idxs)
            train_labeled_idxs = np.hstack((train_labeled_idxs, np.random.choice(train_labeled_idxs, diff)))
        else:
            assert len(train_labeled_idxs) == len(train_unlabeled_idxs)

    # generate datasets
    train_target_tranform = None
    val_target_tranform = None
    if args.tsne:
        transform_labeled = get_transforms(args, 'test')
        # label transforms for tsne plotting
        train_target_tranform = transforms.Lambda(lambda y: 0)
        val_target_tranform = transforms.Lambda(lambda y: 1)

    train_labeled_dataset = GenericSSL(train_root, train_labeled_idxs, transform=transform_labeled, target_transform=train_target_tranform, args=args)
    crossdom_dataset = GenericTEST(test_root, no_class=args.no_class, transform=transform_cross, target_transform=val_target_tranform)
    test_dataset_known = GenericTEST(test_root, no_class=args.no_class, transform=transform_val, labeled_set=list(range(0, args.no_known)), target_transform=val_target_tranform)
    test_dataset_novel = GenericTEST(test_root, no_class=args.no_class, transform=transform_val, labeled_set=list(range(args.no_known, args.no_class)), target_transform=val_target_tranform)
    test_dataset_all = GenericTEST(test_root, no_class=args.no_class, transform=transform_val, target_transform=val_target_tranform)
    # print(set(train_labeled_dataset.targets))
    # check if target id's are same
    assert set(train_labeled_dataset.targets) == set(test_dataset_known.targets), "known set targets unequal"
    # check if all known source in train labeled (as label % is 100)
    # print(set(GenericSSL(train_root, train_unlabeled_idxs, transform=transform_labeled, target_transform=train_target_tranform, args=args).targets))
    
    return train_labeled_dataset, crossdom_dataset, test_dataset_known, test_dataset_novel, test_dataset_all


def x_u_split_known_novel(labels, lbl_percent, no_classes, lbl_set, unlbl_set, val_percent=1):
    labels = np.array(labels)
    labeled_idx = []
    unlabeled_idx = []
    val_idx = []
    for i in range(no_classes):
        idx = np.where(labels == i)[0]
        n_lbl_sample = math.ceil(len(idx)*(lbl_percent/100))
        n_val_sample = max(int(len(idx)*(val_percent/100)), 1)
        np.random.shuffle(idx)
        if i in lbl_set:
            labeled_idx.extend(idx[:n_lbl_sample])
            unlabeled_idx.extend(idx[n_lbl_sample:-n_val_sample])
            val_idx.extend(idx[-n_val_sample:])
        elif i in unlbl_set:
            unlabeled_idx.extend(idx[:-n_val_sample])
            val_idx.extend(idx[-n_val_sample:])
    return labeled_idx, unlabeled_idx, val_idx


class TransformWS224(object):
    '''returns weak and strong augmentations'''
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224),
            ])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

class GenericSSL(datasets.ImageFolder):
    '''__getitem__ returns img (weak+strong in case of unlabelled transforms), target, indexs'''
    def __init__(self, root, indexs,
                 transform=None, target_transform=None, args=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.args = args
        self.imgs = np.array(self.imgs)
        self.targets = self.imgs[:, 1]
        self.targets= list(map(int, self.targets.tolist()))
        self.data = np.array(self.imgs[:, 0])

        self.targets = np.array(self.targets)
        if indexs is not None:
            indexs = np.array(indexs)
            self.data = self.data[indexs] if len(indexs)!=0 else None
            self.targets = np.array(self.targets)[indexs] if len(indexs)!=0 else None
            self.indexs = indexs
        else:
            self.indexs = np.arange(len(self.targets))

    def __len__(self):
        return len(self.data) if self.data is not None else 0

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = self.loader(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.args.tsne is True:
            return img, target
        
        return img, target, self.indexs[index]


class GenericTEST(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, labeled_set=None, no_class=200):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.imgs = np.array(self.imgs)
        self.targets = self.imgs[:, 1]
        self.targets= list(map(int, self.targets.tolist()))
        self.data = np.array(self.imgs[:, 0])

        self.targets = np.array(self.targets)
        indexs = []
        if labeled_set is not None:
            for i in range(no_class):
                idx = np.where(self.targets == i)[0]
                if i in labeled_set:
                    indexs.extend(idx)
            indexs = np.array(indexs)
            self.data = self.data[indexs] if len(indexs)!=0 else None
            self.targets = np.array(self.targets)[indexs] if len(indexs)!=0 else None

    def __len__(self):
        return len(self.data) if self.data is not None else 0

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = self.loader(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def subset_dataset_factory(dataset_name: str):
    dataset = tllib.vision.datasets.__dict__[dataset_name]
    class SubsetDataset(dataset):
        def __init__(self, root: str, task: str, num_classes: int, **kwargs):
            super().__init__(root, task, **kwargs)
            self.CLASSES = self.CLASSES[:num_classes]
            self.classes = self.CLASSES
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            self.samples = [(path, self.class_to_idx[self.CLASSES[target]]) for path, target in self.samples if target < num_classes]

    return SubsetDataset
