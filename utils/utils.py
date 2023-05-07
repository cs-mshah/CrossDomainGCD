import numpy as np
import os
from pathlib import Path
import torch
import torchvision
from collections import defaultdict
import random
import shutil
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

class Losses:
    '''Create a class for all losses and iterate over them'''
    def __init__(self):
        self.losses_dict = {}

    def add_loss(self, name:str):
        self.losses_dict[name] = AverageMeter()

    def update(self, name:str, value, n=1):
        self.losses_dict[name].update(value, n)

    def reset(self):
        for loss in self.losses_dict.values():
            loss.reset()

    def get_averages(self):
        return {name: loss.avg for name, loss in self.losses_dict.items()}

    def __len__(self):
        return len(self.losses_dict)

    def __getitem__(self, name):
        return self.losses_dict[name]

    def __iter__(self):
        return iter(self.losses_dict)


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # import pdb; pdb.set_trace()
        try:
            correct_k = correct[:k].view(-1).float().sum(0)
        except:
            correct_k = correct[:k].reshape(-1).float().sum(0)
        try:
            res.append(correct_k.mul_(100.0 / batch_size))
        except:
            res = (torch.tensor(0.0), torch.tensor(0.0))
    return res


def save_checkpoint(state, is_best, save_path, tag='base'):
    filename=f'checkpoint_{tag}.pth.tar'
    filepath = os.path.join(save_path, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(save_path, f'model_best_{tag}.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def sim_matrix(a, b, args, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def interleave(x, size):
    s = list(x.shape)
    # import pdb; pdb.set_trace()
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def check_input_format(input):
    '''checks if a given Dataset path is in the standard Image Classification format'''
    p_input = Path(input)
    if not p_input.exists():
        err_msg = f'The provided input folder "{input}" does not exists.'
        if not p_input.is_absolute():
            err_msg += f' Your relative path cannot be found from the current working directory "{Path.cwd()}".'
        raise ValueError(err_msg)

    if not p_input.is_dir():
        raise ValueError(f'The provided input folder "{input}" is not a directory')

    dirs = [f for f in Path(input).iterdir() if f.is_dir()]
    if len(dirs) == 0:
        raise ValueError(
            f'The input data is not in a right format. Within your folder "{input}" there are no directories.'
        )


def describe_image_dataset(dataset, ignore_errors=False):
    """
    This function takes the path to an image dataset or an ImageFolder object and returns a dictionary 
    containing information about the dataset. The dataset should be in the standard 
    image classification format.
    
    Args:
    - dataset (str) or (torchvision.datasets.ImageFolder): The path to the dataset folder 
    or ImageFolder object.
    
    Returns:
    - dataset_info (dict): A dictionary containing information about the dataset.
      - classes (list): A list of the classes in the dataset.
      - class_to_idx (dict): A dictionary mapping class names to class indices.
      - class_counts (dict): A dictionary mapping class names to the number of images in that class.
      - num_images (int): The total number of images in the dataset.
      - min_samples (int): The minimum number of samples in a class.
      - max_samples (int): The maximum number of samples in a class.
      - avg_samples (float): The average number of samples in a class.
    """
    
    if isinstance(dataset, str):
        try:
            check_input_format(dataset)
            dataset = torchvision.datasets.ImageFolder(dataset)
        except Exception as e:
            if not ignore_errors:
                raise e
            empty_dict = {}
            return empty_dict
    if not isinstance(dataset, torchvision.datasets.ImageFolder):
        err_msg = f'passed object is not a torchvision.datasets.Dataset'
        raise ValueError(err_msg)
    
    classes = dataset.classes
    class_to_idx = dataset.class_to_idx
    num_images = len(dataset)

    class_counts = defaultdict(int)
    for image_path, class_idx in dataset.imgs:
        class_name = classes[class_idx]
        class_counts[class_name] += 1

    class_sizes = list(class_counts.values())
    min_samples = min(class_sizes)
    max_samples = max(class_sizes)
    avg_samples = sum(class_sizes) / len(class_sizes)
    
    dataset_info = {
        'classes': classes,
        'class_to_idx': class_to_idx,
        'class_counts': dict(class_counts),
        'num_images': num_images,
        'min_samples': min_samples,
        'max_samples': max_samples,
        'avg_samples': avg_samples
    }

    return dataset_info


def describe_splits(args, ignore_errors=False):
    if args.dataset == 'pacs':
        s_train = os.path.join(args.data_root, 'train', args.train_domain)
        t_val = os.path.join(args.data_root, 'val', args.test_domain)
    elif args.dataset == 'officehome':
        s_train = describe_image_dataset(os.path.join(args.data_root, args.train_domain, 'train'), ignore_errors=ignore_errors)
        s_val = describe_image_dataset(os.path.join(args.data_root, args.train_domain, 'val'), ignore_errors=ignore_errors)
        t_train = describe_image_dataset(os.path.join(args.data_root, args.test_domain, 'train'), ignore_errors=ignore_errors)
        t_val = describe_image_dataset(os.path.join(args.data_root, args.test_domain, 'val'), ignore_errors=ignore_errors)
        print(f"s_train: {s_train.get('class_counts')}")
        print(f"s_val: {s_val.get('class_counts')}")
        print(f"t_train: {t_train.get('class_counts')}")
        print(f"t_val: {t_val.get('class_counts')}")


def get_optimizer_and_scheduler(args, model_parameters):
    # Create the optimizer
    optimizer = None
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model_parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model_parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError("Invalid optimizer type. Must be 'sgd' or 'adam'.")
    
    # Create the learning rate scheduler
    scheduler = None
    if args.scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max, eta_min=args.eta_min)
    elif args.scheduler == 'reduce_on_plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode=args.mode, factor=args.factor, patience=args.patience)
    elif args.scheduler == 'lambda':
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    elif args.scheduler == 'constant':
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda x: 1)
    else:
        raise ValueError("Invalid scheduler type. Must be 'step', 'cosine', 'lambda', 'constant' or 'reduce_on_plateau'.")
    
    return optimizer, scheduler