import os
import torch
from .resnetdsbn import resnet18dsbn
from .resnet_big import SupConResNet
from collections import OrderedDict

def modify_state_dict(state_dict, func, s='encoder.'):
    """returns new_stat_dict according to func and string s={'encoder.', 'module.'}
    func={remove_prefix, add_prefix}"""
    def remove_prefix(key):
        return key[len(s):] if key.startswith(s) else key
    
    def add_prefix(key):
        return s + key if (not key.startswith(s)) else key
    
    modification_functions = {
        'remove_prefix': remove_prefix,
        'add_prefix': add_prefix
    }
    
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = modification_functions[func](key)
        new_state_dict[new_key] = value
    
    return new_state_dict


def model_keys_diff(model, pretrained_weights):
    """prints the difference in keys between model and pretrained weights"""
    keys_model = set(model.state_dict().keys())
    keys_pretrained = set(pretrained_weights.keys())
    diff_keys1 = keys_model - keys_pretrained
    diff_keys2 = keys_pretrained - keys_model
    print(f"Keys in model but not in pretrained model: {diff_keys1}")
    print(f"Keys in pretrained model but not in model: {diff_keys2}")


def build_model(args, verbose=False):
    # if args.arch == 'resnet18':
    if args.dataset in ['cifar10', 'cifar100', 'svhn']:
        from . import resnet_cifar as models    
    elif args.dataset == 'tinyimagenet':
        from . import resnet_tinyimagenet as models
    else:
        from . import resnet as models

    if args.dann:
        model = models.dann_resnet18(no_class=args.no_class)
    elif args.dsbn:
        model = resnet18dsbn(num_classes=args.no_class)
    elif args.contrastive:
        model = SupConResNet(name=args.arch, num_classes=args.no_class)
    else:
        model = models.resnet18(no_class=args.no_class)

    # use dataparallel if there's multiple gpus
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    if args.pretrained:
        """if resuming training set this to false"""
        model_fp = os.path.join('pretrained', args.pretrained)
        device = 'cpu'
        map_location = torch.device(device)
        state_dict = torch.load(model_fp, map_location=map_location)

        new_state_dict = modify_state_dict(state_dict, 'remove_prefix', 'encoder.')
        
        if args.contrastive:
            new_state_dict = modify_state_dict(state_dict, 'remove_prefix', 'module.')
            if verbose:
                model_keys_diff(model.encoder, new_state_dict)
            model.encoder.load_state_dict(new_state_dict, strict=False)
        else:
            model.load_state_dict(new_state_dict, strict=False)
            if verbose:
                model_keys_diff(model, new_state_dict)

    return model
