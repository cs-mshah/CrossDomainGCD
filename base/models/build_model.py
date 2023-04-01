import os
import torch
from .resnetdsbn import resnet18dsbn

def build_model(args, verbose=False):
    if args.arch == 'resnet18':
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
        else:
            model = models.resnet18(no_class=args.no_class)

        simnet = models.SimNet(1024, 100, 1)

    # use dataparallel if there's multiple gpus
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        simnet = torch.nn.DataParallel(simnet)

    if args.pretrained:
        '''if resuming set this to false'''
        model_fp = os.path.join('/home/biplab/Mainak/CrossDomainNCD/OpenLDN/pretrained/', args.pretrained)
        device = 'cuda:0' # when using single GPU
        map_location = torch.device(device)
        saved_model = torch.load(model_fp, map_location=map_location)
        new_state_dict = {}
        for key, value in saved_model.items():
            if key.startswith('encoder.'):
                new_key = key[len('encoder.'):]
            else:
                new_key = key
            new_state_dict[new_key] = value

        if verbose:
            # print the differences in the keys
            keys_model = set(model.state_dict().keys())
            keys_pretrained = set(new_state_dict.keys())
            diff_keys1 = keys_model - keys_pretrained
            diff_keys2 = keys_pretrained - keys_model
            print(f"Keys in model but not in pretrained model: {diff_keys1}")
            print(f"Keys in pretrained model but not in model: {diff_keys2}")
        model.load_state_dict(new_state_dict, strict=False)

    return model, simnet
