import os
from typing import Tuple, Optional, List, Dict
from .resnetdsbn import resnet18dsbn
from .resnet_big import SupConResNet
from collections import OrderedDict
from torchsummary import summary
import torch
import torch.nn as nn
import tllib.vision.models as models
from tllib.modules.classifier import Classifier as ClassifierBase


class ImageClassifier(ClassifierBase):
    """lr of backbone is 0.1 * lr of bottleneck and head"""
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = -1, **kwargs):
        bottleneck = None
        if bottleneck_dim != -1:
            bottleneck = nn.Sequential(
                nn.Linear(backbone.out_features, bottleneck_dim),
                nn.BatchNorm1d(bottleneck_dim),
                nn.ReLU()
            )
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """return features, logits"""
        f = self.pool_layer(self.backbone(x))
        f = self.bottleneck(f)
        predictions = self.head(f)
        return f, predictions
        # if self.training:
        #     return predictions, f
        # else:
        #     return predictions

    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
            {"params": self.head.parameters(), "lr": 1.0 * base_lr},
        ]

        return params


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


def get_backbone(args, verbose=False):
    """returns a backbone model"""
    backbone = models.__dict__[args.arch](pretrained=False)
    if 'swav' in args.pretrained:
        ckpt_path = os.path.join('pretrained', args.pretrained)
        state_dict = torch.load(f=ckpt_path, map_location=torch.device('cpu'))
        new_state_dict = modify_state_dict(state_dict, 'remove_prefix', 'module.')
        backbone.load_state_dict(new_state_dict, strict=False)
        if verbose:
            model_keys_diff(backbone, new_state_dict)
            # print(summary(backbone))
    return backbone


def build_model(args, verbose=False):
    """returns models depending on the method"""
    # if args.arch == 'resnet18':
    # if args.dataset in ['cifar10', 'cifar100', 'svhn']:
    #     from . import resnet_cifar as models
    # elif args.dataset == 'tinyimagenet':
    #     from . import resnet_tinyimagenet as models
    # else:
    #     from . import resnet as models
    models = {}
    backbone = get_backbone(args, verbose=verbose)

    if args.dann:
        from tllib.modules.domain_discriminator import DomainDiscriminator
        from tllib.alignment.dann import DomainAdversarialLoss
        # model = models.dann_resnet18(no_class=args.no_class)
        models['classifier'] = ImageClassifier(backbone, args.no_class, bottleneck_dim=args.bottleneck_dim)
        models['domain_discri'] = DomainDiscriminator(in_feature=models['classifier'].features_dim, hidden_size=1024)
        # models['domain_adv_loss'] = DomainAdversarialLoss(models['domain_discri'])
        

    elif args.dsbn:
        models['classifier'] = resnet18dsbn(num_classes=args.no_class)
    elif args.contrastive:
        # model = SupConResNet(name=args.arch, num_classes=args.no_class)
        models['classifier'] = SupConResNet(backbone, num_classes=args.no_class)
    else:
        # model = models.resnet18(no_class=args.no_class)
        models['classifier'] = ImageClassifier(backbone, args.no_class)

    # if args.pretrained:
    #     """if resuming training set this to false"""
    #     model_fp = os.path.join('pretrained', args.pretrained)
    #     device = 'cpu'
    #     map_location = torch.device(device)
    #     state_dict = torch.load(model_fp, map_location=map_location)

    #     new_state_dict = modify_state_dict(state_dict, 'remove_prefix', 'encoder.')
        
    #     if args.contrastive:
    #         new_state_dict = modify_state_dict(state_dict, 'remove_prefix', 'module.')
    #         if verbose:
    #             model_keys_diff(model.encoder, new_state_dict)
    #         model.encoder.load_state_dict(new_state_dict, strict=False)
    #     else:
    #         model.load_state_dict(new_state_dict, strict=False)
    #         if verbose:
    #             model_keys_diff(model, new_state_dict)

    # use dataparallel if there's multiple gpus
    if args.n_gpu > 1:
        for name, model in models.items():
            models[name] = torch.nn.DataParallel(model)
        model = nn.DataParallel(model)

    return models
