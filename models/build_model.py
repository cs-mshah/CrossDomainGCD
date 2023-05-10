import os
from typing import Tuple, Optional, List, Dict
from .resnetdsbn import resnet18dsbn
from collections import OrderedDict
from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
import tllib.vision.models as models
from tllib.modules.classifier import Classifier as ClassifierBase
from tllib.modules.domain_discriminator import DomainDiscriminator


class ImageClassifier(ClassifierBase):
    """lr of backbone is 0.1 * lr of bottleneck and head. set finetune=false for same lr"""
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck: Optional[nn.Module] = None,
                 bottleneck_dim: Optional[int] = -1, head: Optional[nn.Module] = None, **kwargs):
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, head, **kwargs)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """return features, logits"""
        f = self.pool_layer(self.backbone(x))
        f = self.bottleneck(f)
        predictions = self.head(f)
        return f, predictions

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


class SupConResNet(nn.Module):
    """image_classifier + projection head"""
    def __init__(self, image_classifier: nn.Module, head='mlp', feat_dim=128, num_classes=10):
        super(SupConResNet, self).__init__()
        self.encoder = image_classifier
        dim_in = self.encoder.features_dim
        self.fc = nn.Linear(dim_in, num_classes) # for CE loss
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        """returns feat_normal, logits for CE, feats_backbone"""
        feat, preds = self.encoder(x) # as encoder.head=nn.Identity, feat=preds
        out_lin = self.fc(preds)
        preds = F.normalize(self.head(preds), dim=1)
        return preds, out_lin, feat
    
    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        params = []
        params += self.encoder.get_parameters(base_lr)
        params += [
            {"params": self.fc.parameters(), "lr": 1.0 * base_lr},
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
    models = {}
    backbone = get_backbone(args, verbose=verbose)

    if args.method == 'dann':
        bottleneck = nn.Sequential(
                nn.Linear(backbone.out_features, args.bottleneck_dim),
                nn.BatchNorm1d(args.bottleneck_dim),
                nn.ReLU()
            )
        models['classifier'] = ImageClassifier(backbone, args.no_class, bottleneck, args.bottleneck_dim)
        models['domain_discri'] = DomainDiscriminator(in_feature=models['classifier'].features_dim, hidden_size=1024)

    elif args.method == 'dsbn':
        models['classifier'] = resnet18dsbn(num_classes=args.no_class)
    elif args.method == 'contrastive':
        head = nn.Identity()
        image_classifier = ImageClassifier(backbone, args.no_class, head=head, finetune=False)
        models['classifier'] = SupConResNet(image_classifier, num_classes=args.no_class)
    elif args.method == 'dann_contrastive':
        head = nn.Identity()
        bottleneck = nn.Sequential(
                nn.Linear(backbone.out_features, args.bottleneck_dim),
                nn.BatchNorm1d(args.bottleneck_dim),
                nn.ReLU()
            )
        image_classifier = ImageClassifier(backbone, args.no_class, bottleneck, args.bottleneck_dim, head, finetune=False)
        models['classifier'] = SupConResNet(image_classifier, num_classes=args.no_class)
        models['domain_discri'] = DomainDiscriminator(in_feature=image_classifier.features_dim, hidden_size=1024)
    else:
        models['classifier'] = ImageClassifier(backbone, args.no_class)

    # use dataparallel if there's multiple gpus
    if args.n_gpu > 1:
        for name, model in models.items():
            models[name] = torch.nn.DataParallel(model)
        model = nn.DataParallel(model)

    return models
