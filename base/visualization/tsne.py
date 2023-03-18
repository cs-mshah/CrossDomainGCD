import torch
import numpy as np
import argparse
import os
import time
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, '/home/biplab/Mainak/CrossDomainNCD/OpenLDN/base')

from models.build_model import build_model
from utils.utils import set_seed
from datasets.datasets import get_tsne_dataset, get_dataset


def get_dataloader(args, dataset):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


def evaluate(args, dataset, model):
    dataloader = get_dataloader(args, dataset)
    print(next(iter(dataloader)))
    features = []
    labels = []
    with torch.inference_mode():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.cuda()
            feat, _ = model(inputs)
            features.append(feat.cpu().numpy())
            labels.extend(targets.tolist())
    return features, labels


def plot(args, model):
    '''plot tsne given args and model'''
    args.figsize = (17,13)
    
    # _, test_dataset = get_tsne_dataset(args)
    lbl_dataset, _, _, test_dataset_known, _, _ = get_dataset(args)
    
    model = model.cuda()
    model.eval()
    
    features, labels = evaluate(args, lbl_dataset, model)
    # features, labels = evaluate(args, test_dataset, model)
    # features, labels = evaluate(args, test_dataset_known, model)
    features_target, labels_target = evaluate(args, test_dataset_known, model)
    features.extend(features_target)
    labels.extend(labels_target)
    
    # TSNE plotting code
    features = np.array(features, dtype=object)
    labels = np.array(labels, dtype=object)
    features = np.vstack(features)
    labels = np.vstack(labels)

    # start_time = time.time()

    tsne = TSNE(n_jobs=16)

    embeddings = tsne.fit_transform(features)
    vis_x = embeddings[:, 0]
    vis_y = embeddings[:, 1]
    
    if not args.dann:
        palette = sns.color_palette("Spectral", args.no_known)
    else:
        palette = sns.color_palette("Spectral", 2)

    sns.set(rc={'figure.figsize': args.figsize})

    fig, ax = plt.subplots()
    plot = sns.scatterplot(x=vis_x, y=vis_y, 
                           hue=labels[:,0], legend='full', palette=palette,
                           ax=ax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Scatter Plot')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    plt.close()
    return fig

def main():
    parser = argparse.ArgumentParser(description='Get Visualizations')
    parser.add_argument('--description', default='default_run', type=str, help='description of the experiment')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100', 'svhn', 'tinyimagenet', 'oxfordpets', 'aircraft', 'stanfordcars', 'imagenet100', 'herbarium', 'domainnet', 'pacs','officehome'], help='dataset name')
    parser.add_argument('--data-root', default=f'data', help='directory to store data')
    parser.add_argument('--fig-root', default=f'tsne_plots', help='directory to plots')
    parser.add_argument('--lbl-percent', type=int, default=50, help='percent of labeled data')
    parser.add_argument('--novel-percent', default=50, type=int, help='percentage of novel classes, default 50')
    parser.add_argument('--no-class', default=10, type=int, help='total classes')
    parser.add_argument('--cw-ssl', default='mixmatch', type=str, choices=['mixmatch', 'uda'], help='closed-world SSL method to use')
    parser.add_argument('--out', default=f'outputs', help='directory to output the result')
    parser.add_argument('--num-workers', type=int, default=4, help='number of workers')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--split-id', default='split_0', type=str, help='random data split number')
    parser.add_argument('--dann', default=False, type=bool, help='run dann network to discriminate domain')
    parser.add_argument('--arch', default='resnet18', type=str, help='model architecture')
    parser.add_argument('--train-domain', required=False, type=str, help='train domain in case of cross domain setting')
    parser.add_argument('--test-domain', required=False, type=str, help='test domain in case of cross domain setting')
    parser.add_argument('--ssl-indexes', default='', type=str, help='path to random data split')
    parser.add_argument('--split-root', default=f'random_splits', help='directory to store datasets')
    parser.add_argument('--select-best', default=True, type=bool, help='select best model (default: True)')
    
    args = parser.parse_args()
    # overwrite command line args here
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    args.tsne = True # to have domain target(label) transforms
    args.description = 'dann'
    args.lbl_percent = 60
    args.novel_percent = 30
    args.cw_ssl = 'mixmatch'
    args.dataset = 'officehome'
    args.seed = 0
    args.batch_size = 32
    args.figsize = (17,13)
    args.train_domain = 'Product'
    args.test_domain = 'RealWorld'
    args.dann = True
    args.split_id = '56281'
    args.run_started = '28-02-23_2233'
    # end args overwrite
    
    # set dataset specific parameters
    if args.dataset == 'cifar10':
        args.no_class = 10
    elif args.dataset == 'cifar100':
        args.no_class = 100
    elif args.dataset == 'svhn':
        args.no_class = 10
    elif args.dataset == 'tinyimagenet':
        args.no_class = 200
    elif args.dataset == 'aircraft':
        args.no_class = 100
    elif args.dataset == 'stanfordcars':
        args.no_class = 196
    elif args.dataset == 'oxfordpets':
        args.no_class = 37
    elif args.dataset == 'imagenet100':
        args.no_class = 100
    elif args.dataset == 'herbarium':
        args.no_class = 682
    elif args.dataset == 'domainnet':
        args.no_class = 345
    elif args.dataset == 'pacs':
        args.no_class = 7
    elif args.dataset == 'officehome':
        args.no_class = 65
    
    base_path = '/home/biplab/Mainak/CrossDomainNCD/OpenLDN'
    args.data_root = os.path.join(base_path, args.data_root, args.dataset)
    args.fig_root = os.path.join(base_path, args.fig_root)
    os.makedirs(args.fig_root, exist_ok=True)
    args.exp_name = f'dataset_{args.dataset}_arch_{args.arch}_lbl_percent_{args.lbl_percent}_novel_percent_{args.novel_percent}_closed_wordl_ssl_{args.cw_ssl}_{args.dataset}_{args.description}_split_id_split_{args.split_id}_{args.run_started}'
    args.out = os.path.join(base_path, args.out, args.exp_name)
    args.resume = os.path.join(args.out, 'model_best_base.pth.tar')
    args.split_root = os.path.join(base_path, args.split_root)
    args.ssl_indexes = f'{args.split_root}/{args.dataset}_{args.lbl_percent}_{args.novel_percent}_{args.split_id}.pkl'
    args.no_known = args.no_class - int((args.novel_percent*args.no_class)/100)
    
    args.n_gpu = torch.cuda.device_count()
    if args.seed != -1:
        set_seed(args)
    
    model, _ = build_model(args)
    
    if args.resume:
        assert os.path.isfile(
            args.resume), f"Error: no checkpoint directory: {args.resume} found!"
        print(f'loaded best model checkpoint!')
        args.out = os.path.dirname(args.resume) # set output directory same as resume directory
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        print(f'model accuracy: {best_acc}')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    fig = plot(args, model)
    path = os.path.join(args.out, 'cross_domain.png')
    fig.savefig(path)
    
if __name__ == '__main__':
    main()