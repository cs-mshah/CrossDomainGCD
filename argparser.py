import os
import argparse
from datetime import datetime
import torch
from utils.utils import set_seed
from utils.tllib_utils import download_dataset

def get_args():
    parser = argparse.ArgumentParser(description='Cross Domain GCD')
    
    # Basic directory and setup options
    parser.add_argument('--description', default='default_run', type=str, help='description of the experiment')
    parser.add_argument('--data-root', default=f'data', help='directory to store data')
    parser.add_argument('--run-started', default=f'', help='d-m-y_H_M time of run start')
    # parser.add_argument('--split-root', default=f'random_splits', help='directory to store datasets')
    # parser.add_argument('--cw-ssl', default='mixmatch', type=str, choices=['mixmatch', 'uda'], help='closed-world SSL method to use')
    parser.add_argument('--fig-root', default=f'tsne_plots', help='directory to store plots')
    parser.add_argument('--out', default=f'outputs', help='directory to output the result')
    # parser.add_argument('--split-id', default='split_0', type=str, help='random data split number')
    # parser.add_argument('--ssl-indexes', default='', type=str, help='path to random data split')
    parser.add_argument('--no-progress', action='store_true', help="don't use progress bar")
    
    
    # dataset options
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100', 'svhn', 'tinyimagenet', 'oxfordpets', 'aircraft', 'stanfordcars', 'imagenet100', 'herbarium',
                                 'domainnet', 'pacs','officehome','visda17'], help='dataset name')
    parser.add_argument('--no-class', default=10, type=int, help='total classes')
    parser.add_argument('--lbl-percent', type=int, default=100, help='percent of labeled data')
    parser.add_argument('--novel-percent', default=30, type=int, help='percentage of novel classes, default 50')
    parser.add_argument('--create-splits', default=True, required=False, type=bool, help='Whether to create splits splits for each domain. Default: True')
    # parser.add_argument('--pl-percent', type=int, default=10, help='percent of selected pseudo-labels data')
    parser.add_argument('--train-domain', type=str, help='train domain in case of cross domain setting')
    parser.add_argument('--test-domain', type=str, help='test domain in case of cross domain setting')
    parser.add_argument('--train-split', default=0.0, type=float, help='fraction of samples of cross domain in its train split, else in same domain')
    
    # model and training options
    parser.add_argument('--arch', default='resnet18', type=str, help='model architecture')
    parser.add_argument('--pretrained', default=f'', help='path to pretrained backbone model')
    parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run, deafult 50')
    parser.add_argument('--batch-size', default=200, type=int, help='train batchsize')
    parser.add_argument('--img-size', default=224, type=int, help='image size')
    parser.add_argument('--num-workers', type=int, default=4, help='number of workers')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate, default 5e-4')
    # parser.add_argument('--lr-simnet', default=1e-4, type=float, help='earning rate for simnet, default 1e-4')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', type=int, default=-1, help="random seed (-1: don't use random seed)")
    # parser.add_argument('--mu', default=1, type=int, help='coefficient of unlabeled batch size')
    # parser.add_argument('--threshold', default=0.5, type=float, help='pseudo-label threshold, default 0.50')
    
    # method
    parser.add_argument('--dann', default=False, type=bool, help='run dann network to discriminate domain')
    parser.add_argument('--alpha', required=False, type=float, help='value of alpha for dann. (default:0.25)')
    parser.add_argument('--alpha_exp', required=False, type=float, help='whether to use alpha exponential decaying as described in the dann paper')
    parser.add_argument('--dsbn', default=False, type=bool, help='run dsbn for domain adaptation')
    parser.add_argument('--mmd', default=False, type=bool, help='use mmd for domain adaptation')
    parser.add_argument('--contrastive', default=False, type=bool, help='use supervised and self supervised contrastive loss')
    parser.add_argument('--temp',required=False, type=float, default=0.07, help='temperature for loss function')

    args = parser.parse_args()
    
    # ************overwrite command line args here******************
    # Basic directory and setup options
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    args.description = 'officehome_dann'
    args.disk_dataset_path = os.environ['DATASETS_ROOT']
    # args.split_id = 'split_46126' # for checkpoint reload (old)
    # run_started = '31-03-23_1926'
    args.seed = 0
    args.dtype = torch.float32
    args.tsne = False # to be kept false. plotting internally handled
    args.tsne_freq = 10
    
    # dataset options
    args.create_splits = False # if any domain arg is changed, then make True to create new splits
    args.train_split = 0.0 # if cross domain setup, then this is cross_domain train split(0.0) (as train domain uses full), else same domain train split
    args.dataset = 'officehome'
    args.lbl_percent = 100
    args.novel_percent = 0
    args.train_domain = 'Product'
    args.test_domain = 'RealWorld'
    
    # model and training options
    args.pretrained = 'resnet18_simclr_checkpoint_100.tar' # to use resnet18 simCLR pretrained on STL10
    args.epochs = 100
    args.batch_size = 64
    
    # method params
    args.contrastive = False
    args.dann = True
    # args.alpha = 0.25
    args.alpha_exp = True
    # ************end args overwrite******************
    
    # setup
    args.no_class = num_classes(args.dataset)
    # download_dataset(args.dataset, 'Office31', 'A', args.data_root)
    if args.run_started == '':
        args.run_started = datetime.today().strftime('%d-%m-%y_%H%M')
    args.data_root = os.path.join(args.data_root, args.dataset) # make dataset root
    args.exp_name = f'dataset_{args.dataset}_arch_{args.arch}_novel_percent_{args.novel_percent}_{args.description}_{args.run_started}'
    args.out = os.path.join(args.out, args.exp_name)
    args.n_gpu = torch.cuda.device_count()
    # args.resume = os.path.join(args.out, 'checkpoint_base.pth.tar') # uncomment when need to resume training
    
    # set seed
    if args.seed != -1:
        set_seed(args)
    args.no_known = args.no_class - int((args.novel_percent*args.no_class)/100)

    return args


def num_classes(dataset:str):
    """returns the total number of classes for a given dataset name"""
    no_class = 0
    if dataset == 'cifar10':
        no_class = 10
    elif dataset == 'cifar100':
        no_class = 100
    elif dataset == 'svhn':
        no_class = 10
    elif dataset == 'tinyimagenet':
        no_class = 200
    elif dataset == 'aircraft':
        no_class = 100
    elif dataset == 'stanfordcars':
        no_class = 196
    elif dataset == 'oxfordpets':
        no_class = 37
    elif dataset == 'imagenet100':
        no_class = 100
    elif dataset == 'herbarium':
        no_class = 682
    elif dataset == 'domainnet':
        no_class = 345
    elif dataset == 'pacs':
        no_class = 7
    elif dataset == 'officehome':
        no_class = 65
    elif dataset == 'visda17':
        no_class = 12
    return no_class