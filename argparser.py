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
    parser.add_argument('--out', default=f'outputs', help='directory to output the result')
    parser.add_argument('--no-progress', action='store_true', help="don't use progress bar")
    
    # dataset options
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['domainnet','pacs','officehome','visda17','office31'], help='dataset name')
    parser.add_argument('--no-class', default=10, type=int, help='total classes')
    parser.add_argument('--lbl-percent', type=int, default=100, help='percent of labeled data')
    parser.add_argument('--novel-percent', default=30, type=int, help='percentage of novel classes, default 30')
    parser.add_argument('--create-splits', default=False, required=False, type=bool, help='Whether to create splits splits for each domain. Default: False')
    parser.add_argument('--train-domain', type=str, help='train domain in case of cross domain setting')
    parser.add_argument('--test-domain', type=str, help='test domain in case of cross domain setting')
    parser.add_argument('--train-split', default=0.0, type=float, help='fraction of samples of cross domain in its train split, else in same domain')
    
    # model and training options
    parser.add_argument('--arch', default='resnet50', type=str, help='model architecture')
    parser.add_argument('--pretrained', default=f'', help='path to pretrained backbone model')
    parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run, deafult 50')
    parser.add_argument('--batch-size', default=64, type=int, help='train batchsize')
    parser.add_argument('--img-size', default=224, type=int, help='image size')
    parser.add_argument('--num-workers', type=int, default=4, help='number of workers')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate, default 5e-4')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', type=int, default=0, help="random seed (default: 0)")
   
    # method
    parser.add_argument('--dann', default=False, type=bool, help='run dann network to discriminate domain')
    parser.add_argument('--bottleneck-dim', default=-1, type=int, help='Dimension of bottleneck')
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
    # args.run_started = '02-05-23_1406'
    args.seed = 0
    args.dtype = torch.float32
    args.tsne = False # to be kept false. plotting internally handled
    args.tsne_freq = 10
    
    # dataset options
    args.dataset = 'officehome'
    # args.novel_percent = int((3/num_classes(args.dataset))*100)
    args.novel_percent = 0
    # args.train_domain = 'photo'
    # args.test_domain = 'art_painting'
    args.train_domain = 'Product'
    args.test_domain = 'Real_World'
    
    # model and training options
    args.arch = 'resnet50'
    # args.pretrained = 'resnet18_simclr_checkpoint_100.tar' # to use resnet18 simCLR ssl pretrained on STL10
    args.pretrained = 'swav_800ep_pretrain.pth.tar' # to use resnet50 swav ssl pretrained on imagenet
    args.epochs = 20
    args.batch_size = 32
    
    # method params
    args.contrastive = False
    args.dann = True
    args.bottleneck_dim = 256

    # ************end args overwrite******************
    
    # setup
    args.no_class = num_classes(args.dataset)
    if args.run_started == '':
        args.run_started = datetime.today().strftime('%d-%m-%y_%H%M')
    args.data_root = os.path.join(args.data_root, args.dataset) # make dataset root
    # download_dataset(args.dataset, 'OfficeHome', 'Pr', args.data_root)
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
    if dataset == 'domainnet':
        no_class = 345
    elif dataset == 'pacs':
        no_class = 7
    elif dataset == 'officehome':
        no_class = 65
    elif dataset == 'office31':
        no_class = 31
    elif dataset == 'visda17':
        no_class = 12
    return no_class