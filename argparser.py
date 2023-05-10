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
    parser.add_argument('--dataset', default='office31', type=str,
                        choices=['domainnet','pacs','officehome','visda17','office31'], help='dataset name')
    parser.add_argument('--no-class', default=31, type=int, help='total classes')
    parser.add_argument('--lbl-percent', type=int, default=100, help='percent of labeled data')
    parser.add_argument('--no-novel', default=0, type=int, help='number of novel classes,  default: 0')
    parser.add_argument('--create-splits', default=False, required=False, type=bool, help='Whether to create splits splits for each domain. Default: False')
    parser.add_argument('--train-domain', type=str, help='train domain in case of cross domain setting')
    parser.add_argument('--test-domain', type=str, help='test domain in case of cross domain setting')
    parser.add_argument('--train-split', default=0.0, type=float, help='fraction of samples of cross domain in its train split, else in same domain')
    
    # model and training options
    parser.add_argument('--arch', default='resnet50', type=str, help='model architecture')
    parser.add_argument('--pretrained', default=f'', help='path to pretrained backbone model')
    parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run, deafult 50')
    parser.add_argument('--batch-size', default=64, type=int, help='train batchsize')
    parser.add_argument('--img-size', default=224, type=int, help='image size')
    parser.add_argument('--num-workers', type=int, default=4, help='number of workers')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', type=int, default=0, help="random seed (default: 0)")
    parser.add_argument('--iteration', default=1000, type=int,
                        help='Number of iterations per epoch')
    
    # Optimizer options
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer type (default: sgd)')
    parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-3, help='weight decay (default: 1e-3)')

    # Scheduler options
    parser.add_argument('--scheduler', type=str, default='lambda', help='scheduler type (default: step)')
    parser.add_argument('--lr_gamma', default=0.001, type=float, help='learning rate decay factor for lambda scheduler (default=0.001)')
    parser.add_argument('--lr_decay', default=0.75, type=float, help='power factor for lambda scheduler (default=0.75)')
    # parser.add_argument('--gamma', default=0.1, type=float, help='learning rate decay factor for step scheduler (default=0.1)')
    # parser.add_argument('--step_size', default=30, type=int, help='period of learning rate decay for step scheduler (default: 30)')
    # parser.add_argument('--t_max', default=100, type=int, help='maximum number of iterations for cosine scheduler (default: 100)')
    # parser.add_argument('--eta_min', default=0, type=float, help='minimum learning rate for cosine scheduler (default: 0.0)')
    # parser.add_argument('--mode', default='min', type=str, help='monitor mode for reduce_on_plateau scheduler (default: min)')
    # parser.add_argument('--factor', default=0.1, type=float, help='learning rate decay factor for reduce_on_plateau scheduler (default: 0.1)')
    # parser.add_argument('--patience', default=10, type=int, help='patience for reduce_on_plateau scheduler (default: 10)')
    
    # method
    parser.add_argument('--method', default='dann', type=str,
                        choices=['dann', 'dsbn', 'contrastive', 'dann_contrastive'], help='method name')
    parser.add_argument('--bottleneck-dim', default=-1, type=int, help='Dimension of bottleneck in case of dann')
    parser.add_argument('--temp', type=float, default=0.07, help='temperature for ssl loss function')

    args = parser.parse_args()
    
    # ************overwrite command line args here******************
    # Basic directory and setup options
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    args.disk_dataset_path = os.environ['DATASETS_ROOT']
    # args.run_started = '02-05-23_1406'
    args.seed = 0
    args.tsne = False # to be kept false. plotting internally handled
    args.tsne_freq = 10
    
    # dataset options
    args.dataset = 'pacs'
    args.no_novel = 0
    args.train_domain = 'photo'
    args.test_domain = 'art_painting'
    # args.train_domain = 'Product'
    # args.test_domain = 'Real_World'
    
    # model and training options
    args.arch = 'resnet50'
    args.pretrained = 'swav_800ep_pretrain.pth.tar' # to use resnet50 swav ssl pretrained on imagenet
    args.iteration = 10
    args.epochs = 20
    args.batch_size = 2
    args.scheduler = 'lambda'
    
    # method params
    args.method = 'contrastive'
    # args.method = 'dann'
    # args.bottleneck_dim = 256
    # ************end args overwrite******************
    
    # setup
    args.description = f'{args.method}'
    args.no_class = num_classes(args.dataset)
    args.no_known = args.no_class - args.no_novel
    if args.run_started == '':
        args.run_started = datetime.today().strftime('%d-%m-%y_%H%M')
    args.data_root = os.path.join(args.data_root, args.dataset) # make dataset root
    # download_dataset(args.dataset, 'PACS', 'A', args.data_root)
    args.exp_name = f'dataset_{args.dataset}_arch_{args.arch}_no_novel_{args.no_novel}_{args.description}_{args.run_started}'
    args.out = os.path.join(args.out, args.exp_name)
    args.n_gpu = torch.cuda.device_count()
    # args.resume = os.path.join(args.out, 'checkpoint_base.pth.tar') # uncomment when need to resume training
    
    # set seed
    if args.seed != -1:
        set_seed(args)

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