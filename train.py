import numpy as np
import argparse
import math
import os
import random
import time
import pickle
from datetime import datetime


def main(run_started, split_id):
    parser = argparse.ArgumentParser(description='OpenLDN Training')
    parser.add_argument('--data-root', default=f'data', help='directory to store data')
    parser.add_argument('--split-root', default=f'random_splits', help='directory to store datasets')
    parser.add_argument('--out', default=f'outputs', help='directory to output the result')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100', 'svhn', 'tinyimagenet', 'oxfordpets', 'aircraft', 'stanfordcars', 'imagenet100', 'herbarium', 'domainnet', 'pacs','officehome'], help='dataset name')
    parser.add_argument('--lbl-percent', type=int, default=50, help='percent of labeled data')
    parser.add_argument('--novel-percent', default=50, type=int, help='percentage of novel classes, default 50')
    parser.add_argument('--arch', default='resnet18', type=str, help='model architecture')
    parser.add_argument('--cw-ssl', default='mixmatch', type=str, choices=['mixmatch', 'uda'], help='closed-world SSL method to use')
    parser.add_argument('--description', default='default_run', type=str, help='description of the experiment')
    # parser.add_argument('--dann', default=False, type=bool, help='run dann network to discriminate domain')

    args = parser.parse_args()
    
    args.split_id = split_id
    
    # overwrite command line args here
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    args.description = 'officehome_dann'
    # args.split_id = 'split_9541'
    # run_started = '19-03-23_1532'
    
    # end overwrite
    
    args.data_root = os.path.join(args.data_root, args.dataset) # make dataset root
    os.makedirs(args.data_root, exist_ok=True)
    os.makedirs(args.split_root, exist_ok=True)
    best_acc = 0
    args.exp_name = f'dataset_{args.dataset}_arch_{args.arch}_lbl_percent_{args.lbl_percent}_novel_percent_{args.novel_percent}_closed_wordl_ssl_{args.cw_ssl}_{args.description}_split_id_{args.split_id}_{run_started}'
    args.ssl_indexes = f'{args.split_root}/{args.dataset}_{args.lbl_percent}_{args.novel_percent}_{args.split_id}.pkl'

    args.out = os.path.join(args.out, args.exp_name)
    os.makedirs(args.out, exist_ok=True)


    # run base experiment
    if args.dataset in ['cifar10', 'cifar100', 'svhn', 'tinyimagenet', 'pacs','officehome']:
        # os.system(f"python base/train-base.py --dataset {args.dataset} --lbl-percent {args.lbl_percent} --novel-percent {args.novel_percent} --out {args.out} --ssl-indexes {args.ssl_indexes} --split-id {args.split_id} --run-started {run_started}")
        os.system(f"python base/train-base-new.py --dataset {args.dataset} --lbl-percent {args.lbl_percent} --novel-percent {args.novel_percent} --out {args.out} --ssl-indexes {args.ssl_indexes} --split-id {args.split_id} --run-started {run_started}")
    elif args.dataset in ['oxfordpets', 'aircraft', 'stanfordcars', 'herbarium']:
        # higher batch size.
        os.system(f"python base/train-base.py --dataset {args.dataset} --lbl-percent {args.lbl_percent} --novel-percent {args.novel_percent} --batch-size 512 --out {args.out} --ssl-indexes {args.ssl_indexes} --split-id {args.split_id}")
    
    elif args.dataset == 'imagenet100':
        # higher batch size, and higher lr
        os.system(f"python base/train-base.py --dataset {args.dataset} --lbl-percent {args.lbl_percent} --novel-percent {args.novel_percent} --lr 1-2 --batch-size 512 --out {args.out} --ssl-indexes {args.ssl_indexes} --split-id {args.split_id}")
    

    # only compare till base training
    return
    # run closed-world SSL experiment
    if args.dataset in ['cifar10', 'cifar100', 'svhn', 'tinyimagenet', 'pacs','officehome']:
        os.system(f"python closed_world_ssl/train-{args.cw_ssl}.py --dataset {args.dataset} --lbl-percent {args.lbl_percent} --novel-percent {args.novel_percent} --out {args.out} --ssl-indexes {args.ssl_indexes} --run-started {run_started}")
    
    elif args.dataset in ['oxfordpets', 'aircraft', 'stanfordcars', 'herbarium']:
        # higher batch size, and lower epochs
        os.system(f"python closed_world_ssl/train-{args.cw_ssl}.py --dataset {args.dataset} --lbl-percent {args.lbl_percent} --novel-percent {args.novel_percent} --batch-size 512 --epochs 200 --out {args.out} --ssl-indexes {args.ssl_indexes}")
    
    elif args.dataset == 'imagenet100':
        # higher batch size, lower epochs, and larger network
        os.system(f"python closed_world_ssl/train-{args.cw_ssl}.py --dataset {args.dataset} --lbl-percent {args.lbl_percent} --novel-percent {args.novel_percent} --batch-size 512 --epochs 200 --arch resnet50 --out {args.out} --ssl-indexes {args.ssl_indexes}")


if __name__ == '__main__':
    run_started = datetime.today().strftime('%d-%m-%y_%H%M')
    split_id = f'split_{random.randint(1, 100000)}'
    main(run_started, split_id)
