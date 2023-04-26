import numpy as np
import math
import os
import random
import time
import pickle
from datetime import datetime
import argparser


def main(run_started):
    
    args = argparser.get_args()
    
    # args.split_id = split_id
    
    args.data_root = os.path.join(args.data_root, args.dataset) # make dataset root
    os.makedirs(args.data_root, exist_ok=True)
    # os.makedirs(args.split_root, exist_ok=True)
    best_acc = 0
    # args.exp_name = f'dataset_{args.dataset}_arch_{args.arch}_lbl_percent_{args.lbl_percent}_novel_percent_{args.novel_percent}_closed_wordl_ssl_{args.cw_ssl}_{args.description}_split_id_{args.split_id}_{run_started}'
    args.exp_name = f'dataset_{args.dataset}_arch_{args.arch}_novel_percent_{args.novel_percent}_{args.description}_{run_started}'
    # args.ssl_indexes = f'{args.split_root}/{args.dataset}_{args.lbl_percent}_{args.novel_percent}_{args.split_id}.pkl'

    args.out = os.path.join(args.out, args.exp_name)
    os.makedirs(args.out, exist_ok=True)

    # run base experiment
    if args.dataset in ['cifar10', 'cifar100', 'svhn', 'tinyimagenet', 'pacs','officehome','visda17']:
        # os.system(f"python base/train-base.py --dataset {args.dataset} --lbl-percent {args.lbl_percent} --novel-percent {args.novel_percent} --out {args.out} --ssl-indexes {args.ssl_indexes} --split-id {args.split_id} --run-started {run_started}")
        os.system(f"python train-base-new.py --dataset {args.dataset} --lbl-percent {args.lbl_percent} --novel-percent {args.novel_percent} --out {args.out} --ssl-indexes {args.ssl_indexes} --split-id {args.split_id} --run-started {run_started}")
    elif args.dataset in ['oxfordpets', 'aircraft', 'stanfordcars', 'herbarium']:
        # higher batch size.
        os.system(f"python base/train-base.py --dataset {args.dataset} --lbl-percent {args.lbl_percent} --novel-percent {args.novel_percent} --batch-size 512 --out {args.out} --ssl-indexes {args.ssl_indexes} --split-id {args.split_id}")
    
    elif args.dataset == 'imagenet100':
        # higher batch size, and higher lr
        os.system(f"python base/train-base.py --dataset {args.dataset} --lbl-percent {args.lbl_percent} --novel-percent {args.novel_percent} --lr 1-2 --batch-size 512 --out {args.out} --ssl-indexes {args.ssl_indexes} --split-id {args.split_id}")

if __name__ == '__main__':
    run_started = datetime.today().strftime('%d-%m-%y_%H%M')
    # split_id = f'split_{random.randint(1, 100000)}'
    # main(run_started, split_id)
    main(run_started)
    
