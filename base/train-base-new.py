import numpy as np
import argparse
import os
import random
import time
import pickle
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm
import sys
sys.path.insert(0, '/home/biplab/Mainak/CrossDomainNCD/OpenLDN/base')
from models.build_model import build_model
from utils.utils import AverageMeter, accuracy, set_seed, save_checkpoint, sim_matrix, interleave, de_interleave, describe_splits, describe_image_dataset
from datasets.datasets import get_dataset, get_cross_domain
from datasets.multi_domain import create_dataset
from utils.evaluate_utils import hungarian_evaluate
from losses.losses import entropy, symmetric_mse_loss
from utils.pseudo_labeling_utils import pseudo_labeling
import wandb
from visualization.tsne import plot


def main():
    parser = argparse.ArgumentParser(description='Base Training')
    parser.add_argument('--data-root', default=f'data', help='directory to store data')
    parser.add_argument('--run-started', default=f'', help='d-m-y_H_M time of run start')
    parser.add_argument('--split-root', default=f'random_splits', help='directory to store datasets')
    parser.add_argument('--fig-root', default=f'tsne_plots', help='directory to plots')
    parser.add_argument('--pretrained', default=f'', help='path to pretrained backbone model')
    parser.add_argument('--out', default=f'outputs', help='directory to output the result')
    parser.add_argument('--num-workers', type=int, default=4, help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100', 'svhn', 'tinyimagenet', 'oxfordpets', 'aircraft', 'stanfordcars', 'imagenet100', 'herbarium', 'domainnet', 'pacs','officehome'], help='dataset name')
    parser.add_argument('--lbl-percent', type=int, default=50, help='percent of labeled data')
    parser.add_argument('--novel-percent', default=50, type=int, help='percentage of novel classes, default 50')
    parser.add_argument('--pl-percent', type=int, default=10, help='percent of selected pseudo-labels data')
    parser.add_argument('--arch', default='resnet18', type=str, help='model architecture')
    parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run, deafult 50')
    parser.add_argument('--batch-size', default=200, type=int, help='train batchsize')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate, default 1e-3')
    parser.add_argument('--lr-simnet', default=1e-4, type=float, help='earning rate for simnet, default 1e-4')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', type=int, default=-1, help="random seed (-1: don't use random seed)")
    parser.add_argument('--mu', default=1, type=int, help='coefficient of unlabeled batch size')
    parser.add_argument('--no-progress', action='store_true', help="don't use progress bar")
    parser.add_argument('--no-class', default=10, type=int, help='total classes')
    parser.add_argument('--threshold', default=0.5, type=float, help='pseudo-label threshold, default 0.50')
    parser.add_argument('--split-id', default='split_0', type=str, help='random data split number')
    parser.add_argument('--ssl-indexes', default='', type=str, help='path to random data split')
    parser.add_argument('--dann', default=False, type=bool, help='run dann network to discriminate domain')
    parser.add_argument('--train-domain', required=False, type=str, help='train domain in case of cross domain setting')
    parser.add_argument('--test-domain', required=False, type=str, help='test domain in case of cross domain setting')
    parser.add_argument('--train-split', default=0.4, required=False, type=int, help='fraction of samples of cross domain in training (when dann), else in same domain')
    parser.add_argument('--create-splits', default=True, required=False, type=bool, help='Whether to create splits splits for each domain. Default: True')
    parser.add_argument('--alpha', default=0.25, required=False, type=float, help='value of alpha for dann')
    parser.add_argument('--alpha_exp', default=False, required=False, type=float, help='whether to use alpha exponential decaying as described in the paper')

    args = parser.parse_args()
    best_acc = 0
    
    # overwrite command line args here
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    args.seed = 0
    args.tsne = False
    args.pretrained = 'checkpoint_100.tar' # to use resnet18 simCLR pretrained on STL10
    args.epochs = 60
    args.batch_size = 64
    # args.resume = os.path.join(args.out, 'checkpoint_base.pth.tar') # only when need to resume training
    args.create_splits = False # if any domain arg is changed, then make True to create new splits
    args.train_split = 0.0 # if dann true, then this is cross_domain train split(0.0) (as train domain uses full), else same domain train split
    # cross domain args
    args.dann = True
    # args.alpha = 0.20
    args.alpha_exp = True
    args.train_domain = 'Product'
    args.test_domain = 'RealWorld'
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
    
    print(' | '.join(f'{k}={v}' for k, v in vars(args).items()))

    writer = SummaryWriter(logdir=args.out)

    with open(f'{args.out}/score_logger_base.txt', 'a+') as ofile:
        ofile.write('************************************************************************\n\n')
        ofile.write(' | '.join(f'{k}={v}' for k, v in vars(args).items()))
        ofile.write('\n\n************************************************************************\n')
    
    args.n_gpu = torch.cuda.device_count()

    args.dtype = torch.float32
    if args.seed != -1:
        set_seed(args)

    args.data_root = os.path.join(args.data_root, args.dataset)
    os.makedirs(args.data_root, exist_ok=True)
    os.makedirs(args.split_root, exist_ok=True)
    
    if args.create_splits:
        create_dataset(args)
    
    describe_splits(args, ignore_errors=True)
    # load dataset
    args.no_known = args.no_class - int((args.novel_percent*args.no_class)/100)
    lbl_dataset, unlbl_dataset, pl_dataset, test_dataset_known, test_dataset_novel, test_dataset_all = get_dataset(args)
    if args.dann:
        # crossdom_dataset = get_cross_domain(args)
        crossdom_dataset = test_dataset_all
    print(f'known classes: {args.no_known}/{args.no_class}, label %: {args.lbl_percent}%')
    # create dataloaders
    unlbl_batchsize = int((float(args.batch_size) * len(unlbl_dataset))/(len(lbl_dataset) + len(unlbl_dataset)))
    lbl_batchsize = args.batch_size - unlbl_batchsize
    # args.iteration = (len(lbl_dataset) + len(unlbl_dataset)) // args.batch_size
    args.iteration = (len(lbl_dataset)) // args.batch_size

    train_sampler = RandomSampler
    lbl_loader = DataLoader(lbl_dataset, sampler=train_sampler(lbl_dataset), batch_size=lbl_batchsize, num_workers=args.num_workers, drop_last=True)
    unlbl_loader = DataLoader(unlbl_dataset, sampler=train_sampler(unlbl_dataset), batch_size=unlbl_batchsize, num_workers=args.num_workers, drop_last=True)
    # pl_loader = DataLoader(pl_dataset, sampler=SequentialSampler(pl_dataset), batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False)
    test_loader_known = DataLoader(test_dataset_known, sampler=SequentialSampler(test_dataset_known), batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False)
    # test_loader_novel = DataLoader(test_dataset_novel, sampler=SequentialSampler(test_dataset_novel), batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False)
    test_loader_all = DataLoader(test_dataset_all, sampler=SequentialSampler(test_dataset_all), batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False)
    
    if args.dann:
        crossdom_loader = DataLoader(crossdom_dataset, sampler=train_sampler(crossdom_dataset), batch_size=lbl_batchsize, num_workers=args.num_workers, drop_last=True)
    else:
        crossdom_loader = None
        # only for this run to check on cross domain to see dann improvement
        args.dann = True
        args.test_domain = 'RealWorld'
        _, _, _, cross_dataset_known, _, _ = get_dataset(args)
        # describe_image_dataset(cross_dataset_known, ignore_errors=True)
        cross_loader_known = DataLoader(cross_dataset_known, sampler=SequentialSampler(cross_dataset_known), batch_size=lbl_batchsize, num_workers=args.num_workers, drop_last=False)
        args.test_domain = 'Product'
        args.dann = False

    # create model
    model, _ = build_model(args)
    model = model.cuda()
    
    # optimizer
    if args.n_gpu > 1:
        optimizer = torch.optim.Adam(model.module.params(), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.params(), lr=args.lr)

    start_epoch = 0
    if args.resume:
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        print(f'resuming..\n')
        args.out = os.path.dirname(args.resume) # set output directory same as resume directory
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    if args.pretrained:
        model_fp = os.path.join('/home/biplab/Mainak/CrossDomainNCD/OpenLDN/pretrained/', args.pretrained)
        device = 'cuda:0'
        map_location = torch.device(device)
        saved_model = torch.load(model_fp, map_location=map_location)
        new_state_dict = {}
        for key, value in saved_model.items():
            if key.startswith('encoder.'):
                new_key = key[len('encoder.'):]
            else:
                new_key = key
            new_state_dict[new_key] = value
        keys_model = set(model.state_dict().keys())
        keys_pretrained = set(new_state_dict.keys())
        diff_keys1 = keys_model - keys_pretrained
        diff_keys2 = keys_pretrained - keys_model
        # print the differences in the keys
        print(f"Keys in model but not in pretrained model: {diff_keys1}")
        print(f"Keys in pretrained model but not in model: {diff_keys2}")
        model.load_state_dict(new_state_dict, strict=False)
    
    ###################
    ##### Training ####
    ###################
    
    # wandb init
    wandb.init(entity='cv-exp',
               project='IITB-MBZUAI',
               name=f'{args.dataset} {args.train_domain}->{args.test_domain} {args.run_started}',
               config=args,
               group=f'{args.dataset}',
               allow_val_change=True,
               sync_tensorboard=True)
    
    test_accs = []
    model.zero_grad()
    for epoch in range(start_epoch, args.epochs):
        train_loss = train(args, lbl_loader, unlbl_loader, model, optimizer, epoch, crossdom_loader)
        test_acc_known = test_known(args, test_loader_known, model, epoch)
        # novel_cluster_results = test_cluster(args, test_loader_novel, model, epoch, offset=args.no_known)
        # all_cluster_results = test_cluster(args, test_loader_all, model, epoch)
        # test_acc = all_cluster_results["acc"]

        is_best = test_acc_known > best_acc
        best_acc = max(test_acc_known, best_acc)

        # cross_acc_known = test_known(args, cross_loader_known, model, epoch)
        if (epoch + 1) % 10 == 0:
            fig = plot(args, model)
            args.tsne = False
            # path = os.path.join(args.fig_root, args.run_started, 'base-train')
            # os.makedirs(path, exist_ok=True)
            # fig.savefig(os.path.join(path, f'epoch_{epoch+1}.png'))
            wandb.log({"tsne": wandb.Image(fig)}, commit=False)
        
        print(f'epoch: {epoch}, acc-known: {test_acc_known}')
        # print(f'epoch: {epoch}, cross-acc-known: {cross_acc_known}')
        # print(f'epoch: {epoch}, acc-novel: {novel_cluster_results["acc"]}, nmi-novel: {novel_cluster_results["nmi"]}')
        # print(f'epoch: {epoch}, acc-all: {all_cluster_results["acc"]}, nmi-all: {all_cluster_results["nmi"]}, best-acc: {best_acc}')

        wandb.log({'train_loss':train_loss}, commit=False)
        wandb.log({'acc-known':test_acc_known})
        # wandb.log({'cross-acc-known':cross_acc_known})
        # wandb.log({'acc-novel':novel_cluster_results["acc"]})
        # wandb.log({'nmi-novel':novel_cluster_results["nmi"]})
        # wandb.log({'acc-all':all_cluster_results["acc"]})
        # wandb.log({'nmi-all':all_cluster_results["nmi"]})
        
        writer.add_scalar('train/1.train_loss', train_loss, epoch)
        writer.add_scalar('test/1.acc_known', test_acc_known, epoch)
        # writer.add_scalar('test/2.cross_acc_known', cross_acc_known, epoch)
        # writer.add_scalar('test/2.acc_novel', novel_cluster_results['acc'], epoch)
        # writer.add_scalar('test/3.nmi_novel', novel_cluster_results['nmi'], epoch)
        # writer.add_scalar('test/4.acc_all', all_cluster_results['acc'], epoch)
        # writer.add_scalar('test/5.nmi_all', all_cluster_results['nmi'], epoch)

        model_to_save = model.module if hasattr(model, "module") else model
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model_to_save.state_dict(),
            'acc': test_acc_known,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.out, tag='base')

        test_accs.append(test_acc_known)

        with open(f'{args.out}/score_logger_base.txt', 'a+') as ofile:
            ofile.write(f'epoch: {epoch}, acc-known: {test_acc_known}\n')
            # ofile.write(f'epoch: {epoch}, acc-novel: {novel_cluster_results["acc"]}, nmi-novel: {novel_cluster_results["nmi"]}\n')
            # ofile.write(f'epoch: {epoch}, acc-all: {all_cluster_results["acc"]}, nmi-all: {all_cluster_results["nmi"]}, best-acc: {best_acc}\n')

    # close writer
    writer.close()


def train(args, lbl_loader, unlbl_loader, model, optimizer, epoch, crossdom_loader=None):

    model.train()
    lbl_batchsize = lbl_loader.batch_size
    unlbl_batchsize = unlbl_loader.batch_size
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_ce = AverageMeter()
    # losses_pair = AverageMeter()
    # losses_reg = AverageMeter()
    losses_dann_source = AverageMeter()
    losses_dann_target = AverageMeter()

    end = time.time()
    if not args.no_progress:
        args.iteration = len(lbl_loader)
        p_bar = tqdm(range(args.iteration))

    train_loader = zip(lbl_loader, unlbl_loader)
    # for batch_idx, (data_lbl, data_unlbl) in enumerate(train_loader):
    for batch_idx, data_lbl in enumerate(lbl_loader):
        inputs_l, targets_l, _ = data_lbl
        
        inputs_l = inputs_l.cuda()
        targets_l = targets_l.cuda()
        
        _, logits_l = model(inputs_l)
        loss_ce = F.cross_entropy(logits_l, targets_l)
        
        final_loss = loss_ce.clone()
        
        if args.dann:
            # Load target batch
            target_images, _ = next(iter(crossdom_loader))
            target_images = target_images.cuda()
            
            if args.alpha_exp:
                # alpha exponential decaying as described in the paper
                len_dataloader = min(len(lbl_loader), len(crossdom_loader))
                p = float(batch_idx + epoch * len_dataloader) / args.epochs / len_dataloader
                args.alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
            # STEP 2: train the discriminator: forward SOURCE data to Gd
            # labeled
            outputs_l = model.forward(inputs_l, alpha=args.alpha)
            # source's label is 0 for all data
            labels_discr_source_l = torch.zeros(lbl_batchsize, dtype=torch.int64).cuda()
            loss_dann_source = F.cross_entropy(outputs_l, labels_discr_source_l)
            
            # STEP 3: train the discriminator: forward TARGET to Gd
            outputs = model.forward(target_images, alpha=args.alpha)
            labels_discr_target = torch.ones(lbl_batchsize, dtype=torch.int64).cuda() # target's label is 1
            loss_dann_target = F.cross_entropy(outputs, labels_discr_target)

            final_loss += loss_dann_source + loss_dann_target

            losses_dann_source.update(loss_dann_source.item(), inputs_l.size(0))
            losses_dann_target.update(loss_dann_target.item(), inputs_l.size(0))
            
        losses.update(final_loss.item(), inputs_l.size(0))
        losses_ce.update(loss_ce.item(), inputs_l.size(0))

        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if not args.no_progress:
            p_bar.set_description("train epoch: {epoch}/{epochs:4}. itr: {batch:4}/{iter:4}. btime: {bt:.3f}s. loss: {loss:.4f}. l_ce: {loss_ce:.4f}. l_dann_s: {loss_dann_s:.4f}. l_dann_t: {loss_dann_t:.4f}".format(
                epoch=epoch + 1,
                epochs=args.epochs,
                batch=batch_idx + 1,
                iter=args.iteration,
                bt=batch_time.avg,
                loss=losses.avg,
                loss_ce=losses_ce.avg,
                loss_dann_s=losses_dann_source.avg,
                loss_dann_t=losses_dann_target.avg
                ))
            p_bar.update()

    if not args.no_progress:
        p_bar.close()

    wandb.log({'loss_ce':losses_ce.avg}, commit=False)
    if args.dann:
        wandb.log({'loss_dann_source':losses_dann_source.avg}, commit=False)
        wandb.log({'loss_dann_target':losses_dann_target.avg}, commit=False)
    
    return losses.avg


def test_known(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    model.eval()

    if not args.no_progress:
        test_loader = tqdm(test_loader)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.cuda()
            targets = targets.cuda()
            _, outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description("test epoch: {epoch}/{epochs:4}. itr: {batch:4}/{iter:4}. btime: {bt:.3f}s. loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        if not args.no_progress:
            test_loader.close()

    wandb.log({'test_known_loss_ce':losses.avg}, commit=False)
        
    return top1.avg


def test_cluster(args, test_loader, model, epoch, offset=0):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    gt_targets =[]
    predictions = []
    model.eval()

    if not args.no_progress:
        test_loader = tqdm(test_loader)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)

            inputs = inputs.cuda()
            targets = targets.cuda()
            _, outputs = model(inputs)
            _, max_idx = torch.max(outputs, dim=1)
            predictions.extend(max_idx.cpu().numpy().tolist())
            gt_targets.extend(targets.cpu().numpy().tolist())
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description("test epoch: {epoch}/{epochs:4}. itr: {batch:4}/{iter:4}. btime: {bt:.3f}s.".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    bt=batch_time.avg,
                ))
        if not args.no_progress:
            test_loader.close()

    predictions = np.array(predictions)
    gt_targets = np.array(gt_targets)

    predictions = torch.from_numpy(predictions)
    gt_targets = torch.from_numpy(gt_targets)
    eval_output = hungarian_evaluate(predictions, gt_targets, offset)

    return eval_output


if __name__ == '__main__':
    cudnn.benchmark = True
    main()
