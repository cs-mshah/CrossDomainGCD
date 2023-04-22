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
from tqdm import tqdm
from models.build_model import build_model
from utils.utils import Losses, AverageMeter, accuracy, set_seed, save_checkpoint, describe_splits, describe_image_dataset, num_classes
from datasets.datasets import get_dataset
from datasets.multi_domain import create_dataset
from utils.evaluate_utils import hungarian_evaluate
from losses import MMDLoss, SupConLoss
import wandb
from visualization.tsne import plot
from torchsummary import summary


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
                        choices=['cifar10', 'cifar100', 'svhn', 'tinyimagenet', 'oxfordpets', 'aircraft', 'stanfordcars', 'imagenet100', 'herbarium', 'domainnet', 'pacs','officehome','visda17'], help='dataset name')
    parser.add_argument('--lbl-percent', type=int, default=50, help='percent of labeled data')
    parser.add_argument('--novel-percent', default=50, type=int, help='percentage of novel classes, default 50')
    parser.add_argument('--pl-percent', type=int, default=10, help='percent of selected pseudo-labels data')
    parser.add_argument('--arch', default='resnet18', type=str, help='model architecture')
    parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run, deafult 50')
    parser.add_argument('--batch-size', default=200, type=int, help='train batchsize')
    parser.add_argument('--img-size', default=224, type=int, help='image size')
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
    parser.add_argument('--train-split', required=False, type=int, help='fraction of samples of cross domain in training (when dann), else in same domain')
    parser.add_argument('--create-splits', default=True, required=False, type=bool, help='Whether to create splits splits for each domain. Default: True')
    parser.add_argument('--alpha', required=False, type=float, help='value of alpha for dann. (default:0.25)')
    parser.add_argument('--alpha_exp', required=False, type=float, help='whether to use alpha exponential decaying as described in the dann paper')
    parser.add_argument('--dsbn', default=False, type=bool, help='run dsbn for domain adaptation')
    parser.add_argument('--mmd', default=False, type=bool, help='use mmd for domain adaptation')
    parser.add_argument('--contrastive', default=False, type=bool, help='use supervised and self supervised contrastive loss')
    parser.add_argument('--temp',required=False, type=float, default=0.07, help='temperature for loss function')

    args = parser.parse_args()
    
    # overwrite command line args here
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    args.seed = 0
    args.tsne = False # to be kept false. plotting internally handled
    args.tsne_freq = 1
    args.pretrained = 'resnet18_simclr_checkpoint_100.tar' # to use resnet18 simCLR pretrained on STL10
    args.epochs = 2
    args.batch_size = 64
    # args.resume = os.path.join(args.out, 'checkpoint_base.pth.tar') # only when need to resume training
    args.create_splits = True # if any domain arg is changed, then make True to create new splits
    args.train_split = 0.0 # if cross domain setup, then this is cross_domain train split(0.0) (as train domain uses full), else same domain train split
    # cross domain args
    args.train_domain = 'Product'
    args.test_domain = 'RealWorld'
    args.contrastive = False
    args.dann = True
    # args.alpha = 0.25
    args.alpha_exp = True
    # end args overwrite
    
    # set dataset specific parameters
    args.no_class = num_classes(args.dataset)
    
    print(' | '.join(f'{k}={v}' for k, v in vars(args).items()))

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
    return
    # load dataset
    args.no_known = args.no_class - int((args.novel_percent*args.no_class)/100)
    lbl_dataset, crossdom_dataset, test_dataset_known, test_dataset_novel, test_dataset_all = get_dataset(args)
    
    # return
    print(f'known classes: {args.no_known}/{args.no_class}, label %: {args.lbl_percent}%')
    # create dataloaders
    lbl_batchsize = args.batch_size
    args.iteration = (len(lbl_dataset)) // args.batch_size

    train_sampler = RandomSampler
    lbl_loader = DataLoader(lbl_dataset, sampler=train_sampler(lbl_dataset), batch_size=lbl_batchsize, num_workers=args.num_workers, drop_last=True)
    crossdom_loader = DataLoader(crossdom_dataset, sampler=train_sampler(crossdom_dataset), batch_size=lbl_batchsize, num_workers=args.num_workers, drop_last=True)
    test_loader_known = DataLoader(test_dataset_known, sampler=SequentialSampler(test_dataset_known), batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False)
    test_loader_novel = DataLoader(test_dataset_novel, sampler=SequentialSampler(test_dataset_novel), batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False)
    test_loader_all = DataLoader(test_dataset_all, sampler=SequentialSampler(test_dataset_all), batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False)

    # create model
    model = build_model(args, verbose=True)
    model = model.cuda()

    # optimizer
    if args.n_gpu > 1:
        optimizer = torch.optim.Adam(model.module.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


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
    
    # return
    ###################
    ##### Training ####
    ###################
    
    # wandb init
    wandb.init(entity='cv-exp',
               project='IITB-MBZUAI',
               name=f'{args.dataset} {args.train_domain}->{args.test_domain} {args.run_started}',
               config=args,
               group=f'{args.dataset}',
               tags=['our', args.dataset],
               allow_val_change=True)
    
    best_acc = 0
    test_accs = []
    model.zero_grad()
    for epoch in range(start_epoch, args.epochs):
        train_loss = train(args, lbl_loader, model, optimizer, epoch, crossdom_loader)
        test_acc_known = test_known(args, test_loader_known, model, epoch)
        # novel_cluster_results = test_cluster(args, test_loader_novel, model, epoch, offset=args.no_known)
        # all_cluster_results = test_cluster(args, test_loader_all, model, epoch)
        # test_acc = all_cluster_results["acc"]

        is_best = test_acc_known > best_acc
        best_acc = max(test_acc_known, best_acc)

        # cross_acc_known = test_known(args, cross_loader_known, model, epoch)
        if (epoch + 1) % args.tsne_freq == 0:
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

        wandb.log({'train/train_loss':train_loss}, commit=False)
        wandb.log({'test/acc-known':test_acc_known})
        # wandb.log({'test/cross-acc-known':cross_acc_known})
        # wandb.log({'test/acc-novel':novel_cluster_results["acc"]})
        # wandb.log({'test/nmi-novel':novel_cluster_results["nmi"]})
        # wandb.log({'test/acc-all':all_cluster_results["acc"]})
        # wandb.log({'test/nmi-all':all_cluster_results["nmi"]})

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


def train(args, lbl_loader, model, optimizer, epoch, crossdom_loader=None):

    eps = 1e-6
    model.train()
    lbl_batchsize = lbl_loader.batch_size
    
    all_losses = Losses()
    batch_time = AverageMeter()
    all_losses.add_loss('losses')
    all_losses.add_loss('losses_ce')

    if args.dsbn:
        all_losses.add_loss('losses_mse')
    elif args.dann:
        all_losses.add_loss('losses_dann_source')
        all_losses.add_loss('losses_dann_target')
    elif args.mmd:
        all_losses.add_loss('losses_mmd')
    elif args.contrastive:
        all_losses.add_loss('losses_supcon')
        all_losses.add_loss('losses_selfcon')

    end = time.time()

    train_loader = zip(lbl_loader, crossdom_loader) if crossdom_loader is not None else None
    if not args.no_progress:
        args.iteration = min(len(lbl_loader), len(crossdom_loader))
        p_bar = tqdm(range(args.iteration))

    for batch_idx, (data_lbl, data_crossdom) in enumerate(train_loader):
        inputs_l, targets_l, _ = data_lbl
        target_images, _ = data_crossdom
        if not args.contrastive:
            inputs_l = inputs_l.cuda()
            target_images = target_images.cuda()
        targets_l = targets_l.cuda()
        final_loss = 0
        bsz = targets_l.shape[0]
        
        if args.contrastive:
            # self supervised contrastive loss for target
            target_images = torch.cat([target_images[0], target_images[1]], dim=0)
            target_images = target_images.cuda()
            features, _ = model(target_images)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss_selfcon = SupConLoss(temperature=args.temp)(features)
            final_loss += loss_selfcon
            all_losses.update('losses_selfcon', loss_selfcon.item(), bsz)
            
            inputs_l = torch.cat([inputs_l[0], inputs_l[1]], dim=0)
            inputs_l = inputs_l.cuda()
        
        features, logits_l = model(inputs_l) if not args.dsbn else model(inputs_l, torch.zeros(inputs_l.shape[0], dtype=torch.long))
        
        if args.contrastive:
            # supervised contrastive loss on source images
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss_supcon = SupConLoss(temperature=args.temp)(features, targets_l)
            final_loss += loss_supcon
            all_losses.update('losses_supcon', loss_supcon.item(), bsz)
            
            # average CE on 2 transforms of source images
            logits_1, logits_2 = torch.split(logits_l, [bsz, bsz], dim=0)
            loss_ce_1 = F.cross_entropy(logits_1, targets_l)
            loss_ce_2 = F.cross_entropy(logits_2, targets_l)
            loss_ce = (loss_ce_1 + loss_ce_2) / 2
        else:
            loss_ce = F.cross_entropy(logits_l, targets_l)
        
        
        final_loss += loss_ce
        
        # adaptation method
        if args.dsbn:
            _, logits_crossdom = model(target_images, torch.ones(target_images.shape[0], dtype=torch.long))
            loss_mse = F.kl_div(logits_l + eps, logits_crossdom + eps)
            # loss_mse = F.mse_loss(logits_l, logits_crossdom)
            final_loss += loss_mse
            # print(loss_mse)
            # exit()
            all_losses.update('losses_mse', loss_mse.item(), inputs_l.size(0))

        elif args.dann:
            
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

            all_losses.update('losses_dann_source', loss_dann_source.item(), inputs_l.size(0))
            all_losses.update('losses_dann_target', loss_dann_target.item(), target_images.size(0))
        
        elif args.mmd:
            # probs_source = F.softmax(logits_l, dim=1)
            # _, logits_target = model(target_images)
            # probs_target = F.softmax(logits_target, dim=1)
            # loss_mmd = mmd_loss(probs_target, probs_source)
            # print(loss_mmd)
            # exit()
            # from pytorch_adapt.layers import MMDLoss
            # from pytorch_adapt.layers.utils import get_kernel_scales
            # kernel_scales = get_kernel_scales(low=-3, high=3, num_kernels=10)
            loss_fn = MMDLoss()
            loss_mmd = loss_fn(inputs_l, target_images)
            
            # print(loss_mmd)
            final_loss += loss_mmd
            all_losses.update('losses_mmd', loss_mmd.item(), target_images.size(0))

        all_losses.update('losses', final_loss.item(), inputs_l.size(0))
        all_losses.update('losses_ce', loss_ce.item(), inputs_l.size(0))

        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if not args.no_progress:
            losses_str = ' '.join(['{}: {:.4f}'.format(name, avg) for name, avg in all_losses.get_averages().items()])
            p_bar.set_description("train epoch: {}/{}. itr: {}/{}. btime: {:.3f}s. {}".format(
                epoch + 1, args.epochs, batch_idx + 1, args.iteration, batch_time.avg, losses_str))
            p_bar.update()

    if not args.no_progress:
        p_bar.close()

    loss_averages = all_losses.get_averages()
    for name, avg in loss_averages.items():
        if name != 'losses':
            wandb.log({f'train/{name}': avg}, commit=False)

    return all_losses['losses'].avg


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
            if args.dsbn:
                _, outputs = model(inputs, domain_label=torch.ones(inputs.shape[0], dtype=torch.long))
            else:
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

    wandb.log({'test/known_loss_ce':losses.avg}, commit=False)

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
            if args.dsbn:
                _, outputs = model(inputs, domain_label=torch.ones(inputs.shape[0], dtype=torch.long))
            else:
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
