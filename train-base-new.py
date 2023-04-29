import numpy as np
import argparser
import os
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from models.build_model import build_model, modify_state_dict
from utils.utils import Losses, AverageMeter, accuracy, set_seed, save_checkpoint, describe_splits, describe_image_dataset
from datasets.datasets import get_dataset
from datasets.multi_domain import create_dataset
from utils.evaluate_utils import hungarian_evaluate
from losses import MMDLoss, SupConLoss
import wandb
from visualization.tsne import plot
from torchsummary import summary

def setup():
    """create directories, data splits, print and get config. returns args"""
    args = argparser.get_args()
    
    os.makedirs(args.data_root, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    if args.create_splits:
        create_dataset(args)
    describe_splits(args, ignore_errors=True)
    
    # print config
    print(' | '.join(f'{k}={v}' for k, v in vars(args).items()))
    with open(f'{args.out}/score_logger.txt', 'a+') as ofile:
        ofile.write('************************************************************************\n\n')
        ofile.write(' | '.join(f'{k}={v}' for k, v in vars(args).items()))
        ofile.write('\n\n************************************************************************\n')
    
    return args

def main():

    args = setup()
    
    lbl_dataset, crossdom_dataset, test_dataset_known, test_dataset_novel, test_dataset_all = get_dataset(args)
    
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

    # return
    start_epoch = 0
    if args.resume:
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        print(f'resuming from checkpoint..\n')
        args.out = os.path.dirname(args.resume) # set output directory same as resume directory
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        # from collections import OrderedDict
        load_state_dict = checkpoint['state_dict']
        if args.n_gpu > 1:
            load_state_dict = modify_state_dict(load_state_dict, 'add_prefix', 'module.')
        model.load_state_dict(load_state_dict, strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # return
    ###################
    ##### Training ####
    ###################

    # wandb init
    wandb.init(entity='cv-exp',
               project='Cross-Domain-GCD',
               name=f'{args.dataset} {args.train_domain}->{args.test_domain} {args.run_started}',
               config=args,
               id=None,
               resume='allow',
               group=f'{args.dataset}',
               tags=['our', args.dataset],
               allow_val_change=True)
    
    best_acc = 0
    test_accs = []
    model.zero_grad()
    for epoch in range(start_epoch, args.epochs):
        train_loss = train(args, lbl_loader, model, optimizer, epoch, crossdom_loader)
        test_acc_known = test_known(args, test_loader_known, model, epoch)
        novel_cluster_results = test_cluster(args, test_loader_novel, model, epoch, offset=args.no_known)
        all_cluster_results = test_cluster(args, test_loader_all, model, epoch)
        test_acc = all_cluster_results["acc"]

        # currently using best acc as acc of known classes
        is_best = test_acc_known > best_acc
        best_acc = max(test_acc_known, best_acc)

        if (epoch + 1) % args.tsne_freq == 0:
            fig_known, fig_unknown, fig_all = plot(args, model)
            wandb.log({"tsne-known": wandb.Image(fig_known)}, commit=False)
            wandb.log({"tsne-unknown": wandb.Image(fig_unknown)}, commit=False)
            wandb.log({"tsne-all": wandb.Image(fig_all)}, commit=False)
            args.tsne = False # reset to false
        
        print(f'epoch: {epoch}, acc-known: {test_acc_known}')
        print(f'epoch: {epoch}, acc-novel: {novel_cluster_results["acc"]}, nmi-novel: {novel_cluster_results["nmi"]}')
        print(f'epoch: {epoch}, acc-all: {all_cluster_results["acc"]}, nmi-all: {all_cluster_results["nmi"]}, best-acc: {best_acc}')

        wandb.log({'train/train_loss':train_loss}, commit=False)
        wandb.log({'test/acc-known':test_acc_known}, commit=False)
        wandb.log({'test/acc-novel':novel_cluster_results["acc"]}, commit=False)
        wandb.log({'test/nmi-novel':novel_cluster_results["nmi"]}, commit=False)
        wandb.log({'test/acc-all':all_cluster_results["acc"]}, commit=False)
        wandb.log({'test/nmi-all':all_cluster_results["nmi"]})

        model_to_save = model.module if hasattr(model, "module") else model
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model_to_save.state_dict(),
            'acc': test_acc_known,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.out, tag='base')

        test_accs.append(test_acc_known)

        with open(f'{args.out}/score_logger.txt', 'a+') as ofile:
            ofile.write(f'epoch: {epoch}, acc-known: {test_acc_known}\n')
            ofile.write(f'epoch: {epoch}, acc-novel: {novel_cluster_results["acc"]}, nmi-novel: {novel_cluster_results["nmi"]}\n')
            ofile.write(f'epoch: {epoch}, acc-all: {all_cluster_results["acc"]}, nmi-all: {all_cluster_results["nmi"]}, best-acc: {best_acc}\n')

    # wandb.save(f'{args.out}/score_logger.txt')

def train(args, lbl_loader, model, optimizer, epoch, crossdom_loader=None):

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
