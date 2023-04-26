import torch
import numpy as np
import argparser
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from models.build_model import build_model
from datasets.datasets import get_dataset


def get_dataloader(args, dataset):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


def evaluate(args, dataset, model):
    dataloader = get_dataloader(args, dataset)
    # print(next(iter(dataloader)))
    features = []
    labels = []
    with torch.inference_mode():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.cuda()
            if args.dsbn:
                feat, _ = model(inputs, domain_label=torch.ones(inputs.shape[0], dtype=torch.long))
            else:
                feat, _ = model(inputs)
            features.append(feat.cpu().numpy())
            labels.extend(targets.tolist())
    return features, labels


def plot(args, model, tsne=True):
    '''plot tsne given args and model'''
    args.figsize = (17,13)
    
    args.tsne = tsne
    lbl_dataset, _, test_dataset_known, _, _ = get_dataset(args)
    
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
    
    if args.train_domain == args.test_domain:
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

    args = argparser.get_args()
    # overwrite command line args here
    args.plot_name = 'cross_domain' # override name
    
    model = build_model(args)
    
    if args.resume:
        assert os.path.isfile(
            args.resume), f"Error: no checkpoint directory: {args.resume} found!"
        print(f'loaded best model checkpoint!')
        args.out = os.path.dirname(args.resume) # set output directory same as resume directory
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        print(f'model accuracy: {best_acc}')
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    
    fig = plot(args, model)
    path = os.path.join(args.out, args.plot_name)
    fig.savefig(path)
    
if __name__ == '__main__':
    main()