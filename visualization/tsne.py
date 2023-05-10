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
            if args.method == 'dsbn':
                feat, _ = model(inputs, domain_label=torch.ones(inputs.shape[0], dtype=torch.long))
            else:
                feat, _ = model(inputs)
            features.append(feat.cpu().numpy())
            labels.extend(targets.tolist())
    return features, labels


def generate_tsne(args, features, labels, label_nos, title):
    
    features = np.array(features, dtype=object)
    labels = np.array(labels, dtype=object)
    features = np.vstack(features)
    labels = np.vstack(labels)
    
    tsne = TSNE(n_jobs=16)

    embeddings = tsne.fit_transform(features)
    vis_x = embeddings[:, 0]
    vis_y = embeddings[:, 1]
    
    palette = sns.color_palette("Spectral", label_nos)

    sns.set(rc={'figure.figsize': args.figsize})

    fig, ax = plt.subplots()
    plot = sns.scatterplot(x=vis_x, y=vis_y, 
                           hue=labels[:,0], legend='full', palette=palette,
                           ax=ax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    plt.close()
    return fig


def plot(args, model):
    '''plot tsne given args and model'''

    args.figsize = (17,13)
    model = model.cuda()
    model.eval()
    
    # cross domain known class plot
    args.tsne = True # set to true when plotting with 2 colours for cross domain plot
    lbl_dataset, _, test_dataset_known, _, _ = get_dataset(args)
    
    features, labels = evaluate(args, lbl_dataset, model)
    features_target, labels_target = evaluate(args, test_dataset_known, model)
    features.extend(features_target)
    labels.extend(labels_target)

    fig_known_classes = generate_tsne(args, features, labels, 2, 'Cross Domain Known Class t-SNE')
    
    args.tsne = False # set to False for keeping class labels as is
    _, _, _, test_dataset_novel, test_dataset_all = get_dataset(args)
    fig_unknown_classes = None
    # target domain unknown class plot
    # only when there are unknown classes
    if args.no_class != args.no_known:
        features, labels = evaluate(args, test_dataset_novel, model)
        fig_unknown_classes = generate_tsne(args, features, labels, args.no_novel, 'Target Domain Unknown Class t-SNE')
    
    # target domain all class plot
    features, labels = evaluate(args, test_dataset_all, model)
    fig_all_classes = generate_tsne(args, features, labels, args.no_class, 'Target Domain All Class t-SNE')
    
    return fig_known_classes, fig_unknown_classes, fig_all_classes


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