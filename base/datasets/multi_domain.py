import numpy as np
from PIL import Image, ImageFilter, ImageOps
import random
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader, Dataset
from skimage import io
import pickle
import os
import math

pacs_mean, pacs_std = (0.7659, 0.7463, 0.7173), (0.3089, 0.3181, 0.3470)

# Dataset sources: https://blog.csdn.net/qq_43827595/article/details/121345640
# https://people.cs.pitt.edu/~mzhang/cs1699/hw4.html

class PACSDataset(Dataset):

	def __init__(self,
			 root_dir,
			 label_type='domain',
			 is_training=False,
			 transform=None):
		self.root_dir = os.path.join(root_dir, 'train' if is_training else 'val')
		self.label_type = label_type
		self.is_training = is_training
		if transform:
			self.transform = transform
		else:
			self.transform = transforms.Compose([
					transforms.ToTensor(),
					transforms.Normalize(mean=[0.7659, 0.7463, 0.7173],
															 std=[0.3089, 0.3181, 0.3470]),
			])

		self.dataset, self.label_list = self.initialize_dataset()
		self.label_to_id = {x: i for i, x in enumerate(self.label_list)}
		self.id_to_label = {i: x for i, x in enumerate(self.label_list)}

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		image, label = self.dataset[idx]
		label_id = self.label_to_id[label]
		image = self.transform(image)
		return image, label_id

	def initialize_dataset(self):
		assert os.path.isdir(self.root_dir), \
				'`root_dir` is not found at %s' % self.root_dir

		dataset = []
		domain_set = set()
		category_set = set()
		cnt = 0

		for root, dirs, files in os.walk(self.root_dir, topdown=True):
			if files:
				_, domain, category = root.rsplit('/', maxsplit=2)
				domain_set.add(domain)
				category_set.add(category)
				# pbar = tqdm(files)
				for name in files:
					# pbar.set_description('Processing Folder: domain=%s, category=%s' %
					# 										 (domain, category))
					img_array = io.imread(os.path.join(root, name))
					dataset.append((img_array, domain, category))

		images, domains, categories = zip(*dataset)

		if self.label_type == 'domain':
			labels = sorted(domain_set)
			dataset = list(zip(images, domains))
		elif self.label_type == 'category':
			labels = sorted(category_set)
			dataset = list(zip(images, categories))
		else:
			raise ValueError(
					'Unknown `label_type`: Expecting `domain` or `category`.')

		return dataset, labels



if __name__ == '__main__':
    
    base_path = '/home/biplab/Mainak/'
    datasets = os.path.join(base_path, 'datasets')
    openldn_dir = os.path.join(base_path, 'CrossDomainNCD', 'OpenLDN')
    create_dir = os.path.join(openldn_dir, 'data', 'officehome')
    
    os.makedirs(create_dir, exist_ok=True)
    
    print(os.listdir(datasets))
    