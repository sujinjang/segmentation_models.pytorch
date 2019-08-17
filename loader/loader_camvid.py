import torch
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import pdb

class camvidLoader(Dataset):
 
	CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 
	               'tree', 'signsymbol', 'fence', 'car', 
	               'pedestrian', 'bicyclist', 'unlabelled']
		
	def __init__(self, file_paths,	classes=None, augmentation=None,
								preprocessing=None):
	    
	  # read file_paths.txt
	  try:
	    
	    f_paths = open(file_paths)
	    pairs = f_paths.readlines()
	  except:
	    raise ValueError("Check files including input/label paths: {}".format(
	                    file_paths))
	  
	  image_paths, label_paths = [], []
	  for pair in pairs:
	    pair = pair.split(' ')
	    img_path = pair[0].strip()
	    label_path = pair[1].strip()
	    if img_path and label_path:
	      image_paths.append(img_path)
	      label_paths.append(label_path)
	  
	  self.image_files = image_paths
	  self.label_files = label_paths
	  if not classes:
	    classes = self.CLASSES
	  self.class_ids = [self.CLASSES.index(cls.lower()) for cls in classes] 
	  self.augmentation = augmentation
	  self.preprocessing = preprocessing
	
	def __getitem__(self, idx):
		
	      # read data	
	      image = cv2.imread(self.image_files[idx])
	      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	      label = cv2.imread(self.label_files[idx], 0)

	      # extract target classes from label
	      #labels = [(label==v) for v in self.class_ids]
	      #label = np.stack(labels, axis=-1).astype('float')
	      
	      # apply augmentation
	      if self.augmentation:
	      	sample = self.augmentation(image=image, label=label)
	      	image, label = sample['image'], sample['label']
	      
	      # apply preprocessing
	      if self.preprocessing:
	      	sample = self.preprocessing(image=image, label=label)
	      	image, label = sample['image'], sample['label']
	      image, label = self.to_tensor(image, label)
	      return image, label
	
	def __len__(self):
		return len(self.image_files)
	
	def to_tensor(self, image, label):
		image = image.transpose(2,0,1)
		image = torch.from_numpy(image).float()
		label = torch.from_numpy(label).long()
		return image, label
