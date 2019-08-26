import torch
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import pdb

class camvidLoader(Dataset):
 
    CLASSES = ['Sky', 'Building', 'Pole', 'Road', 'Pavement', 
                   'Tree', 'SignSymbol', 'Fence', 'Car', 
                   'Pedestrian', 'Bicyclist', 'Bridge']
    
    CLASS2COLOR={
        "Animal" : [64, 128, 64]    ,   
        "Archway" : [192, 0, 128]  , 
        "Bicyclist" : [0, 128, 192]  , 
        "Bridge" : [0, 128, 64 ]  , 
        "Building" : [128, 0, 0]    , 
        "Car" : [64, 0, 128]   , 
        "CartLuggagePram" : [64, 0, 192]   , 
        "Child" : [192, 128, 64] , 
        "Column_Pole" : [192, 192, 128],
        "Fence" : [64, 64, 128   ],
        "LaneMkgsDriv" : [128, 0, 192 ] , 
        "LaneMkgsNonDriv" : [192, 0, 64  ] , 
        "Misc_Text" : [128, 128, 64 ], 
        "MotorcycleScooter" : [192, 0, 192  ], 
        "OtherMoving" : [128, 64, 64  ], 
        "ParkingBlock" : [64, 192, 128  ], 
        "Pedestrian" : [64, 64, 0     ],
        "Road" : [128, 64, 128  ],
        "RoadShoulder" : [128, 128, 192 ],
        "Sidewalk" : [0, 0, 192     ],
        "SignSymbol" : [192, 128, 128 ],
        "Sky" : [128, 128, 128 ],
        "SUVPickupTruck" : [64, 128, 192  ],
        "TrafficCone" : [0, 0, 64      ],
        "TrafficLight" : [0, 64, 64     ],
        "Train" : [192, 64, 128  ],
        "Tree" : [128, 128, 0   ],
        "Truck_Bus" : [192, 128, 192 ],
        "Tunnel" : [64, 0, 64     ],
        "VegetationMisc" : [192, 192, 0   ],
        "Void" : [0, 0, 0       ], 
        "Wall" : [64, 192, 0    ]}

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
          classes = self.CLASS2COLOR
        #self.class_ids = [self.CLASSES.index(cls) for cls in classes] 
        self.n_classes = len(self.CLASS2COLOR)
        idx = 0
        self.id2color={}
        self.id2class = {}
        for cls in classes:
            self.id2class[idx] = cls
            self.id2color[idx] = classes[cls]
            idx += 1
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
            sample = self.augmentation(image=image, mask=label)
            image, label = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=label)
            image, label = sample['image'], sample['mask']

        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        #image, label = self.to_tensor(image, label)
        return image, label
    
    def __len__(self):
    	return len(self.image_files)
    
    def to_tensor(self, image, label):
    	image = image.transpose(2,0,1)
    	image = torch.from_numpy(image).float()
    	label = torch.from_numpy(label).long()
    	return image, label

    def id_to_color(self, label):

        r = label.copy()
        g = label.copy()
        b = label.copy()

        for idx in range(0, self.n_classes):
            r[label==idx] = self.id2color[idx][0]
            g[label==idx] = self.id2color[idx][1]
            b[label==idx] = self.id2color[idx][2]

        rgb = np.zeros((label.shape[0], label.shape[1], 3))
        rgb[:,:,0] = r
        rgb[:,:,1] = g
        rgb[:,:,2] = b
        return rgb
            
