import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
import segmentation_models_pytorch as seg_models
from loader.loader_camvid import camvidLoader 
from torch.utils.data import DataLoader
import albumentations as albu
import pdb

def to_tensor(x, **kwargs):
    #TODO: albumentations lib requires separate 'to_tensor' function
    #       for 'image' and 'mask'.
    return x.transpose(2,0,1).astype('float32')

if __name__ == '__main__':
    
    # define device
    device = torch.device('cuda:0')
    
    # build model
    pspnet = seg_models.PSPNet(classes=12).to(device)
    print("Created a model")
    
    # preprocessing
    preprocessing_fn = seg_models.encoders.get_preprocessing_fn('resnet34',
                                                                'imagenet')
    preprocessor = albu.Compose([albu.Lambda(image=preprocessing_fn),
                                albu.Lambda(image=to_tensor)])
    
    # load training, validation data
    train_file_paths = '/media/sjang/EC9EAA6D9EAA2FCE/data/camvid/CamVid/train.txt'
    val_file_paths = '/media/sjang/EC9EAA6D9EAA2FCE/data/camvid/CamVid/val.txt'
    
    train_data = camvidLoader(train_file_paths, preprocessing=preprocessor)
    val_data = camvidLoader(val_file_paths, preprocessing=preprocessor)
    
    train_loader = DataLoader(train_data, batch_size=2,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=1,
                            shuffle=False, num_workers=1)
    
    # define loss
    loss_fn = nn.CrossEntropyLoss() 
    #loss_fn = cross_entropy2d  
    
    # define optimizer
    optimizer = optim.SGD(pspnet.parameters(), lr=0.001, momentum=0.9)
    
    # metrics
    metrics = [seg_models.utils.metrics.IoUMetric(eps=1.),
               seg_models.utils.metrics.FscoreMetric(eps=1.)]
    
    ## define epochs
    #train_epoch = seg_models.utils.train.TrainEpoch(
    #              pspnet, loss=loss, metrics=metrics,
    #              optimizer=optimizer, device=device,
    #              verbose=True)
    
    #val_epoch = seg_models.utils.train.ValidEpoch(
    #              pspnet, loss=loss, metrics=metrics,
    #              device=device,
    #              verbose=True)
    
    # run training
    max_score = 0
    n_epochs = 50
    for epoch in range(0, n_epochs):
        print("epochs:{}/{}".format(epoch+1, n_epochs))
        
        step = 0
        for images, labels in train_loader:
            step += 1

            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            predicts = pspnet(images)
            loss = loss_fn(predicts, labels)
            loss.backward()
            optimizer.step()
            
            print("step:{}, loss={}".format(step, loss)) 
