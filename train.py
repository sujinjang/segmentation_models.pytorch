import torch
import segmentation_models_pytorch as seg_models

if __name__ == '__main__':
  pspnet = seg_models.PSPNet()
  print("Created a model")
