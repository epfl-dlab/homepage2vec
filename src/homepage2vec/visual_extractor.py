import torch 
import torchvision.models as models
from torchvision import datasets, transforms
from torch import nn
from PIL import Image

###################### CVM ######################

out_dim = 14 # number of classes
features_dim = 512 # number of features before the classifier

class ResNetPretrained(nn.Module):
    def __init__(self):
        super(ResNetPretrained, self).__init__()
        
        resnet = models.resnet18(pretrained=True)
        
        self.features = torch.nn.Sequential(*list(resnet.children())[:-1])       
        self.fc1 = torch.nn.Linear(features_dim, out_dim)


    def forward(self, x):
        x = self.features(x).reshape(-1, features_dim)

        x = self.fc1(x)
        return x
    
    
###################### FEATURES EXTRACTION ######################


class VisualExtractor:
    """
    Extract visual features from the screenshot of a webpage
    """
    
    def __init__(self, device='cpu'):
        self.model = ResNetPretrained().to(device)
        self.device = device
        self.batch_size = 128
       
    
    def get_features(self, screenshot_path):
        
        # dimension of the images
        valid_xdim = 640 
        valid_ydim = 360 

        # factor for 5-crop transform
        crop_factor = 0.6 

        crop_dim = [int(crop_factor * valid_ydim), int(crop_factor * valid_xdim)]

        # 5-crop transform
        five_crop = transforms.FiveCrop(size=crop_dim) 

        # image to tensor transform
        tensorize = transforms.ToTensor() 

        # normalization transform
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 

        # to apply to each crop
        stack_norm_tensorize = transforms.Lambda(
            lambda crops: torch.stack([normalize(tensorize(crop)) for crop in crops]))

        # final transform: crop > tensorize > normalize
        data_transforms = transforms.Compose([five_crop, stack_norm_tensorize])

        img = Image.open(screenshot_path)
        img_tensor = data_transforms(img)

        self.model.eval()
        with torch.no_grad():
            inputs = img_tensor.to(self.device)
            ncrops, c, h, w = inputs.size()
            outputs = self.model.features(inputs.view(-1, c, h, w)) # output for each crop
            outputs = outputs.view(1, ncrops, -1).mean(1) # mean over the crops
        return outputs.detach()[0]

