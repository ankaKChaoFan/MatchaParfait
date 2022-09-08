import PIL
import torch
import torchvision.transforms as T
import json


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        with open(f'{data_dir}/dataset.json') as f:
            self.data = json.load(f)
        self.length = len(self.data['key_point'])
        
        self.trans = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5), (0.5), inplace=True),
        ])
    def __len__(self):
        return self.length
    def __getitem__(self,idx):
        img_dir = f'{self.data_dir}/{self.data["image_dir"][idx]}'
        return self.trans(PIL.Image.open( img_dir )), torch.tensor( self.data['key_point'][idx] )

    
def param_ema(model_a, model_b, m=0.9):
    for p_a,p_b in zip(model_a.parameters(), model_b.parameters()):
        p_a.data = m*p_a.data + (1-m)*p_b.data
    for p_a,p_b in zip(model_a.buffers(), model_b.buffers()):
        p_a.data = m*p_a.data + (1-m)*p_b.data
        
        