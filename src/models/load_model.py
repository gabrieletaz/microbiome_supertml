import torch.nn as nn
from torchvision import models
import torch
from src.utils import args


# ----- Model selection -----
class CNN_opt(nn.Module):
    def __init__(self, params):
        super(CNN_opt, self).__init__()
        self.layers = []
        out_dim = params['n_kernels']
        self.layers.append(nn.Conv2d(1, out_dim, kernel_size=3, stride=1, padding=1)) 
        self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.BatchNorm2d(out_dim))
        self.layers.append(nn.MaxPool2d(2, 2))
        for i in range(params['n_layers']):
            self.layers.append(nn.Conv2d(out_dim, out_dim*2, kernel_size=3, stride=1, padding=1))
            out_dim = out_dim*2
            self.layers.append(nn.LeakyReLU())
            self.layers.append(nn.BatchNorm2d(out_dim))
            self.layers.append(nn.MaxPool2d(2, 2))

        
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Dropout(params['dropout']))
        img_dim = int(450/(2**(1+params['n_layers'])))
        self.out = nn.Sequential(*self.layers)
        self.reg = nn.Linear(out_dim*img_dim*img_dim, 2)

    def forward(self, x):
        return self.reg((self.out(x)))
    

class MLP_opt(nn.Module):
    def __init__(self, params, input_dim):
        super(MLP_opt, self).__init__()
        self.layers = [] 
        out_dim = params['input_dim']
        for i in range(params['n_layers']):
            self.layers.append(nn.Linear(input_dim, out_dim))
            self.layers.append(nn.LeakyReLU())
            self.layers.append(nn.BatchNorm1d(out_dim))
            input_dim = out_dim
            out_dim = out_dim // 2

        self.layers.append(nn.Dropout(params['dropout']))
        self.out = nn.Sequential(*self.layers)
        self.reg = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.reg((self.out(x)))




def load_model(params,input_dim, model_name=args.model):

    # Architecture to use
    if model_name == 'CNN_opt':
        model = CNN_opt(params)
    elif model_name == 'MLP_opt':
        model = MLP_opt(params, input_dim)

    model = model.to(args.device)
    return model

if __name__ == "__main__":
    pass

