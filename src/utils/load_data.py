import numpy as np
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
import cv2
import torch
import random
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from PIL import Image, ImageDraw, ImageFont
from .args import args
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import pythainlp
import monai
from monai.utils.misc import set_determinism
from src.utils import *
from monai.data.utils import list_data_collate, worker_init_fn


def show_batch(dl):
    for images, labels in dl:
        grid = make_grid((images.detach()[:]), nrow=4)
        img = transforms.ToPILImage()(grid)
        break
    
    return img


def list_of_coordinates(size, n_lines, n_columns):
  coordinates = []
  for i in range(n_columns):
        for j in range(n_lines):
            x = 5+(j*((size//(n_lines))))
            y = 1+(i*((size//n_columns)))
            coordinates.append([x,y])
  return coordinates


class CellDropout(object):
  def __init__(self, num_features, coordinates, h, w, prob, seed):
    self.coordinates = coordinates
    self.h = h
    self.w = w
    self.prob = prob
    self.num_features = num_features
    self.seed = seed
  
  def __call__(self, sample):
    indexes = random.sample(self.coordinates, int(self.num_features*self.prob))
    for i in range(len(indexes)):
        sample = TF.erase(sample, i=indexes[i][0]+1, j=indexes[i][1]+1, h=self.h, w=self.w, v=0)
    return sample
  

# ----- Data to Image Transformer -----
def data2img(arr, n_lines, n_columns, size, font_size=args.font_size, font=cv2.FONT_HERSHEY_SIMPLEX):
   
    resolution=(size, size)
    x, y = resolution
    n_features = len(arr)
    frame = np.ones(resolution, np.uint8)*0
    k = 0
    
    for i in range(n_columns):
        for j in range(n_lines):
            try:
                cv2.putText(
                    frame, str(arr[k]), ((1+i*((x//n_columns))), 10+(j+1)*((y//(n_lines+1)))),
                    fontFace=font, fontScale=font_size, color=(255), thickness=1)
                k += 1
            except IndexError:
                break

    return np.array(frame, np.uint8) #np.array(frame, np.uint8)



# ----- Dataset -----
class TabularDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):
        x = self.data[0][index]
        y = self.data[1][index]

        return x, y


class CustomTensorDataset(Dataset):
    def __init__(self, data, rows, columns, size, to_tensor= transforms.ToTensor(), transform=None, img_type=args.img_type, seed=1):
        self.data = data
        self.transform = transform
        self.counter = 0
        self.img_type = img_type
        self.rows = rows
        self.columns = columns
        self.size = size
        self.seed=seed
        self.to_tensor = to_tensor

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):
        x = self.data[0][index]
        # if we tranform the data into images
        if self.img_type == 'digits':
            img_arr = data2img(x, self.rows, self.columns, self.size)

            img = Image.fromarray(img_arr)
            x = self.to_tensor(img)

        if self.transform:
            x = self.transform(x)
            

        y = self.data[1][index] ### problem with int64 numpy --> solved
        return x, y

def load_tabular(seed, dataset=args.dataset,  batch_size=args.batch_size,
              precision=args.precision, device=args.device, img_type = args.img_type):
    label_dict = {
        # Controls
        'n': 0,
        # Chirrhosis
        'cirrhosis': 1,
        # Colorectal Cancer
        'cancer': 1, 'small_adenoma': 0,
        # IBD
        'ibd_ulcerative_colitis': 1, 'ibd_crohn_disease': 1,
        # T2D and WT2D
        't2d': 1,
        # Obesity
        'leaness': 0, 'obesity': 1,
        }
    
    # read file
    if dataset == 'CT2D':
        name = 'T2D'
    else:
        name = args.dataset
   
    feature_string = "k__"
    label_string="disease"

    raw = pd.read_csv('data/abundance/abundance_' + name + '.txt', sep='\t', index_col=0, header=None)  

    # select rows having feature index identifier string
    X = raw.loc[raw.index.str.contains(feature_string, regex=False)].T
    X = X.values.astype(np.float32)

    # get class labels
    Y = raw.loc[label_string] #'disease'
    Y = Y.replace(label_dict)

    X_train, X_test, y_train, y_test = train_test_split(X, Y.values.astype(np.int64), test_size=0.2, random_state=seed, stratify=Y.values)
    
    return X_train, X_test, y_train, y_test


def load_data(seed, dataset=args.dataset,  batch_size=args.batch_size,
              precision=args.precision, device=args.device, img_type = args.img_type):
    
    if img_type == 'digits':

        label_dict = {
        # Controls
        'n': 0,
        # Chirrhosis
        'cirrhosis': 1,
        # Colorectal Cancer
        'cancer': 1, 'small_adenoma': 0,
        # IBD
        'ibd_ulcerative_colitis': 1, 'ibd_crohn_disease': 1,
        # T2D and WT2D
        't2d': 1,
        # Obesity
        'leaness': 0, 'obesity': 1,
        }

        feature_string = "k__"
        label_string="disease"

        # read file
        if dataset == 'IBD':
            name, rows, columns, size = 'IBD', 37, 12, 450
        elif dataset == 'CT2D':
            name, rows, columns, size = 'T2D', 42, 14, 450
        elif dataset == 'Cirrhosis':
            name, rows, columns, size = 'Cirrhosis', 39, 14, 450
        elif dataset == 'Obesity':
            name, rows, columns, size = 'Obesity', 36, 13, 450
        elif dataset == 'Colorectal':
            name, rows, columns, size = 'Colorectal', 36, 14, 450
        elif dataset == 'WT2D':
            name, rows, columns, size = 'WT2D', 35, 11, 450

        raw = pd.read_csv('data/abundance/abundance_' + name + '.txt', sep='\t', index_col=0, header=None)  

        # select rows having feature index identifier string
        X = raw.loc[raw.index.str.contains(feature_string, regex=False)].T
        X = X.values.astype(np.float32)


        # get class labels
        Y = raw.loc[label_string] #'disease'
        Y = Y.replace(label_dict)

        X_train, X_test, y_train, y_test = train_test_split(X, Y.values.astype(np.int64), test_size=0.2, random_state=seed, stratify=Y.values)
        if precision != 5:
            X_train = X_train.round(precision)
        else: 
            pass


        num_features = X_train.shape[1]
        coordinates = list_of_coordinates(size, rows, columns)
        h, w = (size//rows), size//columns

        # percentage of feature dropped out
        prob = 0.3

        if args.aug == 'RandomErasing':
            transform = transforms.RandomErasing(p=0.5, scale=(0.1, 0.12), ratio=(0.8, 1), value=0, inplace=False)

        elif args.aug == 'CellDropout':
            transform = transforms.RandomApply([CellDropout(num_features=num_features, coordinates=coordinates, h=h, w=w, prob=prob, seed=seed)], p=0.5)
            
        elif args.aug == 'RandGauss':
            transform = monai.transforms.RandGaussianSmooth(sigma_x=(0.25, 1), sigma_y=(0.25, 1), prob=0.5, approx='erf')
            
        elif args.aug == 'RandCoarseDrop':
            transform = monai.transforms.RandCoarseDropout(holes=int(num_features*prob), spatial_size=(20,20), dropout_holes=True, fill_value=0,
                                                    max_holes=None, max_spatial_size=None, prob=0.5)
            
        elif args.aug == 'RandCoarseShuffle':
            transform = monai.transforms.RandCoarseShuffle(holes=int(num_features*prob), spatial_size=(20,20), prob=0.5)
            
        elif args.aug == 'RandBiasField':
            transform = monai.transforms.RandBiasField(degree=3, coeff_range=(0.0, 0.1), prob=0.5)

        elif args.aug == 'RandElastic':
            transform = monai.transforms.Rand2DElastic(spacing=(15,15), magnitude_range=(0,0.3), mode="bilinear", prob=0.5)

        elif args.aug == 'RandZoom':
            transform = monai.transforms.RandZoom(prob=0.5, min_zoom=1.1, max_zoom=1.7)

        elif args.aug == 'RandFlip':
            transform = monai.transforms.RandFlip(prob=0.5)

        elif args.aug == 'RandRotate':
            transform = monai.transforms.RandRotate(prob=0.5, range_x=[-0.4, 0.4], mode="bilinear")

        else:
            transform=None
            

        return (X_train, y_train), (X_test, y_test), (rows, columns, size), transform
    
    # baseline with MLP and tabular data
    else:
        X_train, X_test, y_train, y_test = load_tabular(seed=seed)

                                                     #no meaning: just to keep the structure of the code 
        return (X_train, y_train), (X_test, y_test), (10, 10, 10), None

if __name__ == "__main__":
    pass

