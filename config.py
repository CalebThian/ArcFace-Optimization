import torch
import torchvision.transforms as T

class Config:
    # dataset
    train_root = '/data/CASIA-WebFace'
    test_root = "/data/lfw"
    test_list = "/data/lfw_test_pair.txt"
    
    ## Data Preprocessing
    input_shape = [1, 128, 128]
    train_transform = T.Compose([
        T.Grayscale(),
        T.RandomHorizontalFlip(),
        T.Resize((144, 144)),
        T.RandomCrop(input_shape[1:]),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])
    test_transform = T.Compose([
        T.Grayscale(),
        T.Resize(input_shape[1:]),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])