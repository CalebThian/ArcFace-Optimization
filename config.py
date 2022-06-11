import torch
import torchvision.transforms as T

class Config:
    # dataset
    train_root = './/data//CASIA-WebFace//'
    test_root = ".//data//lfw//"
    test_list = ".//data//lfw_test_pair.txt"
    
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
    
    # Training Parameters setting
    train_batch_size = 64
    test_batch_size = 60
    test_model = "checkpoints/24.pth"
    
    pin_memory = True  # if memory is large, set it True for speed
    num_workers = 4  # dataloader
    
    # Model Parameters setting
    backbone = 'fmobile' # [resnet, fmobile]
    metric = 'arcface'  # [cosface, arcface]
    embedding_size = 512
    drop_ratio = 0.5
    
    epoch = 30
    optimizer = 'sgd'  # ['sgd', 'adam']
    lr = 1e-1
    lr_step = 10
    lr_decay = 0.95
    weight_decay = 5e-4
    loss = 'focal_loss' # ['focal_loss', 'cross_entropy']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoints = "checkpoints"