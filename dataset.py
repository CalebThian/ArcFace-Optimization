from torch.utils.data import DataLoader,RandomSampler,Subset
from torchvision.datasets import ImageFolder
from config import Config as conf
import numpy as np

def load_data(conf, training=True):
    if training:
        dataroot = conf.train_root
        transform = conf.train_transform
        batch_size = conf.train_sample_batch_size
        train_sample_rate = conf.train_sample_rate
    else:
        dataroot = conf.test_root
        transform = conf.test_transform
        batch_size = conf.test_batch_size
    data = ImageFolder(dataroot, transform=transform)
    class_num = len(data.classes)
    loader = DataLoader(data, batch_size = batch_size, shuffle=False, 
        pin_memory=conf.pin_memory, num_workers=conf.num_workers)
    if training:
        it = iter(loader)
        Labels = []
        while True:
            try:
                _, labels = it.next()
                if len(labels) <= 0 :
                    break
                Labels.extend(list(labels.numpy()))
                print("\r%d"%Labels[-1],end="")
            except:
                break
        #indices = {cls: np.random.choice(np.where(np.array(Labels) == cls)[0],
        #                                 int(len(np.where(np.array(Labels) == cls)[0])*conf.train_sample_rate),replace = False) for cls in range(Labels[-1]+1)}
        values,counts = np.unique(np.array(Labels),return_counts = True)
        indices = dict()
        total_imgs = 0
        for i,cls in enumerate(values):
            if i==0:
                start = 0
            else:
                start = counts[i-1]
            end = counts[i]
            end = int(start + (end-start)*train_sample_rate)
            indices[cls] = np.arange(start,end)
            total_imgs += len(indices[cls])
        train = Subset(data, indices=[i for v in indices.values() for i in v])
        sampler = RandomSampler(train, replacement=True, num_samples=256)
        loader = DataLoader(train, batch_size=conf.train_batch_size,
                            pin_memory=conf.pin_memory, num_workers=conf.num_workers, sampler=sampler)
        print("Total train images: %d"%total_imgs)
    return loader, class_num


'''
loader = DataLoader(image_datasets['train'], batch_size=2000, shuffle=False, num_workers=2)
images, labels = iter(loader).next()
print(len(labels))
print
indices = {cls: np.random.choice(np.where(labels.numpy() == cls)[0], shots, replace=False) for cls in range(219)}
train_5shot = torch.utils.data.Subset(image_datasets['train'], indices=[i for v in indices.values() for i in v])
len(train_5shot)
sampler = torch.utils.data.RandomSampler(train_5shot, replacement=True, num_samples=256)
loader_train = torch.utils.data.DataLoader(train_5shot, batch_size=10, num_workers=2, sampler=sampler)
'''