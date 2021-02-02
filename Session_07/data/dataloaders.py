import torch
from torchvision import datasets, transforms

train_transforms = transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                                      ])
test_transforms = transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                                    ])

train_dataset = datasets.CIFAR10(root='./data', train = True, transform = train_transforms, download = True)
test_dataset = datasets.CIFAR10(root='./data', train = False, transform = test_transforms, download = True)

cuda = torch.cuda.is_available()

#Arguments to be fed into dataloaders. 
dataloader_args = dict(shuffle=True, batch_size=100, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

##Train and Test dataloaders. 
train_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_args)
test_loader = torch.utils.data.DataLoader(test_dataset, **dataloader_args)

def dataloaders():
    return train_loader, test_loader