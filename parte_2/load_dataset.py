from sklearn.model_selection import train_test_split
from torchvision import datasets,transforms
from torch.utils.data import DataLoader, Subset,random_split
# We need to transform our images, so we define a transformation method
def load_dataset(
    directory:str="data",
    batch_size:int=32,
    image_size:tuple[int,int]=(224,224),
    num_workers:int=4,
    mean:list[float]=[0.485,0.456,0.406],
    std:list[float]=[0.229,0.224,0.225],
    split_ratio:float=0.8
):
    #transformation
    transform=transforms.Compose(
    [transforms.Resize(image_size),
     transforms.ToTensor(),
     transforms.Normalize(mean=mean,std=std)
    ])
    #dataset loading
    dataset=datasets.ImageFolder(root=directory,transform=transform)
    print(len(dataset))
    targets=[s[1] for s in dataset.samples]
     
    train_size = int(split_ratio * len(dataset))  # 0.8 * 9000 = 7200
    val_size = len(dataset) - train_size         # 1800
    train_idx,val_idx=train_test_split(list(range(len(dataset))),
                                       test_size=1-split_ratio,
                                       stratify=targets,
                                       random_state=42)
    train_dataset=Subset(dataset,train_idx)
    val_dataset=Subset(dataset,val_idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, dataset.classes

