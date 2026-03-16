from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloader(path, batch_size):

    transform = transforms.Compose([
        transforms.Resize((224,2224)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(path, transform=transform)

    loader = DataLoader(dataset,
                        batch_size= batch_size,
                        shuffle=True)
    
    return loader