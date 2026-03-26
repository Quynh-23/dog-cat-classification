from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_dataset(path):

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(path, transform=transform)

    return dataset


def get_loader(dataset, indices, batch_size):

    from torch.utils.data import Subset

    subset = Subset(dataset, indices)

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True
    )

    return loader