from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_dataset(path):

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
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