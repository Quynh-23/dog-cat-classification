import kagglehub
import os

def prepare_dataset():

    dataset_dir = "data/PetImages"

    if os.path.exists(dataset_dir):
        print("Dataset already exists.")
        return dataset_dir

    print("Downloading dataset...")

    path = kagglehub.dataset_download(
        "shaunthesheep/microsoft-catsvsdogs-dataset"
    )

    dataset_path = os.path.join(path, "PetImages")

    return dataset_path