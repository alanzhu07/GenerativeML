import torch
import torchvision
from torchvision import transforms

image_folder = 'album_covers'

def bandcamp_dataset(output_size = (28, 28), image_folder = image_folder, split = True):
    transform = transforms.Compose([
        transforms.Resize(output_size),
        transforms.ToTensor(),
        ])
    dataset = torchvision.datasets.ImageFolder(image_folder, transform = transform)
    if not split:
        return dataset
    train_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
    return train_set, test_set