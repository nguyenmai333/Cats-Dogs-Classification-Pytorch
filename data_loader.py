import os
import shutil
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import random_split

#create dogs, cats folder for training
root = r'dogs-vs-cats'
train_dir = r'dogs-vs-cats\train'
test_dir = r'dogs-vs-cats\test1'
res = r'dogs-vs-cats\res'
list_train = os.listdir(train_dir)
dogs_train_dir = os.path.join(train_dir, 'dogs')
cats_train_dir = os.path.join(train_dir, 'cats')

batch_size = 32 ##
image_size = (64, 64) ##

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
])

dataset = ImageFolder(root=train_dir, transform=transform)
classes = dataset.classes
print('classes: ', classes)

validation_size = int(0.2 * len(dataset))
train_size = len(dataset) - validation_size

train_dataset, val_dataset = random_split(dataset, [train_size, validation_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print("Number of Image in Train Dataset:", len(train_loader))
print("Number of Image in Validation Dataset:", len(val_loader))
