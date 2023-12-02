from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from matplotlib import pyplot as pyplot
from torchvision.models import resnet50, ResNet50_Weights
import os
import shutil

def load_model(model_path):
    state_dict = torch.load(model_path)
    Model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    num_features = Model.fc.in_features
    Model.fc = nn.Linear(num_features, 2)  # 2 classes (dogs and cats)
    Model.load_state_dict(state_dict)
    return Model


def predict(Model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Model.to(device)
    test_dir = r'dogs-vs-cats\test1'
    res = r'dogs-vs-cats\res'
    image_size = (64, 64) 
    list_test_items = os.listdir(test_dir)
    os.makedirs(os.path.join(res,'dogs'), exist_ok=True)
    os.makedirs(os.path.join(res,'cats'), exist_ok=True)

    Model.eval()

    cnt = 0 
    for item in list_test_items:
        image_path = os.path.join(test_dir, item)
        
        cnt += 1

        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        image = transform(image)
        image = image.unsqueeze(0)  
        image = image.to(device) 

        with torch.no_grad():
            output = Model(image)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
        
        if predicted_class == 1:
            shutil.copy(os.path.join(test_dir, item), os.path.join(res, 'dogs'))
        else: 
            shutil.copy(os.path.join(test_dir, item), os.path.join(res, 'cats'))
        print(f'images: {cnt} / {len(list_test_items)}')


if __name__ == "__main__":
    Model = load_model('Cats-Dogs-Recognition.pth')
    predict(Model)