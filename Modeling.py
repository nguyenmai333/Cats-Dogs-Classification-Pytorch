import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torch.optim as optim
from sklearn.metrics import accuracy_score

def train_model(model, device, train_loader, val_loader, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        cnt = 0
        for inputs, labels in train_loader:
            cnt += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            print(f'{cnt}/{len(train_loader)}')

        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                _, val_preds = torch.max(val_outputs, 1)

                all_preds.extend(val_preds.cpu().numpy())
                all_labels.extend(val_labels.cpu().numpy())

        val_accuracy = accuracy_score(all_labels, all_preds)
        print(f'Validation Accuracy: {val_accuracy:.4f}')


def define_model(train_loader, val_loader):
    Model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    num_features = Model.fc.in_features
    Model.fc = nn.Linear(num_features, 2)  # 2 classes (dogs and cats)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Model = Model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(Model.parameters(), lr=0.001)

    train_model(Model, device, train_loader, val_loader, criterion, optimizer, num_epochs=5)

    return Model