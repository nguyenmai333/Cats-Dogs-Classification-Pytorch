from data_loader import train_loader, val_loader, classes, root, train_dir, test_dir
import torch
from matplotlib import pyplot as plt
from Modeling import define_model
from classifer import predict

def main():
    Model = define_model(train_loader, val_loader)
    torch.save(Model.state_dict(), 'Cats-Dogs-Recognition.pth')

if __name__ == "__main__":
    main()