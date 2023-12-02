# Cats vs Dogs Classification with PyTorch

This repository demonstrates how to use a Convolutional Neural Network (CNN) implemented in PyTorch for classifying images between cats and dogs.

# Usage
### Requirements
```
pip install -r requirement.txt
```
### Data
Download the cats and dogs image dataset from the Kaggle competition.
```
kaggle competitions download -c dogs-vs-cats
```
Downloaded file and place it in the root folder of the repository.
### Unzip and Split data to Train, Validation.
```
python get-data.py
```
`get-data.py` Run the script to extract and split the data into training and validation sets.

### Training
Train the model using the ResNet50 architecture on the dataset.
(batch-size = 32, image-size = (64,64), learning-rate = 0.001)
```
python train.py
```

### Classification
To classify cat and dog images in the test folder, run the script using the trained model from `train.py`.
```
python classify.py
```

### Notes

If there are any errors, please check and add frequently asked questions to the instructions.

Ensure the necessary libraries are installed; details can be found in the requirements.txt file.

A Kaggle account is required to download the dataset from the competition.
