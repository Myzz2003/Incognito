'''
Process a clip of image which is the face of a person. 
Classify the person in the image clip.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
import numpy as np
from utils import OptimizeImage

# imagedatastore
from torch.utils.data import Dataset, DataLoader


import os

class Net(nn.Module):
    '''
    BackBone of the network:
        - inputLayer:       Convolutional 3*3, 32 filters, stride 1, padding 1
                            MaxPooling 2*2, stride 2
        - hiddenLayer_1:    Convolutional 3*3, 32 filters, stride 1, padding 1
                            MaxPooling 2*2, stride 2
        - hiddenLayer_2:    Convolutional 3*3, 64 filters, stride 1, padding 1
                            MaxPooling 2*2, stride 2
        - hiddenLayer_3:    flatten -> Fully Connected 128 neurons

        - outputLayer:      Fully Connected {outputSize} neurons
    '''


    def __init__(self, inputSize: tuple[int, int], outputSize: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.inputSize = inputSize
        linear_input_size = inputSize[0] * inputSize[1]
        self.Backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LazyBatchNorm2d(),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=linear_input_size, out_features=200),
            nn.ReLU(),
            # nn.BatchNorm1d(120),
            nn.Linear(in_features=200, out_features=84),
            # nn.ReLU(),
            nn.Sigmoid(),
            nn.Linear(in_features=84, out_features=outputSize),
            nn.Softmax(dim=1)
            )
    
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.Backbone(x)
    
class ImageDataset(Dataset):
        def __init__(self, data_dir: str, labels: list[str], input_size: tuple[int, int], device: torch.device = torch.device("cpu")) -> None:
            self.data_dir = data_dir
            self.device = device
            self.input_size = input_size
            self.labels = labels
            self.data = []
            for label in labels:
                for filename in os.listdir(os.path.join(data_dir, label)):
                    if not filename.startswith("."):
                        self.data.append((os.path.join(data_dir, label, filename), label))
        
        def __len__(self) -> int:
            return len(self.data)
        
        def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
            img_path, label = self.data[index]
            img = cv.imread(img_path)
            img = OptimizeImage(img)
            img = cv.resize(img, self.input_size)
            img = torch.tensor(img).permute(2, 0, 1).float().to(self.device)
            retLabel = torch.zeros(len(self.labels))
            retLabel[self.labels.index(label)] = 1
            retLabel = retLabel.to(self.device)

            return img, retLabel



class NetworkProcessor:
    def __init__(self, data_dir: str, input_shape: tuple[int, int], model_path: str = None, device: str = "cpu") -> None:
        self.data_dir = data_dir
        self.label_dict = {}
        self.input_shape = input_shape
        self.device = torch.device(device)
        
        idx = 0
        for label in os.listdir(data_dir):
            if not label.startswith("."):
                self.label_dict[idx] = label
                idx += 1

        self.net = Net(input_shape, len(self.label_dict))
        if model_path is not None:
            self.net.load_state_dict(torch.load(model_path))
            print(f"Model loaded from {model_path}")
        else:
            print(f"Model initialized.")

    def trainNetwork(self, epochs: int = 100, learning_rate: float = 0.0001, *args, **kwargs):
        labels = list(self.label_dict.values())
        print(f"train with labels: {self.label_dict.values()}")

        self.net.to(self.device)
            
        dataset = ImageDataset(self.data_dir, labels, self.input_shape, device=self.device)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        print(f"Length of Dataset:{len(dataset)}")

        print(f"Training options loading...")
        
        # load criterion and optimizer from kwargs or use default
        if kwargs.get('criterion') is not None:
            criterion_argument = kwargs.get('criterion')
            if type(criterion_argument) == str:
                try:
                    criterion = getattr(nn, criterion_argument)
                except:
                    Warning(f"Criterion {criterion_argument} not found, use default nn.CrossEntropyLoss")
                    criterion = nn.CrossEntropyLoss()
            elif isinstance(criterion_argument, nn.Module):
                criterion = criterion_argument
            else:
                raise TypeError(f"Criterion must be a string or a nn.Module, but got {type(criterion_argument)}")
        else:
            criterion = nn.CrossEntropyLoss()

        if kwargs.get('optimizer') is not None:
            optimizer_argument = kwargs.get('optimizer')
            if type(optimizer_argument) == str:
                try:
                    optimizer = getattr(torch.optim, optimizer_argument)
                except:
                    Warning(f"Optimizer {optimizer_argument} not found, use default torch.optim.SGD")
                    optimizer = torch.optim.SGD
            elif isinstance(optimizer_argument, torch.optim.Optimizer):
                optimizer = optimizer_argument
            else:
                raise TypeError(f"Optimizer must be a string or a torch.optim.Optimizer, but got {type(optimizer_argument)}")
            
        else:
            optimizer = torch.optim.SGD

        optimizer = optimizer(self.net.parameters(), lr=learning_rate)

        # train
        print(f"Training started...")
        for epoch in range(epochs):
            for i, (inputs, labels) in enumerate(dataloader):
                optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                print(f"epoch: {epoch+1}, loss: {loss}") if (epoch+1) % 10 == 0 and i == 0 else None

        # save model
        torch.save(self.net.state_dict(), "./model.pt")
        print(f"Model saved at \"./model.pt\".")
        return {'loss': loss.item(),
                'trainedEpochs': epoch+1, 
                'learningRate': learning_rate, 
                'inputShape': self.input_shape, 
                'labels': self.label_dict, 
                'modelPath': './model.pt'
                }

    def getLabelFromNet(self, img: np.ndarray) -> dict:
        img = cv.resize(img, self.input_shape)
        img = OptimizeImage(img)
        img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        out = self.net(img)
        idx = torch.argmax(out)
        # convert to integer
        idx = idx.item()
        retdict = {"label": self.label_dict[idx], "confidence": out[0][idx].item()}
        return retdict


if __name__ == '__main__':
    net = NetworkProcessor("./sorted_faces", (64, 64))
    max_epochs = 1000
    trained_epochs = 0
    current_loss = 10
    while trained_epochs < max_epochs and current_loss > 0.6:
        training_info = net.trainNetwork(100, 0.0001, optimizer="Adam")
        trained_epochs += training_info.get('trainedEpochs')
        current_loss = training_info.get('loss')

            

