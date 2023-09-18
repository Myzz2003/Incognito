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
        # resize all input images to 64*64
        self.inputSize = inputSize
        linear_input_size = inputSize[0] * inputSize[1]
        self.Backbone = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LayerNorm((32, inputSize[0]//2, inputSize[1]//2)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=linear_input_size, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=outputSize),
            nn.Softmax(dim=1)
            )
    
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.Backbone(x)
    
class ImageDataset(Dataset):
        def __init__(self, data_dir: str, labels: list[str], input_size: tuple[int, int]) -> None:
            self.data_dir = data_dir
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
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            img = OptimizeImage(img)
            img = cv.resize(img, self.input_size)
            nimg = img
            img = torch.tensor(img)
            img = img.unsqueeze(0)
            img = img.float()
            retLabel = torch.zeros(len(self.labels))
            retLabel[self.labels.index(label)] = 1

            return img, retLabel



class NetworkProcessor:
    def __init__(self, data_dir: str, input_shape: tuple[int, int], model_path: str = None) -> None:
        self.data_dir = data_dir
        self.label_dict = {}
        self.input_shape = input_shape
        
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

    def trainNetwork(self):
        labels = list(self.label_dict.values())
        print(f"train with labels: {self.label_dict.values()}")

            
        dataset = ImageDataset(self.data_dir, labels, self.input_shape)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        print(f"Length of Dataset:{len(dataset)}")

        print(f"Training loading...")

        # train
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.net.parameters(), lr=0.001)
        for epoch in range(100):
            for i, (inputs, labels) in enumerate(dataloader):
                optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                print(f"epoch: {epoch}, batch: {i}, loss: {loss}") if epoch % 10 == 0 and i == 0 else None

        # save model
        torch.save(self.net.state_dict(), "./model.pt")
        print(f"Model saved at \"./model.pt\".")

    def getLabelFromNet(self, img: np.ndarray) -> str:
        img = cv.resize(img, self.input_shape)
        img = OptimizeImage(img)
        img = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()
        out = self.net(img)
        idx = torch.argmax(out)
        # convert to integer
        idx = idx.item()
        return self.label_dict[idx]


if __name__ == '__main__':
    npro = NetworkProcessor("./sorted_faces",(96, 96), "./model.pt")
    img = cv.imread("./faces/face_11.jpg", cv.IMREAD_GRAYSCALE)
    print(npro.getLabelFromNet(img))

            

