import cv2
import numpy as np
import base64
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def ndarray_to_tensor(img):
    """Transform np.ndarray to pytorch tensor."""
    torchimage = []
    for array in img:
        internal_array = []
        for number in array:
            updated_number = number / 256
            internal_array.append(updated_number)
        torchimage.append(internal_array)
    torchimage = [torchimage]
    torchimage = torch.tensor(np.array(torchimage), dtype=torch.float32)
    return torchimage


def tensor_to_py(tensor):
    """Transform pytorch tensor to JSON-serializable python-native list."""
    return [float(value) for value in tensor.flatten()]


def get_predictions(dataURI):
    classes = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
    ]
    model = NeuralNetwork()
    model.load_state_dict(torch.load("model.pth"))
    encoded_data = dataURI.split(',')[1]

    npArr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(npArr, cv2.IMREAD_COLOR)

    # convert 3 channel image (RGB) to 1 channel image (GRAY)
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # resize to (28, 28)
    grayImage = cv2.resize(grayImage, (28, 28), interpolation=cv2.INTER_LINEAR)
    # expand to numpy array dimenstion to (1, 28, 28)
    img = np.expand_dims(grayImage, axis=0)

    img_tensor = ndarray_to_tensor(img)
    model.eval()
    # run image through the model without backpropagation
    with torch.no_grad():
        prediction_tensor = model(img_tensor)
        prediction_list = tensor_to_py(prediction_tensor)
        return {
            "prediction_list": prediction_list,
            "top_prediction": classes[prediction_tensor[0].argmax(0)]
        }
