import torch
import torch.nn as nn

class TorchModel(nn.Module):
    # Initialize our model
    def __init__(self, labels=10) -> None:
        super(TorchModel, self).__init__()
        
        # Define or layers which we want to use
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1000, 256)
        self.fc2 = nn.Linear(256, labels)
    
    # The forward function that requires to be overwrriten
    def forward(self, img):
        x = self.flatten(img)
        x = self.fc1(x)
        return self.fc2(x)

if __name__ == "__main__":
    # Define model and random input
    sample_img = torch.randn(1 ,3 , 224, 224)
    model = TorchModel()
    model.eval()
    
    # Run prediction
    with torch.no_grad():
        logits = model(sample_img)
    
    print(logits)