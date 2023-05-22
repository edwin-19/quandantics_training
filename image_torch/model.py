import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, labels=3) -> None:
        super(CNN, self).__init__()
        
        self.extractor1 = nn.Sequential(
            nn.Conv2d(3, 32, 2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.extractor2 = nn.Sequential(
            nn.Conv2d(32, 64, 2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(193600, 256),
            nn.ReLU(),
            nn.Linear(256, labels),
        )
        
    def forward(self, image):
        x = self.extractor1(image)
        x = self.extractor2(x)
        
        return self.fc(x)
    
if __name__ == "__main__":
    model = CNN()
    random_img = torch.randn(5, 3, 224, 224)
    
    model.eval()
    
    with torch.no_grad():
        logits = model(random_img)
        
    print(logits.argmax(dim=1).shape)