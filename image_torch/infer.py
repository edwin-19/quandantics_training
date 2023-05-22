import argparse
import torch
from transfer_model import MobileCNN
from pathlib import Path
from PIL import Image
from torchvision import transforms as T
import numpy as np
from matplotlib import pyplot as plt

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--model_path', default='weights/bagle_classifier.pt')
    args.add_argument('--data_path', default='data/dataset')
    
    return args.parse_args()

def load_pipe(model_path):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    model = MobileCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    return transform, model

if __name__ == "__main__":
    args = get_args()
    data_path = Path(args.data_path)
    label_map = {
        0: 'broken',
        1: 'defect',
        2: 'good'
    }
    
    transforms, model = load_pipe(args.model_path)
    
    images = list(data_path.glob('**/*.jpg'))
    
    plt.figure(figsize=(15, 10))
    index = 0
    
    for image in np.random.choice(images, 6):
        img = Image.open(image)
    
        img_tensor = transforms(img)
        with torch.no_grad():
            logits = model(img_tensor.unsqueeze(0))
        
        preds = label_map[logits.argmax().item()]
        plt.subplot(2, 3, index + 1)
        plt.title('Pred: {}'.format(preds))
        plt.imshow(img)
        index += 1
        
    plt.tight_layout()
    plt.savefig('output.jpg')
        
    