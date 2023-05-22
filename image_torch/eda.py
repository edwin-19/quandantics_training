from pathlib import Path
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    dataset = Path("data/dataset/")
    labels = list(dataset.glob('*'))
    
    plt.figure(figsize=(10, 5))
    index = 0
    for lbl in labels:
        images = list(lbl.glob('*.jpg'))
        print(len(images))
        
        img = Image.open(images[0])
        print(img.size)
        
        for img_path in np.random.choice(images, 3):
            img_np = Image.open(img_path)
            
            img_resized = img_np.resize((224, 224))
            
            plt.subplot(3, 3, index + 1)
            plt.title(lbl.stem)
            plt.imshow(img_resized)
            index += 1
            
    plt.tight_layout()
    plt.savefig('eda.jpg')