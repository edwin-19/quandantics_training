from PIL import Image, ImageDraw
import cv2
import numpy as np
from pathlib import Path

def proc_cv2(img_path):
    # Load image
    img = cv2.imread(img_path)
    print(img.shape)
    
    # Resize image
    img_resized = cv2.resize(img, (224, 224))
    
    # Change color to gray
    img_color = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # Draw rectangle
    cv2.rectangle(
        img_color, (100, 100), (200, 200), (255, 0, 0), 2
    )
    
    # Export image
    cv2.imwrite('export_cv2.jpg', img_color)
    

def proc_pil(img_path):
    img = Image.open(img_path)
    print(img.size)
    
    # convert to np array
    img_np = np.array(img)
    
    # Resize image
    img_resized = img.resize((224, 224))
    
    # convert to gray
    img_gray = img_resized.convert('L')
    
    # Draw image
    draw = ImageDraw.Draw(img_gray)
    draw.rectangle([(100, 100), (200, 200)], outline=(255, ), width=2)

    # export image
    img_gray.save('export_pil.jpg')


if __name__ == "__main__":
    data_path = Path('data/dataset/')
    imgs_path = list(data_path.glob('**/*.jpg'))
    
    img_path = str(np.random.choice(imgs_path))
    
    proc_cv2(img_path)
    proc_pil(img_path)