from pathlib import Path
from sklearn.model_selection import train_test_split

def create_dataset(data_path):
    data = Path(data_path)
    images_dataset = list(data.glob('**/*.jpg'))
    
    labels = [image.parent.stem for image in images_dataset]
    train_images, test_images, train_labels, test_labels = train_test_split(
        images_dataset, labels, test_size=0.2, random_state=2022, stratify=labels
    )
    
    return train_images, test_images, train_labels, test_labels