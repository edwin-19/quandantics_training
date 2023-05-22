# Import the torch dataset class and torch class
import torch
from torch.utils.data import Dataset, DataLoader

# Lets create our dataset class, which is inherited from the dataset object
class TorchDataset(Dataset):
    # Init method
    def __init__(self, images, labels):
        super(TorchDataset, self).__init__()
        self.images = images
        self.labels = labels
    
    # Returns how many inside the dataset
    # Optionally implemented
    def __len__(self):
        return len(self.images)
    
    # Get item method, requires the selection of dataset based off index
    # Define processing logic here
    def __getitem__(self, index):
        # Select based off index
        selected_img = self.images[index]
        selected_label = self.labels[index]
        
        # Now we can return the following
        # Tuple / Dict / List/ Tensor
        return {
            'image': selected_img,
            'labels': selected_label
        }    
    
if __name__ == "__main__":
    # Lets define our random variables
    random_img = torch.randn(1000, 3, 224, 224)
    random_labels = torch.randint(0, 10, (1000,))
    
    # Create our dataset object
    dataset = TorchDataset(random_img, random_labels)
    # Print out our dataset
    # print(next(iter(dataset)))
    
    data = DataLoader(
        dataset, 
        batch_size=8, shuffle=True,
        pin_memory=True, num_workers=4
    )
    
    print(next(iter(data)))