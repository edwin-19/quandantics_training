import torch
import torch.nn as nn
from torch.optim import Adam

from torch.utils.data import DataLoader

from model import TorchModel
from dataset import TorchDataset

from tqdm import tqdm

def create_dataset(batch_size=1000):
    train_random_input = torch.randn(batch_size, 1000)
    train_random_labels = torch.randint(0, 9, (batch_size, ))
    
    train_dataset = TorchDataset(train_random_input, train_random_labels)
    train_loader = DataLoader(
       train_dataset, batch_size=8, 
       pin_memory=True, num_workers=4,
       shuffle=True
    )
    
    test_random_input = torch.randn(batch_size // 10, 1000)
    test_random_labels = torch.randint(0, 9, (batch_size // 10, ))
    
    val_dataset = TorchDataset(test_random_input, test_random_labels)
    val_dataloader = DataLoader(
       val_dataset, batch_size=8, 
       pin_memory=True, num_workers=4,
       shuffle=True
    )
    
    return train_loader, val_dataloader

if __name__ == "__main__":
    # Set seed
    torch.random.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # Define hyper params
    lr = 1e-4
    epochs = 20
    
    device = torch.device('cuda:1')
    
    # Define compoennets
    train_data, val_data = create_dataset()
    model = TorchModel()
    model.to(device)
    
    # Define loss func & optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optim = Adam(model.parameters(), lr=lr)
    
    for epoch in tqdm(range(epochs)):
        # Set to train mode
        # Run training loop
        model.train()
        train_loss = 0
        
        for batch in train_data:
            input_data = batch['image'].to(device)
            labels = batch['labels'].to(device)
            
            model.zero_grad()
            
            logits = model(input_data)
            
            loss = criterion(logits, labels)
            loss.backward()
            optim.step()
            
            train_loss += loss.item() * input_data.size(0)
            
        model.eval()
        eval_loss = 0
        for batch in val_data:
            input_data = batch['image'].to(device)
            labels = batch['labels'].to(device)
            
            with torch.no_grad():
                logits = model(input_data)
            
            loss = criterion(logits, labels)
            eval_loss += loss.item() * input_data.size(0)
        
        print('Train Loss: {}'.format(train_loss / len(train_data)))
        print('Eval Loss: {}'.format(eval_loss / len(val_data)))