from utils import create_dataset
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

from dataset import ImageDataset
from model import CNN
from transfer_model import MobileCNN
from tqdm import tqdm

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--data_path', default='data/dataset/')
    args.add_argument('--outdir', default='weights')
    args.add_argument('--batch_size', default=16, type=int)
    args.add_argument('--epoch', default=10, type=int)
    args.add_argument('--lr', default=1e-4, type=float)
    args.add_argument('--is_transfer', action='store_false')
    return args.parse_args()

def load_pipeline(is_transfer):
    train_dataset = ImageDataset(train_images, train_labels, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    
    val_dataset = ImageDataset(test_images, test_labels, transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    
    if is_transfer:
        model = MobileCNN()
    else:
        model = CNN()
        
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    return train_loader, val_loader, model, criterion, optim

def seed_everything(seed_val):
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)

if __name__ == "__main__":
    args = get_args()
    
    device = torch.device('cuda:1')
    seed_everything(42)
    
    train_images, test_images,\
        train_labels, test_labels = create_dataset(args.data_path)
        
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_loader, val_loader, model, criterion, optim = load_pipeline(args.is_transfer)
    train_acc = Accuracy(task="multiclass", num_classes=3).to(device)
    val_acc = Accuracy(task="multiclass", num_classes=3).to(device)
    
    for epoch in tqdm(range(args.epoch)):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            imgs = batch['img'].to(device)
            lbls = batch['label'].to(device)
            
            model.zero_grad()
            logits = model(imgs)
            
            loss = criterion(logits, lbls)
            loss.backward()
            optim.step()
            
            train_loss += loss.item()
            
            train_acc(logits.argmax(dim=1), lbls)
        
        model.eval()
        eval_loss = 0
        for val_batch in val_loader:
            val_imgs = val_batch['img'].to(device)
            val_lbls = val_batch['label'].to(device)
            
            with torch.no_grad():
                val_logits = model(val_imgs)
                val_loss = criterion(val_logits, val_lbls)    
                
            val_acc(val_logits.argmax(dim=1), val_lbls)
            
            eval_loss += val_loss.item()
        
        print('Train Loss: {}'.format(train_loss / len(train_loader)))
        print('Train Acc: {}'.format(round(train_acc.compute().item() * 100, 2)))
        
        print('Val Loss: {}'.format(eval_loss / len(val_loader)))
        print('Val Acc: {}'.format(round(val_acc.compute().item() * 100, 2)))
        
        train_acc.reset()
        val_acc.reset()
        
    # Save weights
    if not args.is_transfer:
        model_name = '/bagle_classifier.pt'
    else:
        model_name = '/bagle_classifier_mobile_net.pt'
    torch.save(model.state_dict(), args.outdir + model_name)