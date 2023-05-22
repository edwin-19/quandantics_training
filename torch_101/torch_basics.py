# Lets import the torch module
import torch

if __name__ == "__main__":
    # Lets start by createing a torch tensor
    # torch requires a list as input 
    # tensor is just a nxn array with access to gpu
    tensor_a = torch.tensor([100 , 200, 300])
    print(tensor_a) # tensor([100, 200, 300])
    
    # Lets change the type of tensor
    tensor_b = tensor_a.type(torch.float32)
    print(tensor_b) # tensor([100., 200., 300.])
    
    # Create random tensor
    tensor_c = torch.randn(3,)
    print(tensor_c)
    
    # Create based of device
    device_cpu = torch.device('cpu')
    device_gpu = torch.device('cuda:0')
    tensor_c = tensor_c.to(device_gpu)
    tensor_b = tensor_b.to(device_gpu)
    
    # Mathematical operations for 2 tensor
    # Everything must be the same to perform math ops
    # on 2 tensors
    tensor_d = tensor_b + tensor_c
    print(tensor_d) # tensor([ 99.0631, 200.8365, 299.5941], device='cuda:0')