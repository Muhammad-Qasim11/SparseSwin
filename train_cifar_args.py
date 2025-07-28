import torch
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
from traintest import train # Ensure this imports the updated train function
import build_model as build
import argparse
import os # Import os for path checking
import torch.nn as nn # Needed for nn.Linear for vanilla Swin
import torchvision.models as models # Needed for torchvision Swin models

torch.random.manual_seed(1)

"""
Parser:
python train_cifar.py -dataset cifar10 -batchsize=24 -reg_type=None -sparseswin_type tiny -device cuda -epochs 1 -freeze_12 False
"""

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', help='cifar10 or cifar100', type=str, choices=['cifar10', 'cifar100'])
parser.add_argument('-batchsize', help='the number of batch', type=int)
parser.add_argument('-reg_type', help='the type of regularization', type=str, default='None', choices=['None', 'l1', 'l2'])
parser.add_argument('-reg_lambda', help='the lambda for regualrization\nIf regularization None then you dont need to specify this', type=float, default=0)
parser.add_argument('-sparseswin_type', help='Type of the model', type=str, choices=['tiny', 'small', 'base'])
parser.add_argument('-device', help='the computing device [cpu/cuda/etc]', type=str)
parser.add_argument('-epochs', help='the number of epoch', type=int, default=100)
parser.add_argument('-show_per', help='Displaying verbose per batch for each epoch', type=int, default=300)
parser.add_argument('-lf', help='number of lf', type=int, default=2)
parser.add_argument('-ltoken_num', help='the number of latent token', type=int, default=49)
parser.add_argument('-ltoken_dims', help='the dimension of latent token', type=int, default=256)
parser.add_argument('-freeze_12', help='freeze the first two block on swin', type=bool, default=False)
parser.add_argument('-vanilla_swin', action='store_true', help='Use vanilla Swin Transformer without SparseSwin block') # Added for vanilla Swin option
# New argument for resuming training
parser.add_argument('-resume_checkpoint', type=str, default=None, help='Path to a checkpoint file to resume training from.')

args = parser.parse_args()

# Global variables from args
dataset = args.dataset
batch_size = args.batchsize
reg_type = args.reg_type
reg_lambda = args.reg_lambda
swin_type = args.sparseswin_type # Keep this as 'tiny', 'small', 'base' for build_model
device = torch.device(args.device)
epochs = args.epochs
show_per = args.show_per
ltoken_num = args.ltoken_num
ltoken_dims = args.ltoken_dims
lf = args.lf
freeze_12 = args.freeze_12
vanilla_swin = args.vanilla_swin

# Dataset Config
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

data_transform = {
        'train': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.Normalize(mean, std)
                ]),
        'val': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((224, 224), antialias=None),
                    transforms.Normalize(mean, std)
                ])
    }

status = False # Set to True if you need to download
if dataset == 'cifar10':
    train_dataset = datasets.CIFAR10(
                    root='./datasets/torch_cifar10/',
                    train=True,
                    transform=data_transform['train'],
                    download=status)
    val_dataset = datasets.CIFAR10(
                    root='./datasets/torch_cifar10/',
                    train=False,
                    transform=data_transform['val'],
                    download=status)
    num_classes = 10
elif dataset == 'cifar100':
    train_dataset = datasets.CIFAR100(
                    root='./datasets/torch_cifar100/',
                    train=True,
                    transform=data_transform['train'],
                    download=status)
    val_dataset = datasets.CIFAR100(
                    root='./datasets/torch_cifar100/',
                    train=False,
                    transform=data_transform['val'],
                    download=status)
    num_classes = 100
else:
    raise ValueError("Unsupported dataset. Choose 'cifar10' or 'cifar100'.")


train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2, # Changed from 8 to 2
                pin_memory=True)

val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2, # Changed from 8 to 2
                pin_memory=True)


if __name__ == '__main__':
    print(f"Training process will begin..")
    print(f"Model Type : {'Vanilla Swin' if vanilla_swin else 'SparseSwin'} {swin_type} | ltoken_num : {ltoken_num} | ltoken_dims : {ltoken_dims}")
    print(f"Dataset : {dataset}")
    print(f"Epochs : {epochs} | Batch Size : {batch_size} | Freeze first two blocks? : {freeze_12}")
    print(f"Device : {device}")

    start_epoch = 0 # Initialize start_epoch

    # --- Checkpoint Loading Logic ---
    model = None # Initialize model to None
    optimizer = None # Initialize optimizer to None

    if args.resume_checkpoint:
        print(f"Attempting to resume training from checkpoint: {args.resume_checkpoint}")
        if not os.path.exists(args.resume_checkpoint):
            raise FileNotFoundError(f"Checkpoint file not found: {args.resume_checkpoint}")

        checkpoint = torch.load(args.resume_checkpoint, map_location=device)

        # Re-build the model structure first, then load state_dict
        if vanilla_swin:
            if swin_type.lower() == 'tiny':
                model = models.swin_t(weights=None) # Load without weights initially
            elif swin_type.lower() == 'small':
                model = models.swin_s(weights=None)
            elif swin_type.lower() == 'base':
                model = models.swin_b(weights=None)
            num_ftrs = model.head.in_features
            model.head = nn.Linear(num_ftrs, num_classes)
        else: # SparseSwin
            model = build.buildSparseSwin(
                image_resolution=224,
                swin_type=swin_type,
                num_classes=num_classes,
                ltoken_num=ltoken_num,
                ltoken_dims=ltoken_dims,
                num_heads=16,
                qkv_bias=True,
                lf=lf,
                attn_drop_prob=.0,
                lin_drop_prob=.0,
                freeze_12=freeze_12,
                device=device)

        model.to(device) # Move model to device BEFORE loading state_dict for some models, or after. Here, after creation.
        model.load_state_dict(checkpoint['model_state_dict'])

        # Initialize optimizer before loading its state
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint['epoch'] + 1 # Set the epoch to resume from

        print(f"Successfully resumed from epoch {checkpoint['epoch']}, training will continue from epoch {start_epoch} to {epochs-1}.")
    else:
        # Original model building logic if NOT resuming from a checkpoint
        if vanilla_swin:
            print(f"Starting new training for Vanilla Swin Transformer ({swin_type})...")
            if swin_type.lower() == 'tiny':
                model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
            elif swin_type.lower() == 'small':
                model = models.swin_s(weights=models.Swin_S_Weights.IMAGENET1K_V1)
            elif swin_type.lower() == 'base':
                model = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
            else:
                raise ValueError(f"Unsupported Swin type '{swin_type}' for vanilla Swin.")
            num_ftrs = model.head.in_features
            model.head = nn.Linear(num_ftrs, num_classes)
            if reg_type != 'None':
                print("Warning: Regularization ('l1' or 'l2') is specified but running vanilla Swin. Setting reg_type to 'None'.")
                reg_type = 'None' # Override regularization for vanilla Swin if specified
        else: # SparseSwin
            print(f"Starting new training for SparseSwin Transformer ({swin_type})...")
            model = build.buildSparseSwin(
                image_resolution=224,
                swin_type=swin_type,
                num_classes=num_classes,
                ltoken_num=ltoken_num,
                ltoken_dims=ltoken_dims,
                num_heads=16,
                qkv_bias=True,
                lf=lf,
                attn_drop_prob=.0,
                lin_drop_prob=.0,
                freeze_12=freeze_12,
                device=device)

        model.to(device) # Move model to device after creation
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01) # Initialize optimizer

    criterion = nn.CrossEntropyLoss().to(device)

    # Call the train function with the appropriate parameters
    train(
        train_loader=train_loader,
        swin_type=swin_type,
        dataset=dataset,
        total_epochs=epochs, # Pass args.epochs as total_epochs
        model=model,
        lf=lf,
        token_num=ltoken_num,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        show_per=show_per,
        reg_type=reg_type,
        reg_lambda=reg_lambda,
        validation=val_loader,
        start_epoch=start_epoch # Pass the calculated start_epoch
    )
