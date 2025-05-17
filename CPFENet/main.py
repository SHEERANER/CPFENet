import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset import CTDataset 
from train import train_image 

from model.ensemble import CPFENet
from model.resnet import resnet10, resnet34, resnet50, resnet101

from torchvision.models.video import swin3d_t, Swin3D_T_Weights
from torchvision.models.video.swin_transformer import PatchEmbed3d

def main():
    # Set hyperparameters
    batch_size = 32
    num_epoch = 300
    lr = 1e-5
    lr_min = 0
    gpu_ids = [0, 1, 2, 3]  # Use GPUs 0, 1, 2, 3
    device = torch.device(f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')

    # Set random seed for reproducibility
    torch.manual_seed(2002)

    # Define loss function
    loss_fn = nn.CrossEntropyLoss()

    model_type = 'CPFENet'
    # Initialize model 
    if model_type == "vit3d":
        model = ViT3D(
                image_size=(128, 128, 128),
                patch_size=16,
                num_classes=3,
                dim=1024,
                depth=6,
                heads=8,
                mlp_dim=2048,
                dropout=0.1,
                emb_dropout=0.1
        )
    elif model_type == "resnet_10":
        model = resnet10(
                sample_input_W=128,
                sample_input_H=128,
                sample_input_D=128,
                shortcut_type='B',
                no_cuda=False,
                fc_input = 512,
                num_classes=3)
    elif model_type == "resnet_18":
        model = resnet18(
                sample_input_W=128,
                sample_input_H=128,
                sample_input_D=128,
                shortcut_type='B',
                no_cuda=False,
                fc_input = 512,
                num_classes=3)
    elif model_type == "resnet_34":
        model = resnet34(
                sample_input_W=128,
                sample_input_H=128,
                sample_input_D=128,
                shortcut_type='B',
                no_cuda=False,
                fc_input = 512,
                num_classes=3)
    elif model_type == "resnet_50":
        model = resnet50(
                sample_input_W=128,
                sample_input_H=128,
                sample_input_D=128,
                shortcut_type='B',
                no_cuda=False,
                fc_input = 2048,
                num_classes=3)
    elif model_type == "resnet_101":
        model = resnet101(
                sample_input_W=128,
                sample_input_H=128,
                sample_input_D=128,
                shortcut_type='B',
                no_cuda=False,
                fc_input = 2048,
                num_classes=3)
        
    elif model_type == "swin_transformer":
        model = swin3d_t(num_classes=3)
        model.patch_embed = PatchEmbed3d(in_channels=1, patch_size=[2,4,4], embed_dim=96)
        model_stat = torch.load("./swin3d_t-7615ae03.pth")
        del model_stat['patch_embed.proj.weight']
        del model_stat['head.weight']
        del model_stat['head.bias']
        model.load_state_dict(model_stat, strict=False)

    elif model_type == "CPFENet":
        model = CPFENet()

    ct_dataset_all = CTDataset("/home/user04/project/lung_ml/data/CPFE/")
    # ct_dataset_all = torch.load("/home/user04/project/lung_ml/data/pass/lung/lung/ct_dataset_all.pt")

    train_size = int(len(ct_dataset_all) * 0.7)
    test_size = len(ct_dataset_all) - train_size 
    
    train_dataset, test_dataset = torch.utils.data.random_split(ct_dataset_all, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Start training
    train_losses, train_acces, eval_losses, eval_acces, eval_auces = train_image(
        model, loss_fn, train_loader, test_loader, gpu_ids, batch_size,
        num_epoch, lr, lr_min, init=True
    )

    # Optionally save training metrics
    np.save('train_losses.npy', train_losses)
    np.save('train_acces.npy', train_acces)
    np.save('eval_losses.npy', eval_losses)
    np.save('eval_acces.npy', eval_acces)
    np.save('eval_auces.npy', eval_auces)

if __name__ == "__main__":
    main()