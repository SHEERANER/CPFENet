import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import resnet34
from torchvision.models.video import swin3d_t
from torchvision.models.video.swin_transformer import PatchEmbed3d

class CPFENet(nn.Module):
    """
    Combined Patch Feature Extraction Network (CPFENet)
    A hybrid neural network model that combines ResNet and Swin Transformer for 3D medical image analysis.
    
    This network architecture integrates the strengths of CNNs (via ResNet) and vision transformers (via Swin Transformer)
    to capture both local and global features from 3D medical images.
    
    Attributes:
        CNN (resnet34): 3D ResNet component for extracting hierarchical features
        swin_transformer (swin3d_t): 3D Swin Transformer component for capturing long-range dependencies
        fc (nn.Sequential): Fully connected layers for final classification/regression
    
    Methods:
        forward(x): Performs forward pass through the network
        
    Example:
        >>> model = CPFENet(input_w=128, input_h=128, input_d=128, input_channel=1)
        >>> output = model(input_tensor)
        
    Note:
        The input is expected to be a 5D tensor of shape (batch_size, channels, depth, height, width)
    """

    def __init__(self,
                 input_w: int = 128,
                 input_h: int = 128,
                 input_d: int = 128,
                 input_channel: int = 1,
                 CNN_shortcut_type: str = 'B',
                 CNN_fc_input: int = 512,
                 transformer_patch_size: list = [2, 4, 4],
                 transformer_patch_embedding_dim: int = 96,
                 module_outdim: int = 32,
                 output_dim: int = 3
                 ):
        """
        Initialize a new instance of CPFENet
        
        Args:
            input_w (int): Input width dimension
            input_h (int): Input height dimension
            input_d (int): Input depth dimension
            input_channel (int): Number of input channels
            CNN_shortcut_type (str): Type of shortcut connection in ResNet ('A' or 'B')
            CNN_fc_input (int): Number of input features for ResNet's fully connected layer
            transformer_patch_size (list): Patch size for Swin Transformer
            module_outdim (int): Output dimension of each feature extraction module
            output_dim (int): Final output dimension of the network
        """
        super(CPFENet, self).__init__()
        
        # Initialize 3D ResNet component
        # Configures a 3D ResNet34 with specified input dimensions and shortcut type
        self.CNN = resnet34(
            sample_input_W=input_w,
            sample_input_H=input_h,
            sample_input_D=input_d,
            shortcut_type=CNN_shortcut_type,
            no_cuda=False,
            fc_input=CNN_fc_input,
            num_classes=module_outdim
        )

        # Initialize 3D Swin Transformer component
        # Configures a 3D Swin Transformer with custom patch embedding
        print(1)
        self.swin_transformer = swin3d_t(num_classes=module_outdim)
        self.swin_transformer.patch_embed = PatchEmbed3d(
            in_channels=input_channel, 
            patch_size=transformer_patch_size, 
            embed_dim=transformer_patch_embedding_dim
        )

        # Final classification/regression head
        # Combines features from both modules and produces final output
        self.fc = nn.Sequential(
            nn.Linear(2 * module_outdim, output_dim),  # Concatenation of both feature vectors
            nn.Sigmoid()  # Sigmoid activation for output probabilities
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        
        # Extract features using ResNet
        resnet_features = self.CNN(x)
        
        # Extract features using Swin Transformer
        swin_features = self.swin_transformer(x)
        
        # Concatenate features from both modules
        combined_features = torch.cat((resnet_features, swin_features), dim=1)
        
        # Produce final output
        output = self.fc(combined_features)
        
        return output