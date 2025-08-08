"""
Modified CNN Backbone for handling depth and normal inputs

This module modifies the CNN backbone to properly handle depth images and normal vectors
instead of RGB images.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from typing import Dict, List, Optional


class DepthNormalCNNBackbone(nn.Module):
    """
    CNN backbone specifically designed for depth and normal vector inputs.
    
    This backbone:
    1. Accepts depth images (1 channel) and normal vectors (3 channels)
    2. Uses ResNet+FPN architecture
    3. Handles instance mask embeddings
    """
    
    def __init__(
        self,
        use_depth: bool = True,
        use_normal: bool = True,
        use_instance_masks: bool = True,
        resnet_name: str = "resnet50",
        pretrained: bool = True,
        out_channels: int = 256,
        instance_embedding_dim: int = 16,
        max_instances: int = 20,
    ):
        """
        Initialize the depth+normal CNN backbone.
        
        Args:
            use_depth: Whether to use depth images
            use_normal: Whether to use normal vectors
            use_instance_masks: Whether to use instance masks
            resnet_name: ResNet variant to use
            pretrained: Whether to use pretrained weights
            out_channels: Output channels for FPN
            instance_embedding_dim: Instance mask embedding dimension
            max_instances: Maximum number of instances
        """
        super().__init__()
        
        self.use_depth = use_depth
        self.use_normal = use_normal
        self.use_instance_masks = use_instance_masks
        self.instance_embedding_dim = instance_embedding_dim
        
        # Calculate input channels
        input_channels = 0
        if use_depth:
            input_channels += 1
        if use_normal:
            input_channels += 3
        if use_instance_masks:
            input_channels += instance_embedding_dim
        
        assert input_channels > 0, "At least one input modality must be enabled"
        
        # Instance mask embedding
        if use_instance_masks:
            self.instance_embedder = nn.Embedding(max_instances + 1, instance_embedding_dim)
        
        # Create ResNet backbone
        if resnet_name == "resnet18":
            resnet = models.resnet18(pretrained=False)  # We'll handle pretrained weights manually
            in_channels_list = [64, 128, 256, 512]
        elif resnet_name == "resnet34":
            resnet = models.resnet34(pretrained=False)
            in_channels_list = [64, 128, 256, 512]
        elif resnet_name == "resnet50":
            resnet = models.resnet50(pretrained=False)
            in_channels_list = [256, 512, 1024, 2048]
        elif resnet_name == "resnet101":
            resnet = models.resnet101(pretrained=False)
            in_channels_list = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported ResNet variant: {resnet_name}")
        
        # Modify the first conv layer to accept our input channels
        original_conv = resnet.conv1
        new_conv = nn.Conv2d(
            input_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        
        # Initialize the new conv layer
        nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
        
        # If using pretrained weights, initialize appropriately
        if pretrained:
            # Load pretrained ResNet
            pretrained_resnet = getattr(models, resnet_name)(pretrained=True)
            pretrained_conv = pretrained_resnet.conv1
            
            with torch.no_grad():
                channel_idx = 0
                
                # Initialize depth channel
                if use_depth:
                    new_conv.weight[:, channel_idx:channel_idx+1, :, :] = torch.mean(
                        pretrained_conv.weight, dim=1, keepdim=True
                    )
                    channel_idx += 1
                
                # Initialize normal channels
                if use_normal:
                    new_conv.weight[:, channel_idx:channel_idx+3, :, :] = pretrained_conv.weight
                    channel_idx += 3
                
                # Initialize instance embedding channels with random values
                if use_instance_masks:
                    new_conv.weight[:, channel_idx:channel_idx+instance_embedding_dim, :, :] = torch.mean(
                        pretrained_conv.weight, dim=1, keepdim=True
                    ).repeat(1, instance_embedding_dim, 1, 1) * 0.1
            
            # Copy other pretrained weights
            resnet.load_state_dict(pretrained_resnet.state_dict(), strict=False)
        
        # Replace the conv1 layer
        resnet.conv1 = new_conv
        
        # Create FPN backbone
        from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
        extra_blocks = LastLevelMaxPool()
        
        self.backbone = BackboneWithFPN(
            resnet,
            return_layers={"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"},
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
        )
        
        self.out_channels = out_channels
    
    def forward(
        self,
        depth: Optional[torch.Tensor] = None,
        normal: Optional[torch.Tensor] = None,
        instance_masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the backbone.
        
        Args:
            depth: Depth image tensor [B, 1, H, W]
            normal: Normal vector tensor [B, 3, H, W]
            instance_masks: Instance mask tensor [B, 1, H, W]
            
        Returns:
            Dictionary of feature maps at different scales
        """
        inputs = []
        
        if self.use_depth and depth is not None:
            if depth.dim() == 3:
                # Add batch dimension if missing
                depth = depth.unsqueeze(0)
            inputs.append(depth)
        
        if self.use_normal and normal is not None:
            if normal.dim() == 3:
                # Add batch dimension if missing
                normal = normal.unsqueeze(0)
            inputs.append(normal)
        
        if self.use_instance_masks and instance_masks is not None:
            # Convert instance masks to embeddings
            if instance_masks.dim() == 3:
                # Add batch dimension if missing
                instance_masks = instance_masks.unsqueeze(0)
            
            B, _, H, W = instance_masks.shape
            instance_masks = torch.clamp(instance_masks, 0, self.instance_embedder.num_embeddings - 1)
            instance_masks = instance_masks.long()
            
            # Reshape for embedding lookup
            flat_masks = instance_masks.view(B, -1)
            embedded = self.instance_embedder(flat_masks)
            embedded = embedded.view(B, H, W, self.instance_embedding_dim)
            embedded = embedded.permute(0, 3, 1, 2)  # BHWC -> BCHW
            
            inputs.append(embedded)
        
        # Concatenate all inputs
        if len(inputs) == 0:
            raise ValueError("No input modalities provided")
        
        x = torch.cat(inputs, dim=1)
        
        # Forward through backbone
        return self.backbone(x)


if __name__ == "__main__":
    # Test the modified backbone
    print("Testing DepthNormalCNNBackbone...")
    
    backbone = DepthNormalCNNBackbone(
        use_depth=True,
        use_normal=True,
        use_instance_masks=True,
        resnet_name="resnet18",
        pretrained=False,
        out_channels=128,
        instance_embedding_dim=16
    )
    
    print("Backbone created successfully!")
    
    # Test with dummy data
    batch_size = 1
    height, width = 224, 224
    
    depth = torch.randn(batch_size, 1, height, width)
    normal = torch.randn(batch_size, 3, height, width)
    instance_masks = torch.randint(0, 5, (batch_size, 1, height, width)).float()
    
    print("Testing forward pass...")
    try:
        with torch.no_grad():
            features = backbone(depth=depth, normal=normal, instance_masks=instance_masks)
        
        print("Forward pass successful!")
        for key, value in features.items():
            print(f"Feature {key}: {value.shape}")
            
    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback
        traceback.print_exc()

