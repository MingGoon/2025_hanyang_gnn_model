"""
Object Movement Detection Model Training Script

This script trains a GNN-based model to predict whether objects in a scene
have moved or not. It uses depth, normal, and contour data from nested 
episode directories and includes wandb integration for experiment tracking.
"""

import torch
import argparse
import os
import wandb
from torch.utils.data import DataLoader, random_split

from data_training import RealDataLiftingDataset, RealDataTrainer
from improved_baseline_model import create_lifting_prediction_model

def collate_fn_filter_none(batch):
    """Custom collate function that filters out None values from failed data loads."""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: 
        return None
    return batch[0] if len(batch) == 1 else torch.utils.data.dataloader.default_collate(batch)

def main():
    parser = argparse.ArgumentParser(description='Train object movement detection model on nested directory structure')
    parser.add_argument('--base_dir', type=str, required=True, help='Base directory containing episode folders with action_metadata, contour, depth, etc.')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--train_split', type=float, default=0.8, help='Ratio of data used for training (rest for validation)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading worker processes')
    parser.add_argument('--wandb_project', type=str, default="object-movement-detection", help="Weights & Biases project name for experiment tracking")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'auto' else 'cpu')
    print(f"🔧 Using device: {device}")

    # Initialize experiment tracking
    wandb.init(project=args.wandb_project, config=vars(args))

    print("📊 Loading and scanning dataset from nested structure...")
    dataset = RealDataLiftingDataset(
        base_dir=args.base_dir,
        image_size=(224, 224)
    )
    
    if len(dataset) == 0:
        print("❌ No data available for training. Please check the directory structure.")
        return
    
    train_size = int(args.train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    print(f"📊 Data split: Train {len(train_dataset)}, Validation {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_filter_none, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_filter_none, num_workers=args.num_workers)
    
    print("🤖 Creating model architecture...")
    model_config = {
        'train_image_size': (1224, 1024),
        'distance_threshold_px': 200.0,
        'image_size': (224, 224),
        'roi_output_size': 7,  # RoIAlign output feature map size
        'resnet_name': 'resnet18',
        'cnn_out_channels': 128,
        'node_feature_dim': 128,
        'edge_feature_dim': 16,
        'hidden_dim': 128,
        'num_gnn_layers': 3,
        'dropout': 0.2
    }
    model = create_lifting_prediction_model(model_config)
    print(f"📊 Model created with {sum(p.numel() for p in model.parameters()):,} parameters.")
    
    trainer = RealDataTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        save_dir=args.save_dir
    )
    
    print("🚀 Starting training...")
    trainer.train(num_epochs=args.num_epochs, save_best=True)
    
    print("✅ Training finished!")

if __name__ == "__main__":
    main()
