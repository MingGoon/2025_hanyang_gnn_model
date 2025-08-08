"""
Object Movement Detection Model Testing Script

This script evaluates a trained model on test data to measure performance
metrics including accuracy, precision, recall, and F1-score. It generates
classification reports, confusion matrices, and visualizations.
"""

import matplotlib
matplotlib.use('Agg')
import torch
import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_training import RealDataLiftingDataset
from improved_baseline_model import create_lifting_prediction_model

def collate_fn_filter_none(batch):
    """Custom collate function that filters out None values from failed data loads."""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: 
        return None
    return batch[0] if len(batch) == 1 else torch.utils.data.dataloader.default_collate(batch)

def test_model(model, test_loader, device):
    """Run model evaluation on test dataset and collect predictions."""
    model.eval()
    all_predictions, all_labels = [], []
    all_logits = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            if batch is None: continue
            try:
                for key in ['depth', 'normal', 'instance_masks']: batch[key] = batch[key].to(device)
                labels = batch['labels']['movement_labels'].to(device)
                if labels.numel() == 0: continue

                predictions = model(
                    depth=batch['depth'], normal=batch['normal'], instance_masks=batch['instance_masks'],
                    object_metadata=batch['object_metadata'], camera_info=batch['camera_info'])
                logits = predictions['movement_logits']

                if logits.shape[0] != labels.shape[0]: continue
                
                pred_classes = torch.argmax(logits, dim=1)
                all_predictions.extend(pred_classes.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_logits.extend(logits.cpu().numpy().tolist())
                
            except Exception as e:
                print(f"⚠️ Error during testing batch: {e}"); import traceback; traceback.print_exc()
                
    return all_predictions, all_labels, all_logits

def main():
    parser = argparse.ArgumentParser(description='Test trained object movement detection model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--base_dir', type=str, required=True, help='Base directory containing all episode folders for testing')
    parser.add_argument('--output_dir', type=str, default='./test_results', help='Directory to save test results')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading worker processes')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'auto' else 'cpu')
    print(f"🔧 Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    print("🤖 Creating model architecture to load weights...")
    model_config = {
        'train_image_size': (1224, 1024),
        'distance_threshold_px': 200.0,
        'image_size': (224, 224),
        'resnet_name': 'resnet18',
        'cnn_out_channels': 128,
        'node_feature_dim': 128,
        'edge_feature_dim': 16,
        'hidden_dim': 128,
        'num_gnn_layers': 3,
        'dropout': 0.2
    }
    model = create_lifting_prediction_model(model_config)
    
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        print(f"✅ Model weights loaded successfully from {args.model_path}")
    except Exception as e:
        print(f"❌ Failed to load model weights: {e}"); return

    print("📊 Loading Test Dataset from nested structure...")
    test_dataset = RealDataLiftingDataset(base_dir=args.base_dir, image_size=(224, 224))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_filter_none, num_workers=args.num_workers)
    
    if len(test_dataset) == 0: print("❌ No test data found."); return

    print("🧪 Starting evaluation...")
    predictions, labels, logits = test_model(model, test_loader, device)

    if not labels: print("❌ No valid labels found to evaluate."); return

    report = classification_report(labels, predictions, target_names=['Not Moved', 'Moved'], output_dict=True, zero_division=0)
    cm = confusion_matrix(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro', zero_division=0)

    print("\n" + "="*50 + "\n Final Test Results\n" + "="*50)
    print(classification_report(labels, predictions, target_names=['Not Moved', 'Moved'], zero_division=0))
    print(f"Macro F1-Score: {f1:.4f}\n" + "="*50)

    results_path = os.path.join(args.output_dir, 'test_report.json')
    with open(results_path, 'w') as f: json.dump(report, f, indent=4)
    print(f"💾 Classification report saved to {results_path}")

    fig, ax = plt.subplots(figsize=(8, 8))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    class_names = ['Not Moved', 'Moved']
    ax.set_xticks(np.arange(len(class_names))); ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), va='center', ha='center', color="white" if cm[i,j] > cm.max()/2. else "black")
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix')
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close(fig)
    print(f"📊 Confusion matrix saved to {cm_path}")
    
    
    print("📊 Generating logit distribution histogram...")
    # Logit 값을 NumPy 배열로 변환
    logits_array = np.array(logits)
    
    # 클래스 0('Not Moved')과 클래스 1('Moved')의 logit 값을 분리
    logits_for_not_moved = logits_array[:, 0]
    logits_for_moved = logits_array[:, 1]
    
    # 히스토그램 생성
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(logits_for_not_moved, bins=50, color='blue', alpha=0.7)
    plt.title('Logits Distribution for "Not Moved" (Class 0)')
    plt.xlabel('Logit Value')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.hist(logits_for_moved, bins=50, color='red', alpha=0.7)
    plt.title('Logits Distribution for "Moved" (Class 1)')
    plt.xlabel('Logit Value')
    
    plt.tight_layout()
    
    # 히스토그램 이미지 저장
    logit_hist_path = os.path.join(args.output_dir, 'logit_distribution.png')
    plt.savefig(logit_hist_path)
    print(f"📊 Logit distribution histogram saved to {logit_hist_path}")

if __name__ == "__main__":
    main()
