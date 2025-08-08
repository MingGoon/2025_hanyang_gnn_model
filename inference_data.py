"""
Object Movement Detection Model Inference Script

This script runs inference on new data to predict whether objects have moved.
It works with or without ground truth labels - if labels are available, it
provides accuracy validation. Outputs detailed predictions with confidence scores.
"""

import torch
import argparse
import os
import json
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from data_training import RealDataLiftingDataset
from improved_baseline_model import create_lifting_prediction_model

def collate_fn_filter_none(batch):
    """Custom collate function that filters out None values from failed data loads."""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: 
        return None
    return batch[0] if len(batch) == 1 else torch.utils.data.dataloader.default_collate(batch)

def run_inference_no_labels(model, data_loader, device):
    """Run inference on data with optional ground truth validation."""
    model.eval()
    all_results = []
    sample_idx = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Running Inference"):
            if batch is None: 
                continue
            
            try:
                # Move data to device
                for key in ['depth', 'normal', 'instance_masks']: 
                    batch[key] = batch[key].to(device)
                
                # Check if labels exist (for comparison if available)
                has_labels = 'labels' in batch and 'movement_labels' in batch['labels']
                labels = batch['labels']['movement_labels'].to(device) if has_labels else None
                
                if has_labels and labels.numel() == 0: 
                    continue

                # Run model inference
                predictions = model(
                    depth=batch['depth'], 
                    normal=batch['normal'], 
                    instance_masks=batch['instance_masks'],
                    object_metadata=batch['object_metadata'], 
                    camera_info=batch['camera_info']
                )
                logits = predictions['movement_logits']
                pred_classes = torch.argmax(logits, dim=1)
                probabilities = torch.softmax(logits, dim=1)
                
                # Get object metadata for additional info
                object_metadata = batch['object_metadata'][0] if batch['object_metadata'] else []
                
                # Store results for each object in the batch
                num_objects = len(pred_classes)
                for i in range(num_objects):
                    result = {
                        'sample_idx': sample_idx,
                        'object_idx': i,
                        'prediction': int(pred_classes[i].cpu().numpy()),
                        'prob_not_moved': float(probabilities[i][0].cpu().numpy()),
                        'prob_moved': float(probabilities[i][1].cpu().numpy()),
                        'logit_not_moved': float(logits[i][0].cpu().numpy()),
                        'logit_moved': float(logits[i][1].cpu().numpy()),
                        'prediction_label': 'Not Moved' if pred_classes[i] == 0 else 'Moved'
                    }
                    
                    # Add object metadata if available
                    if i < len(object_metadata):
                        obj_meta = object_metadata[i]
                        result.update({
                            'object_name': obj_meta.get('object_name', ''),
                            'prim_path': obj_meta.get('prim_path', ''),
                            'is_lifted': obj_meta.get('is_lifted', False)
                        })
                    
                    # Add ground truth if available (for validation)
                    if has_labels and i < len(labels):
                        result['ground_truth'] = int(labels[i].cpu().numpy())
                        result['ground_truth_label'] = 'Not Moved' if labels[i] == 0 else 'Moved'
                        result['correct'] = bool(pred_classes[i] == labels[i])
                    
                    all_results.append(result)
                
                sample_idx += 1
                
            except Exception as e:
                print(f"⚠️ Error during inference on sample {sample_idx}: {e}")
                import traceback
                traceback.print_exc()
                sample_idx += 1
                continue
    
    return all_results

def analyze_predictions(results):
    """Analyze prediction results (works with or without ground truth)."""
    if not results:
        print("❌ No results to analyze")
        return {}
    
    df = pd.DataFrame(results)
    
    # Basic prediction statistics
    total_objects = len(df)
    not_moved_predictions = (df['prediction'] == 0).sum()
    moved_predictions = (df['prediction'] == 1).sum()
    
    analysis = {
        'total_objects': int(total_objects),
        'not_moved_predictions': int(not_moved_predictions),
        'moved_predictions': int(moved_predictions),
        'not_moved_percentage': float(not_moved_predictions / total_objects * 100),
        'moved_percentage': float(moved_predictions / total_objects * 100)
    }
    
    # Confidence statistics
    analysis.update({
        'avg_confidence_not_moved': float(df['prob_not_moved'].mean()),
        'avg_confidence_moved': float(df['prob_moved'].mean()),
        'high_confidence_predictions': int((df[['prob_not_moved', 'prob_moved']].max(axis=1) > 0.8).sum()),
        'low_confidence_predictions': int((df[['prob_not_moved', 'prob_moved']].max(axis=1) < 0.6).sum())
    })
    
    # If ground truth is available, add accuracy metrics
    if 'ground_truth' in df.columns:
        correct_predictions = df['correct'].sum()
        accuracy = correct_predictions / total_objects
        
        analysis.update({
            'has_ground_truth': True,
            'correct_predictions': int(correct_predictions),
            'overall_accuracy': float(accuracy)
        })
    else:
        analysis['has_ground_truth'] = False
    
    return analysis

def save_results(results, analysis, output_dir):
    """Save inference results and analysis to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    results_file = os.path.join(output_dir, 'inference_predictions.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 Detailed predictions saved to {results_file}")
    
    # Save analysis summary
    analysis_file = os.path.join(output_dir, 'prediction_analysis.json')
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"📊 Analysis summary saved to {analysis_file}")
    
    # Save CSV for easy viewing
    if results:
        df = pd.DataFrame(results)
        csv_file = os.path.join(output_dir, 'inference_predictions.csv')
        df.to_csv(csv_file, index=False)
        print(f"📋 Results CSV saved to {csv_file}")

def main():
    parser = argparse.ArgumentParser(description='Run object movement detection inference on new data')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--base_dir', type=str, required=True, help='Base directory containing data to predict')
    parser.add_argument('--output_dir', type=str, default='./predictions', help='Directory to save prediction results')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading worker processes')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'auto' else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Create model architecture
    print("🤖 Creating model architecture...")
    model_config = {
        'train_image_size': (1224, 1024),
        'distance_threshold_px': 200.0,
        'image_size': (224, 224),
        'roi_output_size': 7,
        'resnet_name': 'resnet18',
        'cnn_out_channels': 128,
        'node_feature_dim': 128,
        'edge_feature_dim': 16,
        'hidden_dim': 128,
        'num_gnn_layers': 3,
        'dropout': 0.2
    }
    model = create_lifting_prediction_model(model_config)
    
    # Load model weights
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print(f"✅ Model weights loaded from {args.model_path}")
    
    # Load dataset
    print("📊 Loading dataset for prediction...")
    dataset = RealDataLiftingDataset(
        base_dir=args.base_dir,
        image_size=(224, 224)
    )
    
    if len(dataset) == 0:
        print("❌ No data found in the dataset")
        return
    
    print(f"📊 Dataset loaded with {len(dataset)} samples")
    
    # Create data loader
    data_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn_filter_none, 
        num_workers=args.num_workers
    )
    
    # Run inference
    print("🔍 Running inference for predictions...")
    results = run_inference_no_labels(model, data_loader, device)
    
    if not results:
        print("❌ No results obtained from inference")
        return
    
    # Analyze results
    print("📊 Analyzing predictions...")
    analysis = analyze_predictions(results)
    
    # Print results summary
    print("\n" + "="*60)
    print(" PREDICTION RESULTS SUMMARY")
    print("="*60)
    print(f"Total Objects Processed: {analysis['total_objects']}")
    print(f"")
    print(f"Prediction Distribution:")
    print(f"  Not Moved: {analysis['not_moved_predictions']} objects ({analysis['not_moved_percentage']:.1f}%)")
    print(f"  Moved: {analysis['moved_predictions']} objects ({analysis['moved_percentage']:.1f}%)")
    print(f"")
    print(f"Confidence Statistics:")
    print(f"  Average confidence for Not Moved: {analysis['avg_confidence_not_moved']:.3f}")
    print(f"  Average confidence for Moved: {analysis['avg_confidence_moved']:.3f}")
    print(f"  High confidence predictions (>80%): {analysis['high_confidence_predictions']}")
    print(f"  Low confidence predictions (<60%): {analysis['low_confidence_predictions']}")
    
    if analysis['has_ground_truth']:
        print(f"")
        print(f"Validation (Ground Truth Available):")
        print(f"  Overall Accuracy: {analysis['overall_accuracy']:.3f} ({analysis['correct_predictions']}/{analysis['total_objects']})")
    
    print("="*60)
    
    # Save results
    save_results(results, analysis, args.output_dir)
    
    print(f"\n✅ Prediction completed successfully!")
    print(f"📁 Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 