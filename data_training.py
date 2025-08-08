"""
Object Movement Detection - Data Loading and Training Components

This module contains the dataset class and trainer for object movement detection.
It handles nested episode directory structures with action metadata, depth, normal,
and contour data to train models that predict whether objects have moved.
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
import cv2
import re
import struct
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb

# Type hints for better code documentation
from typing import Dict, List, Any, Optional, Tuple

class RealDataLiftingDataset(Dataset):
    """
    Dataset for object movement detection from nested episode directories.
    
    Supports nested episode directory structures containing:
    - action_metadata/: Movement labels and object information
    - contour/: Object contour data  
    - depth/: Depth images
    - normal/: Surface normal data
    - image_data/: RGB camera images
    """
    def __init__(self, 
                 base_dir: str,
                 image_size: tuple = (224, 224),
                 min_contour_points: int = 3):
        
        self.base_dir = base_dir
        self.image_size = image_size
        self.min_contour_points = min_contour_points
        self.samples = self._scan_all_samples()

        print(f"📊 Dataset Initialized: Found {len(self.samples)} valid frame pairs across all episodes.")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        try:
            return self._load_single_sample(sample_info)
        except Exception as e:
            print(f"⚠️ Error loading sample (Episode: {sample_info['episode_name']}, Frame: {sample_info['frame_index']}): {e}")
            return None

    def _scan_all_samples(self) -> List[Dict]:
        """Scan all episode directories to create training sample list."""
        all_samples = []
        try:
            episode_names = sorted([d for d in os.listdir(self.base_dir) if d.startswith("episode_") and os.path.isdir(os.path.join(self.base_dir, d))])
        except FileNotFoundError:
            print(f"❌ Base directory not found: {self.base_dir}"); return []

        print(f"🔍 Scanning {len(episode_names)} episodes...")
        for episode_name in tqdm(episode_names, desc="Scanning Episodes"):
            episode_path = os.path.join(self.base_dir, episode_name)
            
            # Check for action_metadata directory instead of results directory
            action_metadata_dir = os.path.join(episode_path, "action_metadata")
            if not os.path.isdir(action_metadata_dir): continue
            
            contour_dir = os.path.join(episode_path, 'contour')
            if not os.path.isdir(contour_dir): continue
            
            try:
                indices = [int(re.findall(r'\d+', f)[0]) for f in os.listdir(contour_dir) if f.startswith('contour_')]
                if not indices: continue
                max_index = max(indices)
            except (ValueError, IndexError): continue

            for frame_idx in range(max_index):
                # Load action metadata for this frame
                action_metadata_path = os.path.join(action_metadata_dir, f"action_metadata_{frame_idx}.json")
                if not os.path.exists(action_metadata_path): continue
                
                with open(action_metadata_path, 'r') as f: 
                    frame_labels = json.load(f)

                before_files_exist = all(os.path.exists(os.path.join(episode_path, f_type, f"{f_name}_{frame_idx}.{f_ext}")) 
                                         for f_type, f_name, f_ext in [("contour", "contour", "json"), ("depth", "depth", "png"), 
                                                                       ("image_data", "camera", "png"), ("normal", "normal", "bin")])
                
                after_contour_exists = os.path.exists(os.path.join(episode_path, "contour", f"contour_{frame_idx + 1}.json"))

                if before_files_exist and after_contour_exists:
                    all_samples.append({
                        "episode_path": episode_path, "episode_name": episode_name,
                        "frame_index": frame_idx, "labels": frame_labels})
        return all_samples

    def _load_single_sample(self, sample_info: Dict) -> Dict:
        """Load and preprocess a single data sample."""
        ep_path, f_idx = sample_info['episode_path'], sample_info['frame_index']
        
        contour_path = os.path.join(ep_path, "contour", f"contour_{f_idx}.json")
        depth_path = os.path.join(ep_path, "depth", f"depth_{f_idx}.png")
        normal_path = os.path.join(ep_path, "normal", f"normal_{f_idx}.bin")
        info_path = os.path.join(self.base_dir, "camera_info.json")

        with open(contour_path, 'r') as f: data_info = json.load(f)
        with open(info_path, 'r') as f: info = json.load(f)
        
        cam_info = info['camera_info']
        orig_height, orig_width = int(cam_info['md.height']), int(cam_info['md.width'])
        
        depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        
        try:
            normal_data = np.fromfile(normal_path, dtype=np.float32).reshape(orig_height, orig_width, 3)
        except ValueError:
            normal_data = np.fromfile(normal_path, dtype=np.float32)[3:].reshape(orig_height, orig_width, 3)

        depth_tensor = torch.from_numpy(cv2.resize(depth_img, (self.image_size[1], self.image_size[0]))).float().unsqueeze(0) / 1000.0
        normal_tensor = torch.from_numpy(cv2.resize(normal_data, (self.image_size[1], self.image_size[0]))).permute(2, 0, 1)

        object_metadata, instance_masks = self._create_metadata_and_masks(data_info, (orig_height, orig_width))
        
        # Handle both old and new format of moved_objects
        moved_objects = sample_info['labels'].get('moved_objects', [])
        if moved_objects and isinstance(moved_objects[0], dict):
            # New format: list of dictionaries with object_name
            moved_object_names = set(obj['object_name'] for obj in moved_objects)
            movement_labels = [1 if obj.get('object_name', '') in moved_object_names else 0 for obj in object_metadata]
        else:
            # Old format: list of strings (prim_paths)
            moved_paths = set(moved_objects)
            movement_labels = [1 if obj['prim_path'] in moved_paths else 0 for obj in object_metadata]
        
        return {
            'depth': depth_tensor, 'normal': normal_tensor, 'instance_masks': instance_masks,
            'object_metadata': [object_metadata],
            'labels': {'movement_labels': torch.tensor(movement_labels, dtype=torch.long)},
            'camera_info': [{'width': orig_width, 'height': orig_height}]
        }

    def _create_metadata_and_masks(self, data_info: Dict, orig_size: Tuple[int, int]) -> Tuple[List[Dict], torch.Tensor]:
        """Generate object metadata and instance masks from contour information."""
        object_metadata = []
        instance_masks = torch.zeros(1, self.image_size[0], self.image_size[1], dtype=torch.int32)
        
        scale_y, scale_x = self.image_size[0] / orig_size[0], self.image_size[1] / orig_size[1]
        lifted_prim_path = data_info.get("next_action_prim_path")

        for i, obj_data in enumerate(data_info.get("contours", [])):
            contour = np.array(obj_data['contour_points'], dtype=np.int32)
            if contour.shape[0] < self.min_contour_points: continue
            
            # Extract object_name from prim_path (e.g., "/World/obj/tuna_can_0/_07_tuna_fish_can" -> "tuna_can_0")
            prim_path = obj_data['prim_path']
            object_name = ""
            if "/World/obj/" in prim_path:
                path_parts = prim_path.split("/")
                for part in path_parts:
                    if part and not part.startswith("_") and part != "World" and part != "obj":
                        object_name = part
                        break
            
            object_metadata.append({
                'prim_path': prim_path,
                'object_name': object_name,
                'contour': contour.tolist(),
                'is_lifted': prim_path == lifted_prim_path})
            
            contour_scaled = contour.copy().astype(np.float32)
            contour_scaled[:, 0] *= scale_x
            contour_scaled[:, 1] *= scale_y
            cv2.fillPoly(instance_masks[0].numpy(), [contour_scaled.astype(np.int32)], i + 1)
            
        return object_metadata, instance_masks


class RealDataTrainer:
    def __init__(self, model, train_loader, val_loader, device, learning_rate, save_dir):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # Handle class imbalance with weighted loss
        # Not Moved (class 0): weight 1.0, Moved (class 1): weight 5.0  
        class_weights = torch.tensor([1.0, 5.0], dtype=torch.float32).to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.5)
        os.makedirs(save_dir, exist_ok=True)

    def train_epoch(self):
        self.model.train()
        total_loss, all_preds, all_labels = 0, [], []
        for batch in tqdm(self.train_loader, desc="Training"):
            if batch is None: continue
            try:
                for key in ['depth', 'normal', 'instance_masks']: batch[key] = batch[key].to(self.device)
                labels = batch['labels']['movement_labels'].to(self.device)
                if labels.numel() == 0: continue

                self.optimizer.zero_grad()
                predictions = self.model(
                    depth=batch['depth'], normal=batch['normal'], instance_masks=batch['instance_masks'],
                    object_metadata=batch['object_metadata'], camera_info=batch['camera_info'])
                logits = predictions['movement_logits']

                if logits.shape[0] != labels.shape[0]: continue
                
                loss = self.criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            except Exception as e:
                print(f"⚠️ Error during training batch: {e}"); import traceback; traceback.print_exc()
        
        avg_loss = total_loss / len(self.train_loader) if self.train_loader else 0
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels)) if all_labels else 0
        return {'loss': avg_loss, 'accuracy': accuracy}

    def validate_epoch(self):
        self.model.eval()
        total_loss, all_preds, all_labels = 0, [], []
        if not self.val_loader: return {'val_loss': 0, 'val_accuracy': 0}
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                if batch is None: continue
                try:
                    for key in ['depth', 'normal', 'instance_masks']: batch[key] = batch[key].to(self.device)
                    labels = batch['labels']['movement_labels'].to(self.device)
                    if labels.numel() == 0: continue

                    predictions = self.model(
                        depth=batch['depth'], normal=batch['normal'], instance_masks=batch['instance_masks'],
                        object_metadata=batch['object_metadata'], camera_info=batch['camera_info'])
                    logits = predictions['movement_logits']

                    if logits.shape[0] != labels.shape[0]: continue
                    
                    loss = self.criterion(logits, labels)
                    total_loss += loss.item()
                    all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                except Exception as e:
                    print(f"⚠️ Error during validation batch: {e}"); import traceback; traceback.print_exc()

        avg_loss = total_loss / len(self.val_loader) if self.val_loader else 0
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels)) if all_labels else 0
        return {'val_loss': avg_loss, 'val_accuracy': accuracy}

    def train(self, num_epochs, save_best=True):
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            train_metrics = self.train_epoch()
            val_metrics = self.validate_epoch()
            self.scheduler.step(val_metrics['val_loss'])
            
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f} | "
                  f"Val Loss: {val_metrics['val_loss']:.4f}, Acc: {val_metrics['val_accuracy']:.4f} | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            if wandb.run:
                wandb.log({
                    "epoch": epoch + 1, "train_loss": train_metrics['loss'], "train_accuracy": train_metrics['accuracy'],
                    "val_loss": val_metrics['val_loss'], "val_accuracy": val_metrics['val_accuracy'],
                    "learning_rate": self.optimizer.param_groups[0]['lr']
                })
            
            if save_best and val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                self.save_checkpoint(epoch, 'best_model.pth', val_metrics)
                print(f"✅ Best model saved at epoch {epoch+1} with validation loss {best_val_loss:.4f}")

    def save_checkpoint(self, epoch, filename, metrics):
        filepath = os.path.join(self.save_dir, filename)
        torch.save({
            'epoch': epoch, 'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics
        }, filepath)
