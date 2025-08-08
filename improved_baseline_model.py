"""
Improved Baseline Model - FINAL REVIEWED VERSION
This version is completely self-contained and does not depend on pre-calculated
'pose', 'size', or 'class_id' from the metadata. It computes all necessary
geometric and relational features from the basic inputs.
This makes the model more robust and applicable to real-world scenarios with unseen objects.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, radius_graph
from torch_geometric.data import Data
from torchvision.ops import RoIAlign
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import cv2

# depth_normal_backbone.py는 별도 파일로 존재한다고 가정합니다.
from depth_normal_backbone import DepthNormalCNNBackbone

class NodeFeatureConstructor(nn.Module):
    """
    Node feature generator for GNN-based object movement detection.
    
    - Eliminates dependency on 3D pose, size, and class ID
    - Dynamically computes relational features from contour and depth data
    - Normalizes all spatial features using external fixed reference points
    """
    def __init__(
        self,
        cnn_feature_dim: int,
        roi_output_size: int,
        node_feature_dim: int,
        image_size: Tuple[int, int],
        normalization_base: float
    ):
        super().__init__()
        self.roi_output_size = roi_output_size
        self.cnn_feature_dim = cnn_feature_dim
        self.image_height, self.image_width = image_size
        self.normalization_base = normalization_base

        # ResNet-FPN의 첫 번째 특징 맵은 보통 입력의 1/4 크기이므로 spatial_scale을 1/4로 설정
        self.roi_align = RoIAlign(
            output_size=(self.roi_output_size, self.roi_output_size),
            spatial_scale=1.0 / 4.0,
            sampling_ratio=2,
            aligned=True
        )

        visual_feature_dim = cnn_feature_dim * roi_output_size * roi_output_size
        hand_crafted_feature_dim = 8 + 1 + 2 + 1  # BBox(8) + Contour(1) + Relational(2) + LiftedFlag(1) = 12
        total_input_dim = visual_feature_dim + hand_crafted_feature_dim
        
        self.feature_mlp = nn.Sequential(
            nn.Linear(total_input_dim, node_feature_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(node_feature_dim * 2),
            nn.Dropout(0.1),
            nn.Linear(node_feature_dim * 2, node_feature_dim)
        )

    def _calculate_geometric_features(self, contour: np.ndarray, orig_size: Tuple[int, int]) -> Dict:
        """Contour로부터 수동 특징을 계산하고 '고정된 기준'으로 정규화합니다."""
        if contour.size < 3: return {}

        orig_height, orig_width = orig_size
        x, y, w, h = cv2.boundingRect(contour)
        rect = cv2.minAreaRect(contour)
        (cx_rot, cy_rot), (rot_w, rot_h), angle = rect
        contour_area = cv2.contourArea(contour)
        
        return {
            "center_x_px": x + w / 2.0,
            "center_y_px": y + h / 2.0,
            "width_px": float(w),
            "height_px": float(h),
            "norm_center_x": (x + w / 2.0) / self.normalization_base,
            "norm_center_y": (y + h / 2.0) / self.normalization_base,
            "norm_width": w / self.normalization_base,
            "norm_height": h / self.normalization_base,
            "aspect_ratio": max(w, h) / (min(w, h) + 1e-6),
            "norm_rotated_width": rot_w / self.normalization_base,
            "norm_rotated_height": rot_h / self.normalization_base,
            "area_efficiency": contour_area / ((w * h) + 1e-6),
            "norm_visible_area": contour_area / (orig_width * orig_height + 1e-6)
        }

    def forward(
        self,
        cnn_features: Dict[str, torch.Tensor],
        depth_tensor: torch.Tensor,
        object_metadata: List[List[Dict]],
        camera_info: List[Dict]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pose/Class ID 없이 노드 특징과 2D 위치를 생성합니다."""
        device = depth_tensor.device
        batch_size = depth_tensor.shape[0]
        feature_map = cnn_features['0']
        
        all_node_features = []
        all_node_positions_2d = []

        for batch_idx in range(batch_size):
            batch_metadata = object_metadata[batch_idx]
            if not batch_metadata: continue

            orig_width = camera_info[batch_idx]['width']
            orig_height = camera_info[batch_idx]['height']
            depth_array = depth_tensor[batch_idx].squeeze(0).cpu().numpy()

            precomputed_info = []
            lifted_idx = -1
            for i, obj_meta in enumerate(batch_metadata):
                contour = np.array(obj_meta['contour'])
                info = self._calculate_geometric_features(contour, (orig_height, orig_width))
                
                mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
                cv2.fillPoly(mask, [contour], 1)
                
                resized_mask = cv2.resize(mask, (self.image_width, self.image_height), interpolation=cv2.INTER_NEAREST)
                depth_values = depth_array[resized_mask == 1]
                mean_depth = np.mean(depth_values[depth_values > 0]) if (depth_values > 0).any() else 0.0
                info['depth'] = mean_depth
                
                precomputed_info.append(info)
                if obj_meta.get('is_lifted', False):
                    lifted_idx = i

            roi_boxes = []
            for info in precomputed_info:
                x1 = info['center_x_px'] - info['width_px'] / 2
                y1 = info['center_y_px'] - info['height_px'] / 2
                x2 = info['center_x_px'] + info['width_px'] / 2
                y2 = info['center_y_px'] + info['height_px'] / 2
                roi_boxes.append(torch.tensor([batch_idx, x1, y1, x2, y2], dtype=torch.float32, device=device))
            
            if not roi_boxes: continue
            roi_boxes = torch.stack(roi_boxes)
            roi_features = self.roi_align(feature_map, roi_boxes).view(len(batch_metadata), -1)

            lifted_info = precomputed_info[lifted_idx] if lifted_idx != -1 else {'norm_center_x': 0.5, 'norm_center_y': 0.5, 'depth': 0.5}

            for obj_idx in range(len(batch_metadata)):
                current_info = precomputed_info[obj_idx]
                
                dist_2d = np.hypot(current_info['norm_center_x'] - lifted_info['norm_center_x'],
                                   current_info['norm_center_y'] - lifted_info['norm_center_y'])
                depth_diff = np.abs(current_info['depth'] - lifted_info['depth'])

                hand_crafted_features = torch.tensor([
                    current_info['norm_center_x'], current_info['norm_center_y'], current_info['norm_width'], current_info['norm_height'],
                    current_info['aspect_ratio'], current_info['norm_rotated_width'], current_info['norm_rotated_height'],
                    current_info['area_efficiency'], current_info['norm_visible_area'],
                    dist_2d, depth_diff, float(batch_metadata[obj_idx].get('is_lifted', False))
                ], dtype=torch.float32, device=device)
                
                combined_feat = torch.cat([roi_features[obj_idx], hand_crafted_features])
                node_feat = self.feature_mlp(combined_feat)
                all_node_features.append(node_feat)
                all_node_positions_2d.append(torch.tensor([current_info['norm_center_x'], current_info['norm_center_y']], device=device))
        
        if not all_node_features:
            return torch.empty(0, node_feature_dim, device=device), torch.empty(0, 2, device=device)
        
        return torch.stack(all_node_features), torch.stack(all_node_positions_2d)


class GraphConstructor(nn.Module):
    def __init__(self, distance_threshold: float, edge_feature_dim: int):
        super().__init__()
        self.distance_threshold = distance_threshold
        self.edge_mlp = nn.Sequential(nn.Linear(2, edge_feature_dim), nn.ReLU(), nn.Linear(edge_feature_dim, edge_feature_dim))
    
    def forward(self, node_features: torch.Tensor, object_positions_2d: torch.Tensor) -> Data:
        edge_index = radius_graph(object_positions_2d, r=self.distance_threshold, batch=None, loop=False)
        src, dst = edge_index
        rel_pos = object_positions_2d[dst] - object_positions_2d[src]
        edge_attr = self.edge_mlp(rel_pos)
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, pos=object_positions_2d)


class GNNLayers(nn.Module):
    def __init__(self, node_feature_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.gnn_layers = nn.ModuleList([GCNConv(node_feature_dim, hidden_dim)] + [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers - 1)])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, (gnn_layer, layer_norm) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            x_new = gnn_layer(x, edge_index)
            x_new = layer_norm(x_new)
            x = x + self.dropout(F.relu(x_new)) if i > 0 else self.dropout(F.relu(x_new))
        return x


class PredictionHeads(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.movement_head = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 2))
    
    def forward(self, node_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {'movement_logits': self.movement_head(node_features)}


class LiftingPredictionModel(nn.Module):
    # Initialize model components
    def __init__(self, 
                 train_image_size: Tuple[int, int], 
                 distance_threshold_px: float, 
                 image_size: Tuple[int, int],
                 roi_output_size: int,      # RoIAlign output feature size
                 resnet_name: str, 
                 cnn_out_channels: int, 
                 node_feature_dim: int, 
                 edge_feature_dim: int,
                 hidden_dim: int, 
                 num_gnn_layers: int, 
                 dropout: float):
        super().__init__()
        train_width, train_height = train_image_size
        train_long_side = float(max(train_width, train_height))
        normalized_distance_threshold = distance_threshold_px / train_long_side
        
        self.cnn_backbone = DepthNormalCNNBackbone(resnet_name=resnet_name, out_channels=cnn_out_channels)
        
        # Initialize node feature constructor with RoI output size
        self.node_feature_constructor = NodeFeatureConstructor(
            cnn_feature_dim=cnn_out_channels, 
            node_feature_dim=node_feature_dim,
            image_size=image_size, 
            normalization_base=train_long_side,
            roi_output_size=roi_output_size
        )
        self.graph_constructor = GraphConstructor(
            distance_threshold=normalized_distance_threshold, 
            edge_feature_dim=edge_feature_dim
        )
        self.gnn = GNNLayers(
            node_feature_dim=node_feature_dim, 
            hidden_dim=hidden_dim, 
            num_layers=num_gnn_layers, 
            dropout=dropout
        )
        self.prediction_head = PredictionHeads(in_dim=hidden_dim)
        
    def forward(self, depth: torch.Tensor, normal: torch.Tensor, instance_masks: torch.Tensor,
                object_metadata: List[List[Dict]], camera_info: List[Dict]) -> Dict[str, torch.Tensor]:
        
        cnn_features = self.cnn_backbone(depth=depth, normal=normal, instance_masks=instance_masks)
        
        node_features, object_positions_2d = self.node_feature_constructor(
            cnn_features=cnn_features, depth_tensor=depth,
            object_metadata=object_metadata, camera_info=camera_info)
        
        if node_features.shape[0] == 0:
            return {'movement_logits': torch.empty(0, 2, device=depth.device)}

        graph = self.graph_constructor(node_features=node_features, object_positions_2d=object_positions_2d)
        updated_features = self.gnn(x=graph.x, edge_index=graph.edge_index)
        predictions = self.prediction_head(updated_features)
        
        return predictions

def create_lifting_prediction_model(config: Optional[Dict] = None) -> LiftingPredictionModel:
    """Helper function to create the model with a config dictionary."""
    default_config = {
        'train_image_size': (1224, 1024),
        'distance_threshold_px': 200.0,
        'image_size': (224, 224),
        'roi_output_size': 7,
        'resnet_name': 'resnet18',
        'cnn_out_channels': 128,
        'node_feature_dim': 128,
        'edge_feature_dim': 16,
        'hidden_dim': 128,
        'num_gnn_layers': 2,
        'dropout': 0.1,
    }
    if config:
        default_config.update(config)
    return LiftingPredictionModel(**default_config)
