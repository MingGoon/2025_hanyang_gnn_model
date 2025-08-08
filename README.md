# Object Movement Detection with Graph Neural Networks

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7.0-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 Project Overview

This project implements a **Graph Neural Network (GNN)**-based deep learning model for detecting object movement in 3D environments. The model leverages multiple modalities including depth images, surface normal vectors, and object contour information to predict whether objects have moved over time.

### 🎯 Key Features

- **Multi-modal Input**: Fusion of depth, normal, and contour information
- **GNN Architecture**: Modeling spatial relationships between objects
- **Real-time Inference**: Efficient CNN+GNN pipeline
- **Robust Features**: Generalized model that doesn't rely on pose, size, or class ID
- **End-to-End Solution**: Complete pipeline from preprocessing to post-processing

### 🔬 Technology Stack

- **Deep Learning Framework**: PyTorch 2.7.0, PyTorch Geometric
- **Computer Vision**: OpenCV, torchvision
- **Experiment Tracking**: Weights & Biases (wandb)
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib
- **Machine Learning**: scikit-learn

## 🏗️ Model Architecture

### 1. CNN Backbone (`depth_normal_backbone.py`)

```
Input: Depth(1ch) + Normal(3ch) + Instance Masks
   ↓
ResNet18/50 + FPN → Multi-scale feature maps
   ↓
RoI Align → Object-specific features
```

### 2. GNN Structure (`improved_baseline_model.py`)

```
Object Features → Node Feature Constructor
                     ↓
                 Graph Constructor
                     ↓
                 GCN Layers (3 layers)
                     ↓
              Movement Prediction Head
```

### 3. Feature Extraction Pipeline

1. **Geometric Features**: Bounding box, contour area, aspect ratio
2. **Relational Features**: Inter-object distances, relationship to lifted object
3. **Visual Features**: RoI features extracted from CNN
4. **Normalization**: Scale normalization using fixed reference points

## 📁 Project Structure

```
gnn_model/
├── 📜 train_data.py              # Main training script
├── 📜 data_training.py           # Dataset and trainer classes
├── 📜 improved_baseline_model.py # GNN model architecture
├── 📜 depth_normal_backbone.py   # CNN backbone (Depth+Normal)
├── 📜 inference_data.py          # Inference script
├── 📜 test_data.py              # Model evaluation script
├── 📜 requirements.txt          # Python dependencies
├── 📜 install_requirements.sh   # Installation script
└── 📁 example_datasets/         # Example data
    ├── 📜 camera_info.json      # Camera configuration
    └── 📁 episode_0/           # Episode data
        ├── 📁 action_metadata/  # Movement label data
        ├── 📁 camera_info/      # Camera bounding boxes
        ├── 📁 contour/          # Object contour data
        ├── 📁 depth/            # Depth images (.png)
        ├── 📁 image_data/       # RGB camera images
        └── 📁 normal/           # Surface normal data (.bin)
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd gnn_model

# Install dependencies (automatic)
chmod +x install_requirements.sh
./install_requirements.sh

# Or manual installation
pip install -r requirements.txt
```

### 2. Data Preparation

Data should follow this nested directory structure:

```
your_dataset/
├── camera_info.json
└── episode_*/
    ├── action_metadata/
    │   └── action_metadata_*.json
    ├── contour/
    │   └── contour_*.json
    ├── depth/
    │   └── depth_*.png
    ├── normal/
    │   └── normal_*.bin
    └── image_data/
        └── camera_*.png
```

### 3. Model Training

```bash
python train_data.py \
    --base_dir ./your_dataset \
    --num_epochs 50 \
    --batch_size 1 \
    --learning_rate 1e-4 \
    --save_dir ./checkpoints \
    --wandb_project "my-object-movement"
```

### 4. Model Evaluation

```bash
python test_data.py \
    --model_path ./checkpoints/best_model.pth \
    --base_dir ./test_dataset \
    --output_dir ./test_results
```

### 5. Running Inference

```bash
python inference_data.py \
    --model_path ./checkpoints/best_model.pth \
    --base_dir ./new_data \
    --output_dir ./predictions
```

## 📊 Data Format

### Action Metadata (JSON)
```json
{
  "step": 0,
  "lifted_objects": ["/World/obj/tuna_can_0/_07_tuna_fish_can"],
  "moved_objects": [
    {"object_name": "pudding_box_1"}
  ],
  "movement_analysis": {
    "object_name": {
      "position_diff": [x, y, z],
      "position_magnitude": float,
      "orientation_magnitude": float
    }
  }
}
```

### Contour Data (JSON)
```json
{
  "contours": [
    {
      "object_id": 15,
      "prim_path": "/World/obj/tuna_can_1/_07_tuna_fish_can",
      "contour_points": [[x1, y1], [x2, y2], ...]
    }
  ],
  "next_action_prim_path": "/World/obj/target_object/mesh"
}
```

### Camera Info (JSON)
```json
{
  "camera_info": {
    "md.height": 1024,
    "md.width": 1224
  }
}
```

## ⚙️ Model Configuration

### Hyperparameters

```python
model_config = {
    'train_image_size': (1224, 1024),     # Original image size
    'distance_threshold_px': 200.0,       # Graph connection threshold
    'image_size': (224, 224),             # Processing image size
    'roi_output_size': 7,                 # RoI feature size
    'resnet_name': 'resnet18',            # Backbone network
    'cnn_out_channels': 128,              # CNN output channels
    'node_feature_dim': 128,              # Node feature dimension
    'edge_feature_dim': 16,               # Edge feature dimension
    'hidden_dim': 128,                    # GNN hidden dimension
    'num_gnn_layers': 3,                  # Number of GNN layers
    'dropout': 0.2                       # Dropout rate
}
```

### Class Weights
- **Not Moved (Class 0)**: Weight 1.0
- **Moved (Class 1)**: Weight 5.0 (to handle class imbalance)

## 📈 Performance Metrics

The model is evaluated using the following metrics:

- **Accuracy**: Overall prediction accuracy
- **Precision**: Class-wise precision
- **Recall**: Class-wise recall
- **F1-Score**: Harmonic mean-based comprehensive score
- **Confusion Matrix**: Classification results visualization