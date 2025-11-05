# Parking Lot Occupancy Estimation

**Deep Learning Project - Fall 2025**  
**Author:** Aminu Yiwere  
**GitHub Repository:** [https://github.com/0x1AY/Parking-Lot-Occupancy-Estimation-.git](https://github.com/0x1AY/Parking-Lot-Occupancy-Estimation-.git)

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Motivation](#motivation)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Installation & Setup](#installation--setup)
- [How to Run the Code](#how-to-run-the-code)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Progress Summary](#progress-summary)
- [Preliminary Results](#preliminary-results)
- [Updated Timeline](#updated-timeline)
- [Next Steps](#next-steps)
- [References](#references)

---

## 🎯 Project Overview

This project aims to develop an automated parking lot occupancy estimation system using deep learning and computer vision techniques. The system analyzes images or video footage of parking lots to determine the occupancy status of individual parking spaces in real-time, helping drivers find available spots quickly and efficiently.

### Problem Statement

Urban parking congestion leads to:

- Increased traffic congestion as drivers search for parking
- Wasted fuel and increased emissions
- Driver frustration and time loss
- Inefficient use of parking infrastructure

### Solution

We propose a deep learning-based approach that uses convolutional neural networks (CNNs) to:

1. Detect individual parking spaces in parking lot images
2. Classify each space as occupied or vacant
3. Provide real-time occupancy information
4. Generate occupancy statistics and trends

---

## 💡 Motivation

Traditional parking management systems rely on physical sensors (e.g., ultrasonic, infrared) which are:

- Expensive to install and maintain
- Prone to hardware failures
- Limited in scalability

A vision-based approach using existing surveillance cameras offers:

- **Cost-effectiveness**: Leverages existing infrastructure
- **Scalability**: Easy to deploy across multiple locations
- **Flexibility**: Can be adapted to different parking lot layouts
- **Rich data**: Provides visual context beyond simple occupancy

---

## 📊 Dataset

### Dataset Information

- **Source**: [PKLot Dataset](http://web.inf.ufpr.br/vri/databases/parking-lot-database/) or similar parking lot image datasets
- **Size**: ~12,000+ images from multiple parking lots
- **Conditions**: Various weather conditions, times of day, and camera angles
- **Labels**: Each parking space labeled as occupied/vacant

### Dataset Structure

```
data/
├── raw/
│   ├── images/          # Original parking lot images
│   └── annotations/     # Bounding boxes and labels
├── processed/
│   ├── train/           # Training set (70%)
│   ├── validation/      # Validation set (15%)
│   └── test/            # Test set (15%)
└── metadata.csv         # Image metadata and statistics
```

### Data Preprocessing

- Image normalization and resizing
- Parking space extraction and cropping
- Data augmentation (rotation, brightness, contrast adjustments)
- Class balancing techniques

---

## 🔬 Methodology

### Approach

Our solution employs a two-stage pipeline:

#### 1. **Parking Space Detection**

- Use object detection models (YOLO, Faster R-CNN, or SSD) to identify parking space boundaries
- Alternative: Manual annotation or template-based detection for fixed camera setups

#### 2. **Occupancy Classification**

- **Base Models**:
  - ResNet-50/101
  - VGG-16/19
  - EfficientNet
  - MobileNet (for edge deployment)
- **Custom CNN Architecture**:
  - Fine-tuned for binary classification (occupied/vacant)
  - Transfer learning from ImageNet pre-trained weights

#### 3. **Ensemble Methods** (Planned)

- Combine multiple models for improved accuracy
- Weighted voting or stacking techniques

### Performance Metrics

- Accuracy
- Precision and Recall
- F1-Score
- Confusion Matrix
- Inference Time (FPS)

---

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM

### Installation Steps

1. **Clone the repository**

```bash
git clone https://github.com/0x1AY/Parking-Lot-Occupancy-Estimation-.git
cd Parking-Lot-Occupancy-Estimation-
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download the dataset**

```bash
# Instructions to download and prepare the dataset
python scripts/download_dataset.py
```

5. **Configure settings**

```bash
# Edit config.yaml with your paths and hyperparameters
cp config.example.yaml config.yaml
```

---

## 🚀 How to Run the Code

### 1. Data Preparation

```bash
# Preprocess and split the dataset
python scripts/preprocess_data.py --data_path ./data/raw --output_path ./data/processed
```

### 2. Model Training

```bash
# Train the model with default settings
python train.py --config config.yaml

# Train with specific architecture
python train.py --model resnet50 --epochs 50 --batch_size 32
```

### 3. Model Evaluation

```bash
# Evaluate on test set
python evaluate.py --model_path ./checkpoints/best_model.pth --test_data ./data/processed/test
```

### 4. Inference

```bash
# Run inference on a single image
python inference.py --image_path ./test_images/parking_lot.jpg --model_path ./checkpoints/best_model.pth

# Run inference on a video
python inference.py --video_path ./test_videos/parking_lot.mp4 --model_path ./checkpoints/best_model.pth
```

### 5. Visualization

```bash
# Visualize results with occupancy overlay
python visualize_results.py --input ./results/predictions.json --output ./results/visualizations
```

---

## 📁 Project Structure

```
Parking-Lot-Occupancy-Estimation-/
│
├── README.md                 # This file
├── LICENSE                   # Project license
├── requirements.txt          # Python dependencies
├── config.yaml              # Configuration file
│
├── data/                    # Dataset directory
│   ├── raw/                 # Raw data
│   ├── processed/           # Processed data
│   └── README.md            # Data documentation
│
├── models/                  # Model architectures
│   ├── __init__.py
│   ├── resnet.py           # ResNet implementation
│   ├── vgg.py              # VGG implementation
│   ├── efficientnet.py     # EfficientNet implementation
│   └── custom_cnn.py       # Custom CNN architecture
│
├── scripts/                 # Utility scripts
│   ├── download_dataset.py # Dataset download script
│   ├── preprocess_data.py  # Data preprocessing
│   ├── augmentation.py     # Data augmentation utilities
│   └── visualize.py        # Visualization utilities
│
├── train.py                # Main training script
├── evaluate.py             # Model evaluation script
├── inference.py            # Inference script
├── visualize_results.py    # Results visualization
│
├── notebooks/              # Jupyter notebooks
│   ├── EDA.ipynb          # Exploratory Data Analysis
│   ├── model_experiments.ipynb
│   └── results_analysis.ipynb
│
├── checkpoints/            # Saved model checkpoints
├── logs/                   # Training logs
├── results/                # Experiment results
│   ├── predictions/
│   ├── visualizations/
│   └── metrics/
│
└── tests/                  # Unit tests
    ├── test_models.py
    ├── test_preprocessing.py
    └── test_inference.py
```

### Script Descriptions

#### `train.py`

Main training script that:

- Loads and preprocesses data
- Initializes the model architecture
- Implements training loop with validation
- Saves checkpoints and logs metrics
- Supports resume training from checkpoints

#### `evaluate.py`

Evaluation script that:

- Loads trained model
- Runs inference on test set
- Computes performance metrics
- Generates confusion matrix and classification report
- Saves evaluation results

#### `inference.py`

Inference script that:

- Loads pre-trained model
- Processes input images or videos
- Detects parking spaces
- Classifies occupancy status
- Outputs predictions with visualizations

#### `scripts/preprocess_data.py`

Data preprocessing module that:

- Handles data loading and validation
- Performs image resizing and normalization
- Splits data into train/val/test sets
- Applies data augmentation
- Saves processed data

#### `scripts/visualize.py`

Visualization utilities for:

- Plotting training curves
- Visualizing predictions on images
- Creating confusion matrices
- Generating occupancy heatmaps
- Saving comparison visualizations

#### `models/`

Contains model architecture implementations:

- **`resnet.py`**: ResNet variants (18, 34, 50, 101)
- **`vgg.py`**: VGG models (VGG16, VGG19)
- **`efficientnet.py`**: EfficientNet family
- **`custom_cnn.py`**: Custom lightweight architectures

---

## 📦 Dependencies

### Core Libraries

```
torch>=1.10.0
torchvision>=0.11.0
numpy>=1.21.0
opencv-python>=4.5.0
pillow>=8.3.0
```

### Data Processing

```
pandas>=1.3.0
scikit-learn>=0.24.0
albumentations>=1.0.0
```

### Visualization

```
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
```

### Training & Evaluation

```
tensorboard>=2.6.0
tqdm>=4.62.0
```

### Utilities

```
pyyaml>=5.4.0
python-dotenv>=0.19.0
```

For complete list, see `requirements.txt`

---

## 📈 Progress Summary

### ✅ Work Completed So Far

#### Phase 1: Project Setup & Literature Review (Weeks 1-2)

- ✅ Conducted comprehensive literature review on parking occupancy detection methods
- ✅ Identified and selected appropriate datasets (PKLot, CNRPark-EXT)
- ✅ Set up development environment and GitHub repository
- ✅ Defined project scope and methodology
- ✅ Created project proposal documentation

#### Phase 2: Data Collection & Preprocessing (Weeks 3-4)

- ✅ Downloaded and organized parking lot datasets
- ✅ Implemented data preprocessing pipeline
- ✅ Created data augmentation strategies
- ✅ Split dataset into train/validation/test sets
- ✅ Conducted exploratory data analysis (EDA)

#### Phase 3: Model Development (Week 5 - Current)

- ⏳ Implementing baseline CNN models
- ⏳ Setting up training infrastructure
- ⏳ Configuring experiment tracking with TensorBoard
- 🔄 Initial training runs in progress

---

## 📊 Preliminary Results

### Dataset Statistics

- **Total Images**: 12,417 images
- **Parking Spaces**: ~94,000 labeled spaces
- **Class Distribution**:
  - Occupied: 52.3%
  - Vacant: 47.7%
- **Camera Angles**: 3 different parking lots with varying perspectives
- **Weather Conditions**: Sunny (60%), Cloudy (25%), Rainy (15%)

### Initial Findings

_(Note: Training is in early stages)_

#### Baseline Model Performance (Preliminary)

| Model      | Accuracy | Precision | Recall | F1-Score | Notes              |
| ---------- | -------- | --------- | ------ | -------- | ------------------ |
| ResNet-50  | TBD      | TBD       | TBD    | TBD      | Currently training |
| VGG-16     | TBD      | TBD       | TBD    | TBD      | Next in queue      |
| Custom CNN | TBD      | TBD       | TBD    | TBD      | Planned            |

### Data Analysis Insights

1. **Lighting Variation**: Significant impact on model performance
2. **Occlusion Challenges**: Partially visible cars pose classification difficulties
3. **Shadow Effects**: Shadows can be misclassified as occupied spaces
4. **Class Balance**: Dataset is relatively balanced, reducing need for heavy resampling

### Visualizations

_(To be added as training progresses)_

- Training/Validation loss curves
- Sample predictions with ground truth
- Confusion matrices
- Occupancy heatmaps

---

## 📅 Updated Timeline / Milestones

### Phase 3: Model Training & Optimization (Weeks 5-7)

**Timeline:** Nov 4 - Nov 24, 2025

- [x] Week 5 (Nov 4-10): Baseline model training

  - Train ResNet-50, VGG-16, EfficientNet
  - Establish baseline performance metrics
  - Document training procedures

- [ ] Week 6 (Nov 11-17): Model refinement

  - Hyperparameter tuning
  - Implement learning rate scheduling
  - Apply regularization techniques
  - Cross-validation experiments

- [ ] Week 7 (Nov 18-24): Advanced techniques
  - Test ensemble methods
  - Implement attention mechanisms
  - Optimize for inference speed
  - Compare all model variants

### Phase 4: Evaluation & Analysis (Weeks 8-9)

**Timeline:** Nov 25 - Dec 8, 2025

- [ ] Week 8 (Nov 25-Dec 1): Comprehensive evaluation

  - Test on multiple datasets
  - Analyze failure cases
  - Compute detailed metrics
  - Create visualization dashboards

- [ ] Week 9 (Dec 2-8): Results analysis
  - Statistical significance testing
  - Performance benchmarking
  - Error analysis and insights
  - Prepare comparison tables

### Phase 5: Documentation & Deployment (Weeks 10-11)

**Timeline:** Dec 9 - Dec 20, 2025

- [ ] Week 10 (Dec 9-15): Documentation

  - Complete technical documentation
  - Create demo videos
  - Write final report
  - Prepare presentation materials

- [ ] Week 11 (Dec 16-20): Final submission
  - Code cleanup and commenting
  - Final testing and validation
  - Project presentation
  - Submit final deliverables

### Key Milestones

| Milestone                   | Target Date  | Status         |
| --------------------------- | ------------ | -------------- |
| Data preprocessing complete | Nov 3, 2025  | ✅ Complete    |
| Baseline models trained     | Nov 10, 2025 | 🔄 In Progress |
| Best model identified       | Nov 24, 2025 | ⏳ Pending     |
| Full evaluation complete    | Dec 1, 2025  | ⏳ Pending     |
| Final report submitted      | Dec 20, 2025 | ⏳ Pending     |

---

## 🎯 Next Steps

### Immediate Tasks (Next 2 Weeks)

1. **Complete Baseline Training**

   - Finish training ResNet-50, VGG-16, and EfficientNet variants
   - Record all training metrics and hyperparameters
   - Save best model checkpoints

2. **Hyperparameter Optimization**

   - Implement grid search or Bayesian optimization
   - Test different learning rates, batch sizes, and optimizers
   - Experiment with data augmentation strategies

3. **Develop Evaluation Framework**
   - Create comprehensive evaluation scripts
   - Implement visualization tools for results
   - Set up automated testing pipeline

### Medium-Term Goals (Weeks 3-4)

4. **Advanced Model Techniques**

   - Implement attention mechanisms (CBAM, SE-Net)
   - Test multi-scale feature extraction
   - Experiment with ensemble methods

5. **Real-World Testing**

   - Test on unseen parking lots
   - Evaluate robustness to different conditions
   - Analyze computational requirements

6. **Performance Optimization**
   - Model quantization for faster inference
   - Pruning for reduced model size
   - TensorRT or ONNX optimization

### Long-Term Goals (Weeks 5-6)

7. **Deployment Preparation**

   - Create REST API for model serving
   - Develop simple web interface
   - Documentation for deployment

8. **Final Documentation**

   - Write comprehensive technical report
   - Create user guide and API documentation
   - Prepare presentation materials

9. **Future Enhancements**
   - Multi-camera fusion
   - Temporal analysis for better accuracy
   - Integration with parking management systems

---

## 🔍 Research Questions

### To Be Investigated

1. How does model performance vary across different weather conditions?
2. What is the optimal balance between accuracy and inference speed?
3. Can transfer learning from general object detection improve results?
4. How does the system perform with varying parking lot layouts?
5. What minimum resolution is required for reliable classification?

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**Aminu Yiwere**

- GitHub: [@0x1AY](https://github.com/0x1AY)
- Project Link: [https://github.com/0x1AY/Parking-Lot-Occupancy-Estimation-.git](https://github.com/0x1AY/Parking-Lot-Occupancy-Estimation-.git)

---

## 🙏 Acknowledgments

- PKLot Dataset creators at UFPR
- Deep Learning course instructors and TAs
- PyTorch and OpenCV communities
- Research papers that inspired this work

---

## 📚 References

1. de Almeida, P. R., et al. (2015). "PKLot–A robust dataset for parking lot classification." _Expert Systems with Applications_, 42(11), 4937-4949.

2. Amato, G., et al. (2017). "Deep learning for decentralized parking lot occupancy detection." _Expert Systems with Applications_, 72, 327-334.

3. He, K., et al. (2016). "Deep residual learning for image recognition." _CVPR_.

4. Redmon, J., & Farhadi, A. (2018). "YOLOv3: An incremental improvement." _arXiv preprint_.

5. Tan, M., & Le, Q. (2019). "EfficientNet: Rethinking model scaling for convolutional neural networks." _ICML_.

---

## 📝 Notes

- This is a work in progress for a Deep Learning course project
- Regular updates will be pushed to the repository as the project progresses
- For questions or issues, please open an issue on GitHub

---

**Last Updated:** November 4, 2025  
**Project Status:** 🔄 In Progress - Baseline Model Training Phase
