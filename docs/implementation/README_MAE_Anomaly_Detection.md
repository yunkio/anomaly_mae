# Self-Distilled Masked Autoencoder for Time Series Anomaly Detection

This implementation provides a complete pipeline for semi-supervised anomaly detection in time series data using a Self-Distilled Masked Autoencoder (MAE) architecture.

## Overview

The model uses a unique teacher-student distillation approach where:
- **Teacher Decoder** (deep): Learns to reconstruct the original signal
- **Student Decoder** (shallow): Mimics the teacher for normal data
- **Anomaly Detection**: Measured by the discrepancy between teacher and student outputs

## Features

✅ **Synthetic Dataset Generation** with 5 anomaly types:
- Spike: Sudden short duration peaks
- Noise: Increased variance (Gaussian noise)
- Drift: Gradual shift in mean
- Trend: Linear increase/decrease
- Frequency: Change in periodicity

✅ **Transformer-based Architecture**:
- Shared encoder with positional encoding
- Deep teacher decoder (4 layers)
- Shallow student decoder (1 layer)
- Random masking mechanism (60% default)

✅ **Custom Loss Function**:
- Reconstruction loss (MSE on masked positions)
- Discrepancy loss with margin-based negative learning
- Warm-up strategy for stable training

✅ **Comprehensive Evaluation**:
- Precision, Recall, F1-Score
- ROC-AUC with optimal threshold
- Visual reconstructions and score distributions

## Installation

### 1. Install PyTorch

For CPU:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

For CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For CUDA 12.1:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. Install Other Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install numpy matplotlib scikit-learn tqdm
```

## Usage

### Quick Start

Run the complete pipeline:
```bash
python self_distilled_mae_anomaly_detection.py
```

This will:
1. Generate synthetic training and test datasets
2. Train the Self-Distilled MAE model (100 epochs)
3. Evaluate on test set
4. Generate visualization plots

### Expected Output

The script will create the following files:
- `training_history.png` - Loss curves during training
- `roc_curve.png` - ROC curve with AUC score
- `score_distribution.png` - Anomaly score distribution
- `sample_reconstruction_*.png` - Sample reconstructions

### Configuration

Modify the `Config` class in the script to customize:

```python
@dataclass
class Config:
    # Data parameters
    seq_length: int = 100
    train_anomaly_ratio: float = 0.03  # 3% labeled anomalies
    test_anomaly_ratio: float = 0.25   # 25% anomalies

    # Model parameters
    d_model: int = 64
    masking_ratio: float = 0.6  # 60% masking

    # Training parameters
    num_epochs: int = 100
    learning_rate: float = 1e-3
    warmup_epochs: int = 20

    # Loss parameters
    margin: float = 1.0
    lambda_disc: float = 0.5
```

## Architecture Details

### Model Flow

```
Input Sequence (batch, seq_len, 1)
    ↓
Input Projection → (batch, seq_len, d_model)
    ↓
Random Masking (60% masked)
    ↓
Positional Encoding
    ↓
Transformer Encoder (3 layers)
    ↓
Latent Representation
    ├─→ Teacher Decoder (4 layers) → Teacher Output
    └─→ Student Decoder (1 layer)  → Student Output
```

### Loss Function

**Total Loss:**
```
L_total = L_rec + λ * L_disc
```

**Reconstruction Loss:**
```
L_rec = MSE(Teacher_Output, Original_Input) on masked positions
```

**Discrepancy Loss:**
```
d(x) = ||Teacher(x) - Student(x)||²

For normal data (y=0):     L_disc += d(x)
For anomaly data (y=1):    L_disc += max(0, margin - d(x))
```

### Warm-up Strategy

- **Epochs 1-20**: Gradually increase anomaly divergence weight from 0 to 1
- **Epochs 21-100**: Full weight on anomaly divergence
- This prevents mode collapse and ensures stable training

### Inference Strategy

During inference:
1. Mask the last N tokens (default: 10)
2. Compute teacher and student reconstructions
3. Calculate discrepancy on masked positions
4. Higher discrepancy = Higher anomaly score

## Code Structure

```
self_distilled_mae_anomaly_detection.py
├── Config                          # Hyperparameter configuration
├── TimeSeriesAnomalyDataset       # Synthetic data generation
├── PositionalEncoding             # Transformer positional encoding
├── SelfDistilledMAE               # Main model architecture
├── SelfDistillationLoss           # Custom loss function
├── Trainer                        # Training loop with warm-up
├── Evaluator                      # Inference and evaluation
└── main()                         # Main execution pipeline
```

## Key Implementation Details

### Type Hints
All functions include Python type hints for better code clarity:
```python
def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    ...
```

### Modular Design
Each component is separated into its own class for easy modification and testing.

### Reproducibility
Random seed is set globally for reproducible results:
```python
set_seed(42)
```

### Visualization
Includes comprehensive plotting functions:
- Training loss curves
- ROC curves
- Score distributions
- Sample reconstructions with discrepancy

## Performance Tips

1. **GPU Acceleration**: Automatically uses CUDA if available
2. **Batch Size**: Increase if you have more GPU memory
3. **Sequence Length**: Longer sequences may capture better patterns
4. **Masking Ratio**: Try 40-75% for different reconstruction difficulties
5. **Warm-up Epochs**: Adjust based on training stability

## Troubleshooting

### Out of Memory
- Reduce `batch_size` (default: 32)
- Reduce `d_model` (default: 64)
- Reduce `seq_length` (default: 100)

### Poor Performance
- Increase `num_epochs` (default: 100)
- Adjust `lambda_disc` weight (default: 0.5)
- Increase `warmup_epochs` (default: 20)
- Try different `margin` values (default: 1.0)

### Training Instability
- Increase `warmup_epochs`
- Reduce `learning_rate`
- Add gradient clipping (already implemented)

## Citation

If you use this implementation, please cite:

```
Self-Distilled Masked Autoencoder for Semi-Supervised Time Series Anomaly Detection
Implementation based on MAE and self-distillation principles
```

## License

MIT License - Feel free to use and modify for your research and applications.

## Contact

For questions or issues, please open an issue on the repository.
