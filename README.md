# CIFAR-10 Image Classification

This project implements a Convolutional Neural Network (CNN) for classifying images from the CIFAR-10 dataset. The model architecture includes various modern CNN techniques like depthwise separable convolutions, dilated convolutions, and global average pooling.

## Project Structure 
```
├── config.yml # Configuration parameters
├── data.py # Data loading and augmentation
├── main.py # Main training script
├── model.py # CNN architecture definition
├── test.py # Model evaluation
├── train.py # Training loop
└── README.md # This file
```

## Features

- **Data Augmentation**: Uses Albumentations library with:
  - Horizontal flips
  - Shift, scale, and rotate transforms
  - Coarse dropout
  - Grayscale conversion
  - Normalization

- **Model Architecture**:
  - Input block with initial convolutions
  - Multiple convolutional blocks with different techniques
  - Depthwise separable convolutions
  - Dilated convolutions
  - Global Average Pooling
  - Batch normalization and dropout for regularization

- **Training**:
  - SGD optimizer with momentum
  - Learning rate scheduling with ReduceLROnPlateau
  - Early stopping when accuracy reaches 86%
  - Progress tracking with tqdm
  - Metrics logging to JSON file

## Requirements

- PyTorch
- torchvision
- albumentations
- PyYAML
- tqdm
- torchsummary

## Usage

1. Configure the parameters in `config.yml` as needed.

2. Run the training:
```bash
python main.py
```

3. The training progress will be displayed in the console, and final metrics will be saved to `metrics.json`.

## Configuration

See `config.yml` for all available parameters.

## Model Architecture

The CNN architecture consists of:
1. Input block (3→32 channels)
2. Three main blocks with transition layers
3. Final convolution block
4. Global Average Pooling
5. Output layer (10 classes)

Each block includes batch normalization, ReLU activation, and dropout for regularization.

## Performance

The model is trained to achieve 86% accuracy on the CIFAR-10 test set, at which point training stops automatically.
