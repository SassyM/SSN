# Spiking Neural Network for MNIST Digit Classification

This project implements a Spiking Neural Network (SNN) for classifying handwritten digits from the MNIST dataset using the [SpikingJelly](https://github.com/fangwei123456/spikingjelly) library (PyTorch-based).

## Features
- Loads and preprocesses the MNIST dataset
- Encodes images into spike trains using Poisson encoding
- Defines a simple SNN: 2 convolutional layers + spiking neurons + fully connected output
- Trains the network with surrogate gradient descent
- Evaluates classification accuracy
- Plots learning curves

## Requirements
- Python 3.10+
- PyTorch
- SpikingJelly
- torchvision
- matplotlib
- tqdm

Install dependencies:
```bash
pip install torch torchvision spikingjelly matplotlib tqdm
```

## Usage
Run the main script:
```bash
python main.py
```

## Project Structure
- `main.py`: Main training and evaluation script
- `README.md`: Project overview and instructions

## How It Works
1. **Data Loading**: MNIST images are loaded and normalized.
2. **Poisson Encoding**: Images are converted to spike trains for SNN input.
3. **Model**: A simple SNN is defined using SpikingJelly's LIF neurons.
4. **Training**: The model is trained using surrogate gradient descent.
5. **Evaluation**: Accuracy is measured on the test set.
6. **Plotting**: Learning curves are displayed after training.

## References
- [SpikingJelly Documentation](https://spikingjelly.readthedocs.io/en/latest/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

## License
This project is for educational purposes.
