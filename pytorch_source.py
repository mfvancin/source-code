# Basic Concepts
# Tensors
import torch

# Create a 1D tensor
tensor_1d = torch.tensor([1, 2, 3])
print(tensor_1d)

# Create a 2D tensor (matrix)
tensor_2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(tensor_2d)

# Create a random tensor
random_tensor = torch.rand(2, 3)  # 2x3 matrix
print(random_tensor)

# Basic Operations
# Tensor addition
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
result = a + b
print(result)

# Element-wise multiplication
result_mul = a * b
print(result_mul)

# Matrix multiplication
matrix_a = torch.tensor([[1, 2], [3, 4]])
matrix_b = torch.tensor([[5, 6], [7, 8]])
matrix_result = torch.matmul(matrix_a, matrix_b)
print(matrix_result)

# Building a Simple Neural Network
# Load MNIST Dataset
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

# Transform to convert images to tensors
transform = transforms.Compose([transforms.ToTensor()])

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Define a Neural Network Model
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Input layer
        self.fc2 = nn.Linear(128, 64)        # Hidden layer
        self.fc3 = nn.Linear(64, 10)         # Output layer

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = F.relu(self.fc1(x))  # Apply ReLU activation
        x = F.relu(self.fc2(x))  # Apply ReLU activation
        x = self.fc3(x)          # Output layer
        return x

# Instantiate the model
model = SimpleNN()
print(model)

# Training the Model
import torch.optim as optim

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    for images, labels in train_loader:
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluating the Model
# Load test dataset
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Evaluate the model
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')

# Convolutional Neural Networks (CNNs)
# Building a CNN in PyTorch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # Convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # Another convolutional layer
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, 10)           # Output layer

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Apply first convolution
        x = F.max_pool2d(x, 2)     # Max pooling
        x = F.relu(self.conv2(x))  # Apply second convolution
        x = F.max_pool2d(x, 2)     # Max pooling
        x = x.view(-1, 64 * 7 * 7)  # Flatten the tensor
        x = F.relu(self.fc1(x))     # Fully connected layer
        x = self.fc2(x)              # Output layer
        return x

# Instantiate and print the CNN model
cnn_model = SimpleCNN()
print(cnn_model)

# Recurrent Neural Networks (RNNs)
# Building an RNN in PyTorch
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)  # RNN layer
        self.fc = nn.Linear(hidden_size, output_size)  # Output layer

    def forward(self, x):
        out, _ = self.rnn(x)  # Forward pass through RNN
        out = self.fc(out[:, -1, :])  # Get the output from the last time step
        return out

# Example usage
input_size = 10  # Input feature size
hidden_size = 20  # Hidden state size
output_size = 1   # Output size

rnn_model = SimpleRNN(input_size, hidden_size, output_size)
print(rnn_model)

# Transfer Learning
# Using a Pre-trained Model
import torchvision.models as models

# Load a pre-trained ResNet model
model = models.resnet18(pretrained=True)

# Replace the last layer for your specific task
num_classes = 10  # Adjust this for your dataset
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Set model to training mode
model.train()

# Custom Datasets and Data Loaders
# Creating a Custom Dataset
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path)
        label = self.images[idx].split('_')[0]  # Example: label from filename

        if self.transform:
            image = self.transform(image)

        return image, label

# Usage
dataset = CustomDataset(image_dir='./data/images')
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model Saving and Loading
# Saving
torch.save(model.state_dict(), 'model.pth')

# Loading
model = SimpleCNN()  # Reinitialize the model
model.load_state_dict(torch.load('model.pth'))
model.eval()  # Set the model to evaluation mode

# Distributed Training
# Using torch.nn.DataParallel
# Assuming you have multiple GPUs
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

model.to('cuda')  # Move model to GPU

# Using Distributed Data Parallel (DDP)
#DDP is more efficient for multi-GPU training. Here’s a high-level view:
#Initialize the process group.
#Wrap your model with torch.nn.parallel.DistributedDataParallel.

# Hyperparameter Tuning
# Manual Tuning Example
learning_rates = [0.001, 0.01, 0.1]
for lr in learning_rates:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Train the model and evaluate performance

# Advanced Convolutional Neural Networks (CNNs)
#Common CNN Architectures
#VGGNet: A very deep CNN architecture that uses small 3x3 convolution filters.
#Inception (GoogLeNet): Uses multiple filter sizes at the same layer to capture information at various scales.
#ResNet: Introduces skip connections (or residual connections) to combat the vanishing gradient problem in very deep networks.
# Implementing a Deeper CNN (ResNet Example)
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Skip connection
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1])
        self.layer3 = self._make_layer(block, 256, num_blocks[2])
        self.layer4 = self._make_layer(block, 512, num_blocks[3])
        
        self.fc = nn.Linear(512, 10)  # Adjust for your output classes

    def _make_layer(self, block, out_channels, blocks):
        layers = []
        for _ in range(blocks):
            layers.append(block(self.in_channels, out_channels))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

# Example: Create a ResNet with 3 blocks in each layer
resnet_model = ResNet(BasicBlock, [2, 2, 2, 2])
print(resnet_model)

# Long Short-Term Memory Networks (LSTMs)
# Building an LSTM in PyTorch
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)  # Output from LSTM
        out = self.fc(out[:, -1, :])  # Use the last time step's output
        return out

# Example usage
input_size = 10
hidden_size = 20
num_layers = 2
output_size = 1

lstm_model = LSTMModel(input_size, hidden_size, num_layers, output_size)
print(lstm_model)

# Attention Mechanisms and Transformers
# Implementing a Simple Attention Layer

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        weights = F.softmax(scores, dim=1)
        return torch.bmm(weights.permute(0, 2, 1), keys)  # Weighted sum of keys

# Example usage
attention_layer = AttentionLayer(hidden_size)

# Using Transformers
class TransformerModel(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                           num_encoder_layers=num_encoder_layers,
                                           num_decoder_layers=num_decoder_layers)
        self.fc_out = nn.Linear(d_model, output_size)

    def forward(self, src, tgt):
        out = self.transformer(src, tgt)
        return self.fc_out(out)

# Example usage
transformer_model = TransformerModel(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6)
print(transformer_model)

# Custom Loss Functions
# Implementing a Custom Loss Function

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, output, target):
        return torch.mean((output - target) ** 2)  # Mean Squared Error

# Example usage
custom_loss = CustomLoss()

# Gradient Clipping
# Implementing Gradient Clipping

# Training loop example
for _ in range(num_epochs):
    optimizer.zero_grad()  # Reset gradients
    outputs = model(inputs)  # Forward pass
    loss = criterion(outputs, labels)  # Compute loss
    loss.backward()  # Backpropagation
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients
    optimizer.step()  # Update parameters

# Data Augmentation Techniques
# Using torchvision for Data Augmentation

from torchvision import transforms

# Define transformations
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

# Apply the transformations in your dataset class
class AugmentedDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image

# Mixed Precision Training
# Implementing Mixed Precision Training
from torch.cuda.amp import GradScaler, autocast

# Initialize the GradScaler
scaler = GradScaler()

# Training loop
for _ in range(num_epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()  # Reset gradients

        with autocast():  # Enable mixed precision
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()  # Scale the loss
        scaler.step(optimizer)  # Step the optimizer
        scaler.update()  # Update the scale for the next iteration


