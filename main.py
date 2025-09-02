import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from spikingjelly.clock_driven import neuron, encoding, functional, layer
from torch import nn
from tqdm import tqdm

# Load MNIST dataset
def get_mnist_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# Poisson encoding for spike trains
def poisson_encode(images, time_steps=20):
    encoder = encoding.PoissonEncoder()
    # images: [batch, 1, 28, 28]
    images = images.view(images.size(0), -1)
    spike_trains = []
    for t in range(time_steps):
        spike_trains.append(encoder(images))
    spike_trains = torch.stack(spike_trains)  # shape: [time_steps, batch, input_dim]
    return spike_trains

#  Define SNN model
class SimpleSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.lif1 = neuron.LIFNode()
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.lif2 = neuron.LIFNode()
        self.fc1 = nn.Linear(32*28*28, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.lif1(x)
        x = self.conv2(x)
        x = self.lif2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# Training and evaluation functions will be added next.

if __name__ == "__main__":
    train_loader, test_loader = get_mnist_loaders()
    print("MNIST loaders ready.")
    # Next: training loop, evaluation, and plotting

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleSNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    num_epochs = 3
    time_steps = 20
    train_accs = []
    test_accs = []

    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            # Poisson encode images for all time steps
            spike_trains = poisson_encode(images, time_steps)  # shape: [time_steps, batch, input_dim]
            out_spikes = 0
            functional.reset_net(model)
            for t in range(time_steps):
                # Reshape spike_trains[t] to [batch, 1, 28, 28]
                input_t = spike_trains[t].view(images.size(0), 1, 28, 28)
                out = model(input_t)
                out_spikes += out
            out_spikes = out_spikes / time_steps
            loss = loss_fn(out_spikes, labels)
            loss.backward()
            optimizer.step()
            _, predicted = out_spikes.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        train_acc = correct / total
        train_accs.append(train_acc)

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                spike_trains = poisson_encode(images, time_steps)
                out_spikes = 0
                functional.reset_net(model)
                for t in range(time_steps):
                    input_t = spike_trains[t].view(images.size(0), 1, 28, 28)
                    out = model(input_t)
                    out_spikes += out
                out_spikes = out_spikes / time_steps
                _, predicted = out_spikes.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        test_acc = correct / total
        test_accs.append(test_acc)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

    # Plot learning curve
    plt.figure()
    plt.plot(range(1, num_epochs+1), train_accs, label='Train Acc')
    plt.plot(range(1, num_epochs+1), test_accs, label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve')
    plt.legend()
    plt.show()
