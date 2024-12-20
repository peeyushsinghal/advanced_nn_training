from torchsummary import summary
import torch
import torch.optim as optim
import yaml
import json
from torch.optim.lr_scheduler import ReduceLROnPlateau


from data import CIFAR10Data
from model import Network as Net
from train import train_model
from test import test_model


def get_device() -> torch.device:
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    return device


def get_data() -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    obj_data = CIFAR10Data()
    train_loader = obj_data.get_train_loader()
    test_loader = obj_data.get_test_loader()
    return train_loader, test_loader


def get_model(device: torch.device, requires_summary: bool = True) -> torch.nn.Module:
    model = Net()
    if requires_summary:
        summary(model, input_size=(3, 32, 32))
    return model.to(device)


def load_config() -> dict:
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    device = get_device()
    train_loader, test_loader = get_data()

    model = get_model(device)
    print(f"\n Number of parameters: {sum(p.numel() for p in model.parameters())}")
    config = load_config()

    optimizer = optim.SGD(
        model.parameters(),
        lr=config["learning_rate"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=float(config["factor"]),
        patience=int(config["patience"]),
        threshold=float(config["threshold"]),
        verbose=bool(config["verbose"]),
    )

    list_train_accuracy = []
    list_test_accuracy = []
    for epoch in range(config["epochs"]):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        model, epoch_train_metric = train_model(model, device, train_loader, optimizer)
        model, epoch_test_metric = test_model(model, device, test_loader)

        list_train_accuracy.append(float(epoch_train_metric["accuracy"]))
        list_test_accuracy.append(float(epoch_test_metric["accuracy"]))

    with open("metrics.json", "w") as f:
        json.dump(
            {
                "train_accuracy_over_epochs": list_train_accuracy,
                "test_accuracy_over_epochs": list_test_accuracy,
            },
            f,
        )
