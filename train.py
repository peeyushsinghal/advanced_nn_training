import torch.nn.functional as F
from tqdm import tqdm
import torch


def train_model(
    model: torch.nn.Module,
    device: str,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
) -> tuple[torch.nn.Module, dict]:
    model.train()
    pbar = tqdm(train_loader)
    train_loss = 0
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(
            desc=f"loss={loss.item():.4f} | accuracy={100*correct/processed:0.2f}%"
        )

    return model, {"train_loss": round(train_loss, 4), "accuracy": round(100*correct/processed, 2)}
