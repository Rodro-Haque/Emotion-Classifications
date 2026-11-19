import torch


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_true = []
    all_logits = []

    for batch in loader:
        text = batch["text"].to(device)
        audio = batch["audio"].to(device)
        visual = batch["visual"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(text, audio, visual)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        all_true.append(labels.detach().cpu())
        all_logits.append(logits.detach().cpu())

    avg_loss = total_loss / len(loader.dataset)
    y_true = torch.cat(all_true, dim=0).numpy()
    y_logits = torch.cat(all_logits, dim=0).numpy()
    return avg_loss, y_true, y_logits


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_true = []
    all_logits = []

    with torch.no_grad():
        for batch in loader:
            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            visual = batch["visual"].to(device)
            labels = batch["labels"].to(device)

            logits = model(text, audio, visual)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            all_true.append(labels.detach().cpu())
            all_logits.append(logits.detach().cpu())

    avg_loss = total_loss / len(loader.dataset)
    y_true = torch.cat(all_true, dim=0).numpy()
    y_logits = torch.cat(all_logits, dim=0).numpy()
    return avg_loss, y_true, y_logits
