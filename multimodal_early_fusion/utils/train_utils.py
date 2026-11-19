import torch

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    y_true = []
    y_pred = []

    for batch in loader:
        text = batch["text"].to(device)
        audio = batch["audio"].to(device)
        visual = batch["visual"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(text, audio, visual)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, y_true, y_pred

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in loader:
            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            visual = batch["visual"].to(device)
            labels = batch["label"].to(device)

            logits = model(text, audio, visual)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, y_true, y_pred