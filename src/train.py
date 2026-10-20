import torch
import torch.nn.functional as F


def combined_loss(out, mask, class_weights, smooth=1e-6):
    # part 1: weighted cross-entropy - checks every pixel
    ce_loss = F.cross_entropy(out, mask, weight=class_weights)

    # part 2: dice loss - checks how well
    probs = torch.softmax(out, dim=1)
    mask_onehot = F.one_hot(mask, num_classes=5).permute(0, 3, 1, 2).float()

    overlap = (probs * mask_onehot).sum(dim=(0, 2, 3))
    total = probs.sum(dim=(0, 2, 3)) + mask_onehot.sum(dim=(0, 2, 3))
    dice_per_class = (2 * overlap + smooth) / (total + smooth)

    present = mask_onehot.sum(dim=(0, 2, 3)) > 0
    dice_loss = 1 - dice_per_class[present].mean()

    return ce_loss + dice_loss


def train_one_epoch(model, loader, optimizer, class_weights, device):
    model.train()
    total_loss = 0.0

    for pre, post, mask in loader:
        pre = pre.to(device)
        post = post.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        out = model(pre, post)
        loss = combined_loss(out, mask, class_weights)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, class_weights, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for pre, post, mask in loader:
            pre = pre.to(device)
            post = post.to(device)
            mask = mask.to(device)

            out = model(pre, post)
            loss = combined_loss(out, mask, class_weights)

            total_loss += loss.item()

    return total_loss / len(loader)