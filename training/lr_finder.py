import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def lr_finder(model, dataloader, optimizer, device, 
              start_lr=1e-7, end_lr=1, num_iter=100, beta=0.98, save_path="lr_finder_plot.png"):
    """
    Gradually increases the learning rate between start_lr and end_lr over num_iter batches.
    Logs the loss and plots loss vs LR to find optimal LR.
    """
    model.train()
    lr_mult = (end_lr / start_lr) ** (1 / num_iter)
    lr = start_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    avg_loss = 0.0
    best_loss = float('inf')
    losses = []
    lrs = []

    iterator = iter(dataloader)
    for iteration in range(num_iter):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)

        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        metadata = batch["metadata"].to(device) if batch["metadata"] is not None else None

        # Forward pass
        optimizer.zero_grad()
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            metadata=metadata,
        )

        loss = F.cross_entropy(logits, labels)
        # Compute running average of loss
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** (iteration + 1))

        # Record the LR and loss
        lrs.append(lr)
        losses.append(smoothed_loss)

        # Stop if the loss explodes
        if iteration > 10 and smoothed_loss > 4 * best_loss:
            print(f"[LR Finder] Loss diverged at iteration {iteration}. Stopping early.")
            break

        if smoothed_loss < best_loss or iteration == 0:
            best_loss = smoothed_loss

        # Backward and optimize
        loss.backward()
        optimizer.step()

        # Update LR
        lr *= lr_mult
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # Plot LR vs Loss
    plt.figure(figsize=(8, 5))
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel("Learning Rate (log scale)")
    plt.ylabel("Smoothed Loss")
    plt.title("LR Finder: Loss vs Learning Rate")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

    # Print best LR suggestion
    min_loss_idx = np.argmin(losses)
    best_lr = lrs[min_loss_idx] / 10  # Slightly before minimum
    print(f"[LR Finder] Suggested max_lr: {best_lr:.2e} (just before loss minimum)")
    return best_lr, lrs, losses
