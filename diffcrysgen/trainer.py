

import time
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau as RLROP
from tqdm.auto import tqdm
from model import UNet, Model
from utils import minmax

def load_data(train_path, test_path, batch_size=64):
    # Original IRCR has shape [batch,142,3] : IRCR={E,L,C,O,P}
    # We are not considering property matrix (P) for unconditional generation.
    train_data = np.load(train_path)[:, :136, :]
    test_data = np.load(test_path)[:, :136, :]

    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    # Normalize [0, 1]
    train_scaled, train_scaler = minmax(train_data)
    test_scaled, test_scaler = minmax(test_data)

    # Normalize to [-1, 1]
    train_scaled = 2 * train_scaled - 1
    test_scaled = 2 * test_scaled - 1

    # Convert to torch tensors (N, 3, F)
    train_tensor = torch.from_numpy(train_scaled).float().permute(0, 2, 1)
    test_tensor = torch.from_numpy(test_scaled).float().permute(0, 2, 1)

    print(f"Transformed training shape: {train_tensor.shape}")
    print(f"Transformed test shape: {test_tensor.shape}")
    print(f"Train value range: {train_tensor.min().item()} to {train_tensor.max().item()}")

    train_loader = DataLoader(train_tensor, batch_size=batch_size)
    test_loader = DataLoader(test_tensor, batch_size=batch_size)

    return train_loader, test_loader, train_scaler, test_scaler


def train_diffcrysgen(
    train_loader,
    test_loader,
    num_epochs=1000,
    save_path="sdm.pt",
    initial_lr=5e-4,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    denoise_fn = UNet(in_c=3, out_c=3, time_emb_dim=256).to(device)
    model = Model(denoise_fn=denoise_fn).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=0)
    scheduler = RLROP(optimizer, mode="min", factor=0.3, patience=50, verbose=True)

    best_loss = float("inf")
    train_losses = []
    test_losses = []
    lrs = []

    start_time = time.time()

    for epoch in tqdm(range(num_epochs), desc="Training Progress", colour="green"):
        # Manual LR scheduling based on epochs
        if epoch == 500:
            for group in optimizer.param_groups:
                group["lr"] = 1e-4
        elif epoch == 800:
            for group in optimizer.param_groups:
                group["lr"] = 5e-5

        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1}/{num_epochs}", colour="blue"):
            batch = batch.to(device)
            loss = model(batch)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, leave=False, desc=f"Epoch {epoch+1}/{num_epochs}", colour="blue"):
                batch = batch.to(device)
                loss = model(batch)
                test_loss += loss.item()
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
 
        # step the scheduler
        # scheduler.step(test_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        lrs.append(current_lr)

        log_msg = f"Epoch {epoch+1}: train_loss: {train_loss:.4f}, test_loss: {test_loss:.4f}, lr: {current_lr:.6f}"
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), save_path)
            log_msg += " --> Best model ever (stored)"
        print(log_msg)

        # scheduler.step(val_loss)

    print(f"Training completed in {(time.time() - start_time):.2f}s")

    # Save epoch-wise loss
    results_df = pd.DataFrame({
        "epoch": np.arange(num_epochs),
        "train_loss": train_losses,
        "test_loss": test_losses,
        "lr": lrs
    })
    results_df.to_csv("epoch-loss.csv", index=False)
    print(results_df.tail())

