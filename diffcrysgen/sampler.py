
import torch
import numpy as np
import tqdm
import time
from diffcrysgen.model import Model, UNet
from diffcrysgen.utils import *


# Adapted from "Elucidating the Design Space of Diffusion-Based
# Generative Models" by Karras et al.


def round_sigma(sigma):
    return torch.as_tensor(sigma)


def generate_samples(num_samples: int = 100, batch_size: int = 1000, model_path: str = "assets/saved-model/sdm.pt"):
    """
    Generate IRCR samples using our pre-trained version.
    
    Args:
        num_samples (int): Total number of samples to generate.
        batch_size (int): Batch size for sampling.
        model_path (str): Path to the saved model (.pt file).

    Returns:
        np.ndarray: Generated IRCR array of shape (num_samples, F, C)
    """
    # Load scaler
    scaler = load_saved_diffusion_scaler()
    print("Scaler loaded.")

    # Load device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load the model
    denoise_fn = UNet(in_c=3, out_c=3, time_emb_dim=256).to(device)
    model = Model(denoise_fn=denoise_fn).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print("Model loaded from:", model_path)

    # Sampling hyperparameters
    num_steps = 100
    sigma_min = 0.002
    sigma_max = 80
    rho = 7
    S_churn = 1
    S_min = 0
    S_max = float("inf")
    S_noise = 1

    net = model.denoise_fn_D
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time steps
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # append t_N = 0

    img_channels = 3
    img_resolution = 136

    total_loops = int(np.ceil(num_samples / batch_size))
    final_data = []

    start_time = time.time()

    with torch.no_grad():
        for loop in range(total_loops):
            curr_batch_size = min(batch_size, num_samples - loop * batch_size)
            latents = torch.randn([curr_batch_size, img_channels, img_resolution], device=device)
            x_next = latents * t_steps[0]

            # Main Sampling Loop
            for i, (t_cur, t_next) in tqdm.tqdm(list(enumerate(zip(t_steps[:-1], t_steps[1:]))), desc=f"Sampling {loop+1}/{total_loops}", leave=False):
                x_cur = x_next

                # Increase Noise temporarily
                gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
                t_hat = round_sigma(t_cur + gamma * t_cur)

                t_cur = t_cur.repeat(curr_batch_size)
                t_next = t_next.repeat(curr_batch_size)
                t_hat = t_hat.repeat(curr_batch_size)

                x_hat = x_cur + (t_hat[:, None, None] ** 2 - t_cur[:, None, None] ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

                denoised = net(x_hat, t_hat).float()
                d_cur = (x_hat - denoised) / t_hat[:, None, None]
                x_next = x_hat + (t_next[:, None, None] - t_hat[:, None, None]) * d_cur

                if i < num_steps - 1:
                    denoised = net(x_next, t_next).float()
                    d_prime = (x_next - denoised) / t_next[:, None, None]
                    x_next = x_hat + (t_next[:, None, None] - t_hat[:, None, None]) * (0.5 * d_cur + 0.5 * d_prime)

            x_next = x_next.permute(0, 2, 1).cpu().numpy()  # shape: (batch, F, C)
            x_next = (x_next + 1) / 2
            x_next = inv_minmax(x_next, scaler)
            final_data.append(x_next)

            print(f"Sampling batch {loop + 1}/{total_loops} done.")

    elapsed = (time.time() - start_time) / 60
    print(f"Total sampling time: {elapsed:.2f} minutes")

    generated_array = np.concatenate(final_data, axis=0)[:num_samples]
    print("Generated IRCR shape:", generated_array.shape)
    return generated_array



