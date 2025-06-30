import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision.transforms.functional import resize
from datetime import datetime
import torch.optim as optim
from skimage import data
import matplotlib.pyplot as plt
import numpy as np

from synthesis import WaveletImageSynthesis
from utils import *
from analysis import pngs_to_mp4
from schedulers import *


if __name__ == "__main__":

    # Configuration
    num_epochs = 2000000
    analyze_every = 100
    shrink_scale = 16
    learning_rate = 0.005

    spatial_depth = 8
    scale_depth = 32

    # Time stamp for results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.getcwd(), "trained", timestamp)
    recons_dir = os.path.join(results_dir, "recons")
    hists_dir = os.path.join(results_dir, "hists")
    slices_dir = os.path.join(results_dir, "slices")
    for p in [results_dir, recons_dir, hists_dir, slices_dir]:
        os.makedirs(p, exist_ok=True)

    # Get device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load image
    img = read_image("./images/uv_face.jpg") / 255.0
    _, img_H, img_W = img.shape

    # Create target image
    target = resize(img, (img_H // shrink_scale, img_W // shrink_scale)).unsqueeze(0).to(device)

    # Resize image to match target
    img_small = target.clone()

    # Define the model, loss function, and optimizer
    model     = WaveletImageSynthesis(spatial_depth=spatial_depth, scale_depth=scale_depth).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = LINEAR_WARMUP_HOLDON_COOLDOWN(
        optimizer=optimizer,
        warmup_steps=2000,
        holdon_steps=20000,
        cooldown_steps=20000,
        max_scale=1.0, min_scale=0.1
    )

    # Number of learnable parameters
    num_params = model.num_params()

    # Number of pixels in target image
    num_pixels = np.prod(target.shape[-3:])

    # Compression Ratio
    comp_ratio = num_params / num_pixels

    # Print compression ratio
    print(f"Learnable parameters: {num_params:d}")
    print(f"Image Pixels:         {num_pixels:d}")
    print(f"Compression Ratio:    {comp_ratio:0.4f}")

    # Train loop
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        recon = model(img_small)
        loss  = criterion(recon, img_small)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch == 1 or epoch % analyze_every == 0:
            print(f"Epoch {epoch:d}/{num_epochs:d} | Loss: {loss.item():.6f} | LR: {scheduler.get_last_lr()[0]:0.4f} | Max: {recon.max():0.4f} | Min: {recon.min():0.4f} | Mean: {recon.mean():0.4f} | Std: {recon.std():0.4f}")

            model.eval()
            with torch.no_grad():
                final = model(img_small).cpu().squeeze(0).permute(1,2,0).numpy()
                orig  = img_small.cpu().squeeze(0).permute(1,2,0).numpy()

            # Save reconstruction comparison
            fig, axs = plt.subplots(1,2,figsize=(8,4))
            axs[0].imshow(orig);  axs[0].set_title("Original");  axs[0].axis('off')
            axs[1].imshow(final); axs[1].set_title("Reconstruction"); axs[1].axis('off')
            plt.suptitle(
                f"Wavelets: {spatial_depth**2:d} | Freqs/Wavelet: {scale_depth:d} | Total Params: {num_params:d} | Image Dims ({target.shape[-3]}, {target.shape[-2]}, {target.shape[-1]}) | Num Pixels: {num_pixels:d} | Ratio: {comp_ratio}", fontsize=8)
            plt.tight_layout()
            plt.savefig(f"{recons_dir}/trained_recon_{epoch:05d}.png")
            plt.close(fig)

            # Plot histograms
            compare_rgb_hist(
                img1=recon[0].detach().cpu().numpy(),
                img2=target[0].detach().cpu().numpy(),
                labels=["Recon", "Target"],
                percentile=99.9,
                path=os.path.join(hists_dir, f"trained_hists_{epoch:05d}.png")
            )
            
            # Plot Slice
            plot_slice(
                img1=recon[0].detach().cpu().numpy(),
                img2=target[0].detach().cpu().numpy(),
                labels=["Recon", "Target"],
                path=os.path.join(slices_dir, f"trained_slice_{epoch:05d}.png")
            )

        if epoch % (10 * analyze_every) == 0 or epoch == num_epochs:
            # Make videos
            pngs_to_mp4(recons_dir, f"{recons_dir}/trained_recons.mp4", fps=15)
            pngs_to_mp4(hists_dir, f"{hists_dir}/trained_hists.mp4", fps=4)
