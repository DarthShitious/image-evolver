import os
import torch
import torch.nn.functional as F
import numpy as np
from skimage import data
from tqdm import trange
import matplotlib.pyplot as plt
from datetime import datetime
from torchvision.io import read_image
from torchvision.transforms.functional import resize

from synthesis import WaveletImageSynthesis
from utils import *
from analysis import pngs_to_mp4


def fitness(coeffs, target):
    model.wavelet_coefficients.data = coeffs.unsqueeze(0)
    recon = model(target)
    mse   = F.mse_loss(recon, target)
    return mse.item()

if __name__ == "__main__":

    # Configuration
    pop_size    = 128
    parent_k    = 64
    num_elites  = 2
    generations = 2000000
    mut_rate    = 0.05
    mut_scale   = 0.1
    analyze_every = 10
    shrink_scale = 16
    spatial_depth = 8
    scale_depth = 32

    # Time stamp for results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.getcwd(), "evolved", timestamp)
    recons_dir = os.path.join(results_dir, "recons")
    hists_dir = os.path.join(results_dir, "hists")
    slices_dir = os.path.join(results_dir, "slices")
    for p in [results_dir, recons_dir, hists_dir, slices_dir]:
        os.makedirs(p, exist_ok=True)

    checkpoint_loadpath = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare the target image
    img = read_image("./images/uv_face.jpg") / 255.0
    _, img_H, img_W = img.shape

    target = resize(img, (img_H // shrink_scale, img_W // shrink_scale)).unsqueeze(0).to(device)

    # Build the model
    model = WaveletImageSynthesis(spatial_depth=spatial_depth, scale_depth=scale_depth).to(device)
    C, N = 3, model.num_bases

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

    # Load from checkpoint if it exists
    if checkpoint_loadpath and os.path.exists(checkpoint_loadpath):
        ckpt = torch.load(checkpoint_loadpath, map_location=device)
        pop = ckpt["pop"]
        history = ckpt["history"]
        start_gen = ckpt["gen"] + 1
        print(f"Loaded checkpoint from '{checkpoint_loadpath}' at gen {ckpt['gen']}. Resuming at {start_gen}.")
    else:
        pop = torch.randn(pop_size, C, N, device=device)
        history = []
        start_gen = 1

    # Evolution Loop
    for gen in trange(start_gen, generations+1):

        # Evaluate Fitnesses
        fitnesses = torch.tensor([fitness(ind, target) for ind in pop], device=device)

        # Rank
        topk_vals, topk_idx = torch.topk(fitnesses, parent_k, largest=False)

        # Select Breeding pool
        parents = pop[topk_idx]

        # Record Best
        best_mse = topk_vals.max().item()
        history.append(best_mse)

        # Analysis
        if gen == 1 or gen % analyze_every == 0:
            print(f"Gen {gen}/{generations}  Best MSE: {best_mse:.6f}")

            # Decrease mutation scale
            mut_scale = max(0.005, mut_scale * 0.999)
            print(f"Mutation scale: {mut_scale:.4f}")

            with torch.no_grad():
                model.wavelet_coefficients.data = parents[0].unsqueeze(0)
                recon = model(target).cpu()[0].permute(1,2,0).numpy()
                orig  = target.cpu()[0].permute(1,2,0).numpy()

            fig, ax = plt.subplots(1,2,figsize=(6,3))
            ax[0].imshow(orig);  ax[0].set_title("Orig");  ax[0].axis('off')
            ax[1].imshow(recon); ax[1].set_title("Best"); ax[1].axis('off')
            plt.suptitle(
                f"Wavelets: {spatial_depth**2:d} | Freqs/Wavelet: {scale_depth:d} | Total Params: {num_params:d} | Image Dims ({target.shape[-3]}, {target.shape[-2]}, {target.shape[-1]}) | Num Pixels: {num_pixels:d} | Ratio: {comp_ratio}", fontsize=8)
            plt.savefig(f"{recons_dir}/evolved_{gen:06d}.png")
            plt.close(fig)

            plt.figure()
            plt.plot(history)
            plt.xlabel("Generation")
            plt.ylabel("Best MSE")
            plt.title("GA Training Progress")
            plt.savefig(f"{results_dir}/evolution_progress.png")
            plt.close()

            # Plot histograms
            compare_rgb_hist(
                img1=recon,
                img2=target[0].cpu().numpy(),
                labels=["Recon", "Target"],
                percentile=99.9,
                path=os.path.join(hists_dir, f"evolved_hists_{gen:05d}.png")
            )

            # Plot Slice
            plot_slice(
                img1=recon.transpose(2, 0, 1),
                img2=target[0].cpu().numpy(),
                labels=["Recon", "Target"],
                path=os.path.join(slices_dir, f"evolved_slice_{gen:05d}.png")
            )

            # Save Checkpoint
            checkpoint_savepath = os.path.join(results_dir, "checkpoint.pth")
            torch.save({
                "pop": pop,
                "history": history,
                "gen": gen
            }, checkpoint_savepath)
            print(f"Saved checkpoint to '{checkpoint_savepath}' at gen {gen}")

        if gen % (10 * analyze_every) == 0 or gen == generations:
            # Make videos
            pngs_to_mp4(recons_dir, f"{recons_dir}/evolved_recons.mp4", fps=15)
            pngs_to_mp4(hists_dir, f"{hists_dir}/evolved_hists.mp4", fps=4)


        # Create the next generation
        next_pop = []

        # Elitism
        for i in range(num_elites):
            next_pop.append(parents[i].clone())

        # Crossover + Mutation
        while len(next_pop) < pop_size:
            i, j = np.random.choice(parent_k, size=2, replace=False)
            p1, p2 = parents[i], parents[j]
            point = np.random.randint(1, N)
            child = torch.cat([p1[:, :point], p2[:, point:]], dim=1)
            mask  = (torch.rand_like(child) < mut_rate).float()
            noise = torch.randn_like(child) * mut_scale

            # Small chance for super mutation
            if np.random.rand() < 0.05:
                noise = noise * 10

            next_pop.append(child + mask * noise)

        pop = torch.stack(next_pop, dim=0)


