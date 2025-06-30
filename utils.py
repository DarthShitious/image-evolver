import torch
import numpy as np
import matplotlib.pyplot as plt

def zscore(x:torch.Tensor):
    """Standardize tensor to zero mean and unit variance."""
    if not isinstance(x, torch.Tensor):
        x = torch.Tensor(x)
    return (x - x.mean(dim=(-1, -2), keepdim=True)) / (x.std(dim=(-1, -2), keepdim=True) + 1e-8)


def plot_rgb_hist(img, bins=256, path=""):
    """
    img: numpy array of shape (3, H, W), values in [0..255] or [0..1]
    """
    colors = ('r','g','b')
    fig = plt.figure(figsize=(6,4))
    for i, c in enumerate(colors):
        channel = img[i].ravel()            # flatten to 1D
        plt.hist(channel, bins=bins,
                 color=c, alpha=0.5,
                 label=f'{c.upper()} channel')
    plt.xlabel('Pixel value')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.title('RGB Histograms')
    plt.tight_layout()
    if path:
        plt.savefig(path)
    plt.close(fig)


def compare_rgb_hist(img1, img2, bins=256, labels=('Image 1', 'Image 2'),
                     percentile=99, path='hist_comparison.png'):
    """
    Plot overlaid RGB histograms of two images in separate subplots,
    where Image 1 is drawn as thick outline histograms and Image 2
    as filled histograms in channel colors, with grids and percentile-based
    y-limits to exclude extreme outliers, then save to disk.

    Parameters
    ----------
    img1, img2 : array-like
        Images in shape (3, H, W) or (H, W, 3), values in [0..255] or [0..1].
    bins : int
        Number of histogram bins.
    labels : tuple of str
        Labels for the histograms of img1 and img2.
    percentile : float
        Percentile cutoff for clipping the y-axis.
    save_path : str
        File path where the resulting figure will be saved (PNG, PDF, etc.).
    """
    def to_cfh(img):
        arr = np.asarray(img)
        if arr.ndim == 3 and arr.shape[0] == 3:
            return arr
        if arr.ndim == 3 and arr.shape[2] == 3:
            return arr.transpose(2, 0, 1)
        raise ValueError("Image must be shape (3,H,W) or (H,W,3)")

    cf1 = to_cfh(img1)
    cf2 = to_cfh(img2)

    plt.style.use('fivethirtyeight')
    channel_colors = ['red', 'green', 'blue']
    channel_names = ['Red', 'Green', 'Blue']

    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

    for i, ax in enumerate(axes):
        d1 = cf1[i].ravel()
        d2 = cf2[i].ravel()
        color = channel_colors[i]

        # Image 1: thick outline histogram
        n1, bins1, _ = ax.hist(
            d1, bins=bins, histtype='step',
            linewidth=2.5, color=color, label=labels[0]
        )
        # Image 2: filled histogram
        n2, bins2, _ = ax.hist(
            d2, bins=bins, histtype='stepfilled',
            alpha=0.4, color=color, label=labels[1]
        )
        # percentile cutoff for y-axis
        cutoff = np.percentile(np.concatenate([n1, n2]), percentile)
        ax.set_ylim(0, cutoff)

        ax.set_title(f'{channel_names[i]} Channel Distribution')
        ax.set_ylabel('Frequency')
        ax.legend(loc='upper right')
        ax.grid(True)

    axes[-1].set_xlabel('Pixel Value')
    fig.tight_layout()

    # Save to file and close figure
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def plot_slice(
        img1:np.ndarray,
        img2:np.ndarray,
        labels=('recon', 'target'),
        path:str=""
    ):

    fig = plt.figure(figsize=(20, 10))
    H = img1.shape[1]
    colors = ['r', 'g', 'b']
    for i in range(3):
        ax1 = fig.add_subplot(3, 1, i + 1)
        ax1.plot(img1[i, H//2, :], label=labels[0], color=colors[i])
        ax1.plot(img2[i, H//2, :], '-', label=labels[1], color=colors[i], alpha=0.5)
        ax1.legend()
        ax1.grid("True")
    if path:
        plt.savefig(path)
    plt.close(fig)