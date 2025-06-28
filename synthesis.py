import torch
import torch.nn as nn

class HardLeakySigmoid(torch.nn.Module):
    def __init__(self, slope=0.01):
        super().__init__()
        self.m = slope

    def forward(self, x):
        f = torch.where(
            x <= -1,
            self.m * (x + 1) - 1,
            torch.where(
                x > 1,
                self.m * (x - 1) + 1,
                x
            )
        )
        return torch.maximum(0.5 * (f + 1), self.m * x)


class WaveletImageSynthesis(nn.Module):
    def __init__(self, spatial_depth, scale_depth, boundx=1.0, boundy=1.0, device=None):
        super().__init__()
        self.boundx = boundx
        self.boundy = boundy
        self.spatial_depth = spatial_depth
        self.scale_depth = scale_depth
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

        # Compute tile widths
        tile_width_x = 2 * boundx / spatial_depth
        tile_width_y = 2 * boundy / spatial_depth

        # Compute centers of tiles
        shifts_x = -boundx + tile_width_x * (0.5 + torch.arange(spatial_depth))
        shifts_y = -boundy + tile_width_y * (0.5 + torch.arange(spatial_depth))

        # Register as buffers
        self.register_buffer('shifts_x', shifts_x)
        self.register_buffer('shifts_y', shifts_y)

        # Scale factors along x and y
        freqs = torch.Tensor([0.5 * 2**(n/4) for n in range(scale_depth)])


        scales = 1/freqs
        scales_safe = scales.abs().clamp(min=1e-4)
        self.register_buffer('inv_scales', (1.0 / scales_safe))
        self.register_buffer('scales', scales)

        # Learnable parameters
        self.num_bases = spatial_depth * spatial_depth * scale_depth
        self.wavelet_coefficients = nn.Parameter(
            torch.randn(1, 3, self.num_bases)
        )

        # Output Activation
        self.hls = HardLeakySigmoid(slope=0.01)

    def ricker(self, u, v):
        return (1 - 2 * (u**2 + v**2)) * torch.exp(-(u**2 + v**2))

    def sinegaussian(self, u, v):
        return torch.sin(u) * torch.sin(v) * torch.exp(-(u**2 + v**2))

    def sinc(self, u, v):
        return torch.sinc(u) * torch.sinc(v)

    def sinc_radial(self, u, v):
        r = (u**2 + v**2)**0.5
        return torch.sinc(r)
    
    def sinc_taxi(self, u, v):
        return torch.sinc(torch.abs(u) + torch.abs(v))
    
    def sine_radial(self, u, v):
        r = torch.sqrt(u**2 + v**2)
        return torch.sin(2*torch.pi*r)
    
    def grid(self, u, v):
        return torch.sin(2*torch.pi*u) + torch.sin(2*torch.pi*v)

    def morlet_1d(self, x):
        s = torch.Tensor([5.0]).to(self.device)
        c = 1/torch.sqrt(1 + torch.exp(-s**2) - 2*torch.exp(-(3/4)*s**2))
        K = torch.exp(-(s**2/2))
        return c * torch.pi**(-1/4) * torch.exp(-(x**2)/2) * (torch.exp(1j * s * x) - K)

    def morlet_2d(self, u, v):
        # return torch.real(self.morlet_1d(u) + self.morlet_1d(v))
        return torch.real(self.morlet_1d((u**2 + v**2)**0.5))

    def forward(self, x):
        # Just use the input image to extract shape info.. this is dumb, I know.
        # I'll move it to the init eventually, ugh.
        B, C, H, W = x.shape
        device = x.device

        # Build a mesh grid
        xs = torch.linspace(-self.boundx, self.boundx, W, device=device)
        ys = torch.linspace(-self.boundy, self.boundy, H, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # (H,W)

        # Bring grid into a [1,1,1,H,W] shape
        grid_x = grid_x.view(1, 1, 1, H, W)
        grid_y = grid_y.view(1, 1, 1, H, W)

        # Reshape shifts and scales for broadcasting:
        S, Sd = self.spatial_depth, self.scale_depth
        sx = self.shifts_x.view(S, 1, 1, 1, 1)    # → (S,1,1,1,1)
        sy = self.shifts_y.view(1, S, 1, 1, 1)    # → (1,S,1,1,1)
        isc = self.inv_scales.view(1, 1, Sd, 1, 1)# → (1,1,Sd,1,1)

        # Vectorized wavelet bank: (S, S, Sd, H, W) ---
        u = (grid_x - sx) * isc
        v = (grid_y - sy) * isc
        basis = self.sinc_radial(u, v)

        # Flatten to (N, H, W) then batch-broadcast: (B, N, H, W)
        basis = basis.reshape(self.num_bases, H, W).unsqueeze(0).expand(B, -1, -1, -1)

        # Broadcast learnable coeffs: (B,3,N)
        coeffs = self.wavelet_coefficients.expand(B, -1, -1)

        # Einsum to get (B,3,H,W)
        out = torch.einsum('bcn, bnhw -> bchw', coeffs, basis)

        # Normalize
        norm = coeffs.sum(dim=2).view(B, C, 1, 1)
        out = out / norm

        # Yup
        out = self.hls(out)
        return out

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

