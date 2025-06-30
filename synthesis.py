import torch
import torch.nn as nn
import math

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
        freqs = torch.Tensor([1.0 * 2**(n/2) for n in range(scale_depth)])
        scales = 1 / freqs
        scales_safe = scales.abs().clamp(min=1e-4)
        self.register_buffer('inv_scales', (1.0 / scales_safe))
        self.register_buffer('scales', scales)

        # Learnable parameters: double for sine and cosine phases
        self.num_bases = spatial_depth * spatial_depth * scale_depth * 2  # factor 2 for sine & cosine
        self.wavelet_coefficients = nn.Parameter(torch.randn(1, 3, self.num_bases))

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

    def cosinc_radial(self, u, v):
        # Cosine-phase counterpart of the radial sinc wavelet
        r = (u**2 + v**2)**0.5
        pi_r = math.pi * r
        # Avoid division by zero
        cosr = torch.where(r == 0, torch.ones_like(r), torch.cos(pi_r) / pi_r)
        return cosr

    def sinc_taxi(self, u, v):
        r = torch.abs(u) + torch.abs(v)
        return torch.sinc(r)

    def sine_radial(self, u, v):
        r = torch.sqrt(u**2 + v**2)
        return torch.sin(2 * torch.pi * r)

    def grid(self, u, v):
        return torch.sin(2 * torch.pi * u) + torch.sin(2 * torch.pi * v)

    def morlet_1d(self, x):
        s = torch.Tensor([5.0]).to(self.device)
        c = 1 / torch.sqrt(1 + torch.exp(-s**2) - 2 * torch.exp(-(3/4) * s**2))
        K = torch.exp(-(s**2 / 2))
        return c * torch.pi**(-1/4) * torch.exp(-(x**2)/2) * (torch.exp(1j * s * x) - K)

    def morlet_2d(self, u, v):
        return torch.real(self.morlet_1d((u**2 + v**2)**0.5))

    def wavepacket_sine(self, u, v):
        s = 1
        r = torch.sqrt(u**2 + v**2)
        g = torch.exp(-0.5 * (r/s)**2)
        f = torch.sin(2*torch.pi*r)
        return g * f
    
    def wavepacket_cosine(self, u, v):
        s = 1
        r = torch.sqrt(u**2 + v**2)
        g = torch.exp(-0.5 * (r/s)**2)
        f = torch.cos(2*torch.pi*r)
        return g * f
    
    def sawtooth_wavepacket_sine_radial(self, u, v):
        N = 8
        S = 1
        sawtooth = 0
        r = torch.sqrt(u**2 + v**2)
        for n in range(1, N + 1):
            sawtooth = sawtooth + (2/torch.pi) * (((-1)**(n+1))/n) * torch.sin(2*torch.pi*n*r)
        env = torch.exp(-0.5 * (r/S)**2)
        return sawtooth * env
    
    def sawtooth_wavepacket_cosine_radial(self, u, v):
        N = 8
        S = 1
        sawtooth = 0
        r = torch.sqrt(u**2 + v**2)
        for n in range(1, N + 1):
            sawtooth = sawtooth + (2/torch.pi) * (((-1)**(n+1))/n) * torch.sin(2*torch.pi*n*(r + 0.25))
        env = torch.exp(-0.5 * (r/S)**2)
        return sawtooth
    
    def sawtooth_wavepacket_sine(self, u, v, phase=0):
        N = 4
        S = 1
        sawtooth_u = 0
        sawtooth_v = 0
        for n in range(1, N + 1):
            sawtooth_u = sawtooth_u + (2/torch.pi) * (((-1)**(n+1))/n) * torch.sin((2*torch.pi*n*(u + phase/(2*torch.pi))))
            sawtooth_v = sawtooth_v + (2/torch.pi) * (((-1)**(n+1))/n) * torch.sin((2*torch.pi*n*(v + phase/(2*torch.pi))))
        env_u = torch.exp(-0.5 * (u/S)**2)
        env_v = torch.exp(-0.5 * (v/S)**2)
        return sawtooth_u * env_u * sawtooth_v * env_v
    
    def sawtooth_wavepacket_cosine(self, u, v):
        return self.sawtooth_wavepacket_sine(u, v, phase=torch.pi/2)

    def forward(self, x):
        # Extract shape
        B, C, H, W = x.shape
        device = x.device

        # Build mesh grid
        xs = torch.linspace(-self.boundx, self.boundx, W, device=device)
        ys = torch.linspace(-self.boundy, self.boundy, H, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # (H,W)

        # Reshape for broadcasting
        grid_x = grid_x.view(1, 1, 1, H, W)
        grid_y = grid_y.view(1, 1, 1, H, W)

        S, Sd = self.spatial_depth, self.scale_depth
        sx = self.shifts_x.view(S, 1, 1, 1, 1)
        sy = self.shifts_y.view(1, S, 1, 1, 1)
        isc = self.inv_scales.view(1, 1, Sd, 1, 1)

        # Vectorized wavelet bank: sine & cosine phases
        u = (grid_x - sx) * isc
        v = (grid_y - sy) * isc
        basis_sin = self.sawtooth_wavepacket_sine_radial(u, v)
        basis_cos = self.sawtooth_wavepacket_cosine_radial(u, v)
        basis = torch.cat([basis_sin, basis_cos], dim=2)  # (S, S, 2*Sd, H, W)

        # Flatten and broadcast
        basis = basis.reshape(self.num_bases, H, W).unsqueeze(0).expand(B, -1, -1, -1)
        coeffs = self.wavelet_coefficients.expand(B, -1, -1)

        # Synthesis
        out = torch.einsum('bcn, bnhw -> bchw', coeffs, basis)
        norm = coeffs.sum(dim=2).view(B, C, 1, 1)
        out = out / norm
        out = self.hls(out)
        return out

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
