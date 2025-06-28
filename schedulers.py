import torch
import math


class LINEAR_WARMUP(torch.optim.lr_scheduler._LRScheduler):
    # Ramp up learning rate linearly
    def __init__(self, optimizer, warmup_steps, max_lr, min_lr, last_epoch=-1):
        # Maximum learning rate
        self.max_lr = max_lr
        # Minimum learning rate
        self.min_lr = min_lr
        # Number of steps to ramp up
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        t = self.last_epoch + 1
        T = self.warmup_steps
        b = self.min_lr / self.max_lr
        s = b + ((1 - b) / T) * t
        scale = min(1.0, s)
        return [base_lr * scale for base_lr in self.base_lrs]


class COSINE_ANNEALING_WARM_RESTARTS(torch.optim.lr_scheduler._LRScheduler):
    # Cosine annealing schedule with warm restarts
    def __init__(
        self, optimizer, T_0, T_mult=1, max_lr=1e-3, min_lr=1e-6, last_epoch=-1
    ):
        # Base cycle length
        self.T_0 = T_0
        # Cycle length multiplier between restarts
        self.T_mult = T_mult
        # Maximum learning rate
        self.max_lr = max_lr
        # Minimum learning rate
        self.min_lr = min_lr
        # Epoch at last restart
        self.last_restart = 0
        # Current cycle length
        self.T_i = T_0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        t = self.last_epoch + 1
        # If we've completed the current cycle, advance to the next
        if t - self.last_restart > self.T_i:
            self.last_restart += self.T_i
            self.T_i *= self.T_mult
        # Progress within the current cycle
        T_cur = t - self.last_restart
        # Compute cosine annealing scale
        b = self.min_lr / self.max_lr
        cos_scale = 0.5 * (1 + math.cos(math.pi * T_cur / self.T_i))
        scale = b + (1 - b) * cos_scale
        lrs = [base_lr * scale for base_lr in self.base_lrs]
        return lrs


class LINEAR_WARMUP_HOLDON_COOLDOWN(torch.optim.lr_scheduler._LRScheduler):
    # Ramp up learning rate linearly
    def __init__(
        self,
        optimizer,
        warmup_steps,
        holdon_steps,
        cooldown_steps,
        max_scale=1.0,
        min_scale=0.1,
        last_epoch=-1,
    ):
        # Maximum learning rate
        self.s_max = max_scale
        # Minimum learning rate
        self.s_min = min_scale
        # Number of steps to ramp up
        self.T_w = warmup_steps
        # Number of steps to hold
        self.T_h = holdon_steps
        # Number of steps to cooldown
        self.T_c = cooldown_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        t = self.last_epoch + 1
        if t <= self.T_w:
            scale = self.s_min + t * (self.s_max - self.s_min) / self.T_w
        elif (t > self.T_w) and (t <= self.T_w + self.T_h):
            scale = self.s_max
        elif t > self.T_w + self.T_h:
            scale = (
                self.s_max
                - (t - (self.T_w + self.T_h)) * (self.s_max - self.s_min) / self.T_c
            )
        else:
            print(
                "[WARNING] Temporal anomaly detected. Learning rate schedule has drifted into forbidden space. Proceed at the peril of your model’s sanity."
            )

        scale = min(max(scale, self.s_min), self.s_max)

        lrs = [base_lr * scale for base_lr in self.base_lrs]
        return lrs


if __name__ == "__main__":
    from torch.optim.adamw import AdamW

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ff = torch.nn.Sequential(
                torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1)
            )

        def forward(self, x):
            return self.ff(x)

    model = Model()
    optimizer = AdamW(params=model.parameters(), lr=1e-3)
    scheduler = COSINE_ANNEALING_WARM_RESTARTS(
        optimizer=optimizer, T_0=64 * 16, T_mult=1.5, max_lr=0.0004, min_lr=0.0001
    )

    import numpy as np
    import matplotlib.pyplot as plt

    steps = np.arange(64 * 12 * 10)
    lrs = []
    for _ in steps:
        lrs.append(scheduler.get_last_lr())

        optimizer.step()
        scheduler.step()

    lrs = np.array(lrs)

    fig = plt.figure(figsize=(10, 5))
    plt.plot(steps / (64 * 12), lrs)
    plt.grid("both")
    plt.show()

    print()
