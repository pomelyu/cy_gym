# reference from https://zhuanlan.zhihu.com/p/28731517852
# Demo flow matching algorithm by mapping normal distribution to sin wave
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from tqdm import trange


def build_target_data(n_samples: int) -> Tensor:
    x = torch.rand(n_samples, 1) * (4 * torch.pi)
    y = torch.sin(x)
    return torch.cat([x, y], -1)    # (N, 2)

class VectorField2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + 1, 64),   # 2d tensor + time
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        xt = torch.cat([x, t], -1)
        return self.net(xt)

def main():
    n_samples = 1000
    lr = 1e-3
    n_epoch = 5000
    batch_size = 1000

    device = torch.device("cpu")

    noise_data = (torch.randn(n_samples, 2) * 2).to(device)
    target_data = build_target_data(n_samples).to(device)

    model = VectorField2D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    pbar = trange(n_epoch, desc="Training...")
    for epoch in pbar:
        batch_idx = torch.randperm(n_samples)[:batch_size]

        x0 = noise_data[batch_idx]      # T=0, gaussian noise
        x1 = target_data[batch_idx]     # T=1, sin wave

        t = torch.rand(batch_size, 1).to(device)
        xt = x0 * (1 - t) + x1 * t      # T=t

        vt_pred = model(xt, t)
        vt_gt = x1 - x0

        loss = torch.nn.functional.mse_loss(vt_pred, vt_gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(f"Epoch: {epoch:0>3d}, Loss: {loss.item():.4f}")


    # Visualize flow matching process
    @torch.no_grad()
    def visualize():
        n_steps = 50
        x = noise_data[:5]
        t = torch.zeros(5, 1).to(device)
        delta_t = 1 / n_steps

        x_trajectory = [x]
        for _ in range(n_steps):
            vt = model(x, t)
            x = x + vt * delta_t
            t = t + delta_t

            x_trajectory.append(x.cpu().numpy())
        x_trajectory = np.stack(x_trajectory, 1)

        plt.figure(figsize=(10, 5))
        plt.scatter(noise_data[:, 0], noise_data[:, 1], c="b", label="noise")
        plt.scatter(target_data[:, 0], target_data[:, 1], c="g", label="target")
        plt.plot(x_trajectory[0, :, 0], x_trajectory[0, :, 1], c="r", label="trajectory")
        plt.plot(x_trajectory[1, :, 0], x_trajectory[1, :, 1], c="r", label="trajectory")
        plt.plot(x_trajectory[2, :, 0], x_trajectory[2, :, 1], c="r", label="trajectory")
        plt.plot(x_trajectory[3, :, 0], x_trajectory[3, :, 1], c="r", label="trajectory")
        plt.plot(x_trajectory[4, :, 0], x_trajectory[4, :, 1], c="r", label="trajectory")
        plt.legend()
        plt.show()

    visualize()


if __name__ == "__main__":
    main()
