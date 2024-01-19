import torch
from mamba_ssm import Mamba
import matplotlib.pyplot as plt
import matplotlib
from torch.nn import MSELoss
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def generate_sequences(n=512):
    offset = (torch.rand(n) * torch.pi)
    seq_length = ((1 + torch.rand(n)) * torch.pi)
    cosine_data = []
    for i in tqdm(range(n)):
        xs = torch.arange(n)
        xs = torch.cos((xs + offset[i]) * seq_length[i])
        cosine_data.append(xs)

    return torch.stack(cosine_data)


def prepare_dataset(tensorset, window_size=10, prediction_window_size=10,
                    step=1, device="cpu"):
    """
    Create a dataset from the input tensor dict.
    Filters out all elements where either the `xs` or the `ys` contain NaNs.

    :param tensorset: the list of input tensors
    :param device: the device to which to move the tensors
    :param window_size: the window view size the prediction will see
    :param prediction_window_size: the amount of samples that have to be predicted in advance
    :param step: the overlap between two adjacent samples
    :return: TensorDataset containing sets of in/outputs.
    """
    # Fetch the data we want to predict
    lxs = []
    lys = []
    for tensor in tqdm(tensorset):
        data = tensor[1:]

        # Normalize data, ignore all nan values.
        data = (data - data[~data.isnan()].mean()) / data[~data.isnan()].std()

        # Do not use the last N datapoints, because then we cannot predict the next timestep
        xs = data.unfold(0, window_size, step)[:-prediction_window_size]
        # Predict the next timestep, so take the input data starting from the context_size
        ys = data[window_size:].unfold(0, prediction_window_size, step)

        # Filter all segments containing nans from the dataset
        filter_elems = ~(torch.any(xs.isnan(), dim=1) | torch.any(ys.isnan(), dim=1))
        lxs.append(xs[filter_elems])
        lys.append(ys[filter_elems])

    xs = torch.cat(lxs).unsqueeze(-1)
    ys = torch.cat(lys).unsqueeze(-1)
    return TensorDataset(xs, ys)


def main():
    # Set params
    batch, length, dim = 512, 10, 1
    device = "cuda"

    # Init data
    data = generate_sequences(n=256)
    dataset = prepare_dataset(data, device=device, window_size=32, prediction_window_size=32)
    dataloader = DataLoader(dataset, batch_size=batch)

    # Init model
    model = Mamba(
        # This module uses roughly 3 * expand * d_model^2 parameters
        d_model=dim,  # Model dimension d_model
        d_state=16,  # SSM state expansion factor
        d_conv=4,  # Local convolution width
        expand=2,  # Block expansion factor
    ).to(device)

    # Init optimizer
    optim = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss_fn = MSELoss()

    for epoch in range(100):
        loss_sum = 0
        for minibatch, y in tqdm(dataloader):
            optim.zero_grad()

            pred = model(minibatch.to("cuda"))
            loss = ((pred - y.to("cuda")) ** 2).mean()
            loss.backward()
            loss_sum += loss.item()

            optim.step()

        print("Loss:", loss_sum / len(dataloader))
    sample = dataset[0]
    pred = model(sample[0].to("cuda").unsqueeze(0))[0]
    plt.plot(sample[0].to("cpu"))
    plt.plot(sample[1].to("cpu"))
    plt.plot(pred.to("cpu").detach().numpy())
    plt.legend(["x", "y", "pred"])
    plt.savefig("out.png")


def plot_data(data):
    w, h = 4, 3
    fig, ax = plt.subplots(w, h)
    for i in range(w * h):
        ax[i % w, i // w].plot(data[i])
    plt.show()


if __name__ == "__main__":
    main()
