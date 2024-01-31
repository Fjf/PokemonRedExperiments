import torch
from mamba_ssm import Mamba
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from torch import nn
from torch.nn import MSELoss
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

torch.manual_seed(47)

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


def prepare_dataset(tensorset, window_size=10, prediction_window_size=1,
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
    i = 0
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

        x = xs[filter_elems]
        y = ys[filter_elems]

    xs = torch.cat(lxs).unsqueeze(-1)
    ys = torch.cat(lys).unsqueeze(-1)
    return TensorDataset(xs, ys)




# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Model definition using Transformer
class TransformerModel(nn.Module):
    def __init__(self, input_dim=1, d_model=64, num_head=4, num_layers=2, dropout=0.2):
        super(TransformerModel, self).__init__()

        self.encoder = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, num_head)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.decoder(x[:, -1, :])
        return x


def sanity_data():


    import pandas as pd

    names = ['year', 'month', 'day', 'dec_year', 'sn_value',
             'sn_error', 'obs_num', 'unused1']
    df = pd.read_csv(
        "https://data.heatonresearch.com/data/t81-558/SN_d_tot_V2.0.csv",
        sep=';', header=None, names=names,
        na_values=['-1'], index_col=False)

    # Data Preprocessing
    start_id = max(df[df['obs_num'] == 0].index.tolist()) + 1
    df = df[start_id:].copy()
    df['sn_value'] = df['sn_value'].astype(float)
    df_train = df[df['year'] < 2000]
    df_test = df[df['year'] >= 2000]

    spots_train = df_train['sn_value'].to_numpy().reshape(-1, 1)
    spots_test = df_test['sn_value'].to_numpy().reshape(-1, 1)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    spots_train = scaler.fit_transform(spots_train).flatten().tolist()
    spots_test = scaler.transform(spots_test).flatten().tolist()

    # Sequence Data Preparation
    SEQUENCE_SIZE = 10

    def to_sequences(seq_size, obs):
        x = []
        y = []
        for i in range(len(obs) - seq_size):
            window = obs[i:(i + seq_size)]
            after_window = obs[i + seq_size]
            x.append(window)
            y.append(after_window)
        return torch.tensor(x, dtype=torch.float32).view(-1, seq_size, 1), torch.tensor(y, dtype=torch.float32).view(-1,
                                                                                                                     1)

    x_train, y_train = to_sequences(SEQUENCE_SIZE, spots_train)
    x_test, y_test = to_sequences(SEQUENCE_SIZE, spots_test)
    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return dataset, dataloader

def main():
    # Set params
    batch, length, dim = 256, 10, 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Init data
    data = generate_sequences(n=64)
    #dataset = prepare_dataset(data, device=device, window_size=32, prediction_window_size=1, step=1)
    #dataloader = DataLoader(dataset, batch_size=batch, shuffle=False)

    dataset, dataloader = sanity_data()
    # Init model
    # model = Mamba(
    #     # This module uses roughly 3 * expand * d_model^2 parameters
    #     d_model=dim,  # Model dimension d_model
    #     d_state=16,  # SSM state expansion factor
    #     d_conv=4,  # Local convolution width
    #     expand=2,  # Block expansion factor
    # ).to(device)
    model = TransformerModel().to(device)

    # Init optimizer
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = MSELoss()

    for epoch in range(10):
        loss_sum = 0
        for minibatch, y in tqdm(dataloader):
            optim.zero_grad()
            pred = model(minibatch.to(device))
            loss = criterion(pred, y.to(device))
            loss.backward()
            loss_sum += loss.item()

            optim.step()

        print("Loss:", loss_sum / len(dataloader))
    sample = dataset[0]
    pred = model(sample[0].to(device).unsqueeze(0))[0]
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
