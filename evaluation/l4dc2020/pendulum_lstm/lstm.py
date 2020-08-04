import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import MSELoss


# # Here we define our model as a class
# class LSTM(nn.Module):
#
#     def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,
#                  num_layers=2):
#         super(LSTM, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.batch_size = batch_size
#         self.num_layers = num_layers
#
#         # Define the LSTM layer
#         self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)
#
#         # Define the output layer
#         self.linear = nn.Linear(self.hidden_dim, output_dim)
#         self.hidden = self.init_hidden()
#
#     def init_hidden(self):
#         # This is what we'll initialise our hidden state as
#         return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
#                 torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
#
#     def forward(self, input):
#         # Forward pass through LSTM layer
#         # shape of lstm_out: [input_size, batch_size, hidden_dim]
#         # shape of self.hidden: (a, b), where a and b both
#         # have shape (num_layers, batch_size, hidden_dim).
#         # lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1), self.hidden)
#         lstm_out, self.hidden = self.lstm(input.view(len(input), 1, -1), self.hidden)
#
#         # Only take the output from the final timetep
#         # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
#         # y_pred = self.linear(lstm_out[-1].view(len(input), -1))
#         y_pred = self.linear(lstm_out.view(len(input), -1))
#         return y_pred.view(-1)
#
#
# def train_lstm(x, y, model: nn.Module, lr=1e-3, batch_size=256, epochs=100):
#     optim = torch.optim.Adam(model.parameters(), lr=lr)
#     dataset = TensorDataset(x, y)
#     dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
#
#     loss_fn = MSELoss()
#     for epoch in range(epochs):
#         model.hidden = model.init_hidden()
#         for _x, _y in dataloader:
#             out = model.forward(_x)
#             loss = loss_fn(_y.view(-1), out)
#             optim.zero_grad()
#             loss.backward()
#             optim.step()
#         print(epoch)
#     return model


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2, use_cuda=False):
        super(LSTM, self).__init__()
        self.device = torch.device('cuda') if use_cuda else torch.device('cpu')
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h):
        out, h = self.lstm(x, h)
        out = self.fc(out)

        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device))
        return hidden


def train_lstm(train_x, train_y, model, batch_size, lr=0.001, epochs=10):

    criterion = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    train_data = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)

    model.train()
    print("---Training starts---")

    for epoch in range(epochs):
        h = model.init_hidden(batch_size)
        avg_loss = 0
        for x, y in train_loader:
            h = tuple([e.data for e in h])
            model.zero_grad()
            out, h = model(x.float(), h)
            loss = criterion(out, y.float())
            loss.backward()
            optim.step()
            avg_loss += loss.item()

        print("Epoch {}/{} Done, Total Loss: {}".format(epoch + 1, epochs, avg_loss / len(train_loader)))

    return model