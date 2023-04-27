import torch
import torch.nn as nn

class BLSTMNet(nn.Module):
    def __init__(
        self, args, input_dim
    ):
        super(BLSTMNet, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.emb_dim = self.args.nxvec if self.args.use_xvec else 2 * input_dim
        self.units = self.args.noUnits

        if not self.args.use_xvec and not self.args.use_stats:
            self.lstm_dim = self.input_dim
        else:
            self.lstm_dim = 232

    
    def init_model(self):
        """ 3 layer BLSTM with linear regression."""
        if self.args.use_xvec or self.args.use_stats:
            self.input_dense = nn.Linear(
                self.input_dim,
                200,
                bias = True
            )
            self.emb_dense = nn.Linear(
                self.emb_dim,
                32,
                bias = True
            )

        self.lstm1 = nn.LSTM(
            input_size = self.lstm_dim,
            hidden_size = self.units,
            num_layers = 1,
            batch_first = True,
            bidirectional = True
        )
        self.lstm2 = nn.LSTM(
            input_size = self.units * 2,
            hidden_size = self.units,
            num_layers = 1,
            batch_first = True,
            bidirectional = True
        )
        self.lstm3 = nn.LSTM(
            input_size = self.units * 2,
            hidden_size = self.units,
            num_layers = 1,
            batch_first = True,
            bidirectional = True
        )
        self.linear = nn.Linear(
            self.units * 2, 
            self.args.nEMA, 
            bias = True
        )

        
    def forward(self, x, emb = None):
        """ Forward pass of the model """
        if emb is not None:
            x = self.input_dense(x)
            emb = self.emb_dense(emb)
            x = torch.cat((x, emb), dim = -1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = self.linear(x)
        return x