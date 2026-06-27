import torch.nn as nn
import torch

class RNNTextClassifierV2(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=128, fc_hidden_dim=64, num_layers=1):
        super().__init__()
        vocab_size, embedding_dim = embedding_matrix.shape

        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=False)

        self.rnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_dim * 2, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        x = torch.mean(out, dim=1)  # global average over sequence
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)


class LSTMTextClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=128, fc_hidden_dim=64):
        super().__init__()
        vocab_size, embedding_dim = embedding_matrix.shape

        # Embedding layer (not frozen)
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=False)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False)

        # Dropout and FC layers
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_dim, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)                         
        output, (hn, cn) = self.lstm(x)               # hn shape: (num_layers, batch_size, hidden_dim)
        x = self.dropout(hn[-1])                      # take last hidden state from last layer
        x = torch.relu(self.fc1(x))                   # first dense layer
        x = self.fc2(x)                               # final logits
        return torch.sigmoid(x)                       # apply sigmoid for binary output



class CNNTextClassifier(nn.Module):
    def __init__(self, embedding_matrix, num_filters=100, filter_sizes=[3, 4, 5]):
        super(CNNTextClassifier, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape

        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=False)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim,
                      out_channels=num_filters,
                      kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_filters * len(filter_sizes), 1)

    def forward(self, x):
        x = self.embedding(x)                          # [batch, seq_len, emb_dim]
        x = x.permute(0, 2, 1)                         # [batch, emb_dim, seq_len]
        conv_outs = [torch.relu(conv(x)) for conv in self.convs]
        pooled = [torch.max(out, dim=2)[0] for out in conv_outs]  # Global max pooling
        cat = torch.cat(pooled, dim=1)                # [batch, num_filters * len(filter_sizes)]
        out = self.dropout(cat)
        return torch.sigmoid(self.fc(out))
