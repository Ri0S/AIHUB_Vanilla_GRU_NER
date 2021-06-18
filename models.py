import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_class=174, vocab_size=8293):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.rnn = nn.GRU(128, 256, num_layers=2, dropout=0.2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, num_class)
        self.num_class = num_class

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, x, label, length, mode='train'):
        emb = torch.nn.utils.rnn.pack_padded_sequence(self.embedding(x), length, batch_first=True)
        outputs, hidden = self.rnn(emb)

        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, True)

        logits = self.fc(outputs)
        if mode == 'train':
            loss = self.criterion(logits.view(-1, self.num_class), label.view(-1))
        elif mode == 'test':
            loss = 0
        else:
            loss = 0
        return logits, loss
