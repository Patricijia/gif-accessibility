import torch
import torch.nn as nn


class SimpleStudentCaptioner(nn.Module):
    """Placeholder small captioning model.

    Replace with a TinyViT / MobileNet encoder + small GRU/Transformer decoder
    once you decide on the exact architecture.
    """

    def __init__(self, vocab_size: int, hidden_size: int = 256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, **kwargs):
        x = self.embed(input_ids)
        out, _ = self.lstm(x)
        logits = self.fc(out)
        return logits
