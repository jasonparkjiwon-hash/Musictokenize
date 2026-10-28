"""GRU next-song model: vocabulary = songs, trained with next-song cross-entropy."""
import numpy as np
import torch
from torch import nn


class SongGRU(nn.Module):
    def __init__(self, vocab_size: int, embed_dim=64, hidden_dim=128, num_layers=1,
                 dropout=0.3, tie_weights=True):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.proj = nn.Linear(hidden_dim, embed_dim)
        self.out = nn.Linear(embed_dim, vocab_size)
        if tie_weights:
            self.out.weight = self.embed.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T] song ids -> logits [B, T, V] for the song at each next position."""
        h, _ = self.gru(self.drop(self.embed(x)))
        return self.out(self.drop(self.proj(h)))


class GRUScorer:
    """Adapter so the shared evaluator can score a SongGRU."""
    name = "gru"

    def __init__(self, model: SongGRU, device="cpu"):
        self.model, self.device = model.eval(), device

    @torch.no_grad()
    def score_sequence(self, ids: list[int]) -> np.ndarray:
        x = torch.tensor([ids[:-1]], device=self.device)
        return self.model(x)[0].cpu().numpy()

    @torch.no_grad()
    def score(self, context: list[int]) -> np.ndarray:
        x = torch.tensor([context], device=self.device)
        return self.model(x)[0, -1].cpu().numpy()
