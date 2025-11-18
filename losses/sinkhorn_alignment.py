from typing import List
import torch
from sentence_transformers import SentenceTransformer
from geomloss import SamplesLoss

class SinkhornAligner:
    def __init__(self, encoder_name: str = "all-MiniLM-L6-v2", blur: float = 0.1):
        self.encoder = SentenceTransformer(encoder_name)
        self.loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=blur)

    def encode(self, texts: List[str]) -> torch.Tensor:
        emb = self.encoder.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
        return emb

    def loss(self, y_prime_emb: torch.Tensor, nu_c_emb: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(y_prime_emb, nu_c_emb)