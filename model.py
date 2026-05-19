import math
import copy
import os
import gdown
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    raise NotImplementedError


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    raise NotImplementedError

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads   # depth per head
        raise NotImplementedError
    
    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class EncoderLayer(nn.Module):
   
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
       
        raise NotImplementedError

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class DecoderLayer(nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        # TODO: instantiate:
        raise NotImplementedError

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


class Encoder(nn.Module):
    

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
       
        raise NotImplementedError


class Decoder(nn.Module):
  

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
       
        raise NotImplementedError


class Transformer(nn.Module):
    

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model:   int   = 512,
        N:         int   = 6,
        num_heads: int   = 8,
        d_ff:      int   = 2048,
        dropout:   float = 0.1,
        checkpoint_path: str = None,
    ) -> None:
        super().__init__()
        if checkpoint_path is not None:
            gdown.download(id="<.pth drive id>", output=checkpoint_path, quiet=False)
        raise NotImplementedError

    def encode(
        self,
        src:      torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        
    
        raise NotImplementedError

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
       
        raise NotImplementedError

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
       
        raise NotImplementedError


    def infer(self, src_sentence: str) -> str:
       
        raise NotImplementedError
