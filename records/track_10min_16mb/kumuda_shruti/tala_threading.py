"""
TALA THREADING LAYER — Inspired by Carnatic Music's Tala System
================================================================
The hyper-threading engine of our "Shruti" architecture.

Just as Carnatic music uses different Tala cycles running simultaneously
to create complex rhythmic textures, our Tala Threading layer processes
the Chakra token stream through 4 parallel threads — each with a
different cycle length — capturing linguistic patterns at different
temporal scales simultaneously.

Architecture role: Hyper-threading (runs ON TOP of Chakra Core)

Thread cycle lengths: 5, 6, 7, 8 (Khanda, Rupaka, Misra, Adi)
Why these numbers? They are coprime-ish — meaning threads stay
out of phase with each other, maximizing temporal coverage.

LCM(5,6,7,8) = 840 — full pattern only repeats every 840 tokens!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from chakra_core import ChakraTokenizer, VOCAB_SIZE


# ─────────────────────────────────────────────
# TALA DEFINITIONS
# Each Tala = a rhythmic cycle from Carnatic music
# mapped to a linguistic processing window
# ─────────────────────────────────────────────

TALAS = {
    "khanda":  {"beats": 5, "role": "clause_boundary",  "color": "🔴"},
    "rupaka":  {"beats": 6, "role": "phrase_rhythm",     "color": "🟡"},
    "misra":   {"beats": 7, "role": "coarticulation",    "color": "🟢"},
    "adi":     {"beats": 8, "role": "word_level",        "color": "🔵"},
}

TALA_CYCLES = [5, 6, 7, 8]   # Khanda, Rupaka, Misra, Adi


# ─────────────────────────────────────────────
# GAMAKA POSITION ENCODING
# Unlike standard sinusoidal PE, Gamaka PE
# encodes position WITHIN each Tala cycle —
# like a drummer knowing where they are in the beat
# ─────────────────────────────────────────────

class GamakaPositionEncoding(nn.Module):
    """
    Position encoding inspired by Carnatic Gamaka (ornaments).
    
    For each token position, we compute:
    - Where it falls in each Tala cycle (phase)
    - A smooth sinusoidal encoding of that phase
    - Combined across all 4 Talas
    
    This gives the model rhythmic awareness of
    where each phoneme sits in the speech pattern.
    """

    def __init__(self, dim: int, max_len: int = 2048):
        super().__init__()
        self.dim = dim

        # Standard positional encoding base
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:dim//2])

        # Gamaka: additional Tala-phase encoding
        # For each Tala cycle T, encode sin/cos of position % T
        tala_pe = torch.zeros(max_len, dim)
        slots_per_tala = dim // len(TALA_CYCLES)  # divide dim across 4 talas

        for t_idx, T in enumerate(TALA_CYCLES):
            phase = (position % T) / T * 2 * math.pi  # phase within cycle
            start = t_idx * slots_per_tala
            end = start + slots_per_tala

            half = (end - start) // 2
            tala_pe[:, start:start+half] = torch.sin(
                phase * torch.arange(1, half+1).float()
            )
            tala_pe[:, start+half:end] = torch.cos(
                phase * torch.arange(1, half+1).float()
            )

        # Blend standard PE + Gamaka PE (50/50)
        combined = 0.5 * pe + 0.5 * tala_pe
        self.register_buffer("pe", combined.unsqueeze(0))  # (1, max_len, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, dim)"""
        return x + self.pe[:, :x.size(1)]


# ─────────────────────────────────────────────
# SINGLE TALA THREAD
# One RNN + local attention unit
# processes the token stream at its own cycle
# ─────────────────────────────────────────────

class TalaThread(nn.Module):
    """
    A single Carnatic Tala thread.
    
    Contains:
    - GRU (recurrent) — remembers phoneme flow like a drummer
      keeping the beat in memory
    - Local attention window — attends within its Tala cycle
      (like a musician focusing on their current rhythmic phrase)
    - Beat projection — mixes recurrent + attention output
    """

    def __init__(self, dim: int, cycle: int, tala_name: str):
        super().__init__()
        self.cycle = cycle
        self.name = tala_name
        self.dim = dim

        # GRU — lightweight recurrent memory
        # Weight tying: input and hidden have same dim (parameter efficient)
        self.gru = nn.GRU(
            input_size=dim,
            hidden_size=dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False  # causal — like music flowing forward
        )

        # Local attention within Tala cycle window
        self.attn_q = nn.Linear(dim, dim, bias=False)
        self.attn_k = nn.Linear(dim, dim, bias=False)
        self.attn_v = nn.Linear(dim, dim, bias=False)
        self.attn_scale = math.sqrt(dim)

        # Beat projection — combines GRU + attention
        self.beat_proj = nn.Linear(dim * 2, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def local_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Attend only within a window of size = Tala cycle.
        Like a musician only "hearing" within their current beat cycle.
        """
        B, T, D = x.shape
        Q = self.attn_q(x)  # (B, T, D)
        K = self.attn_k(x)
        V = self.attn_v(x)

        # Build local attention mask — each position attends to
        # at most `cycle` positions back
        mask = torch.full((T, T), float("-inf"), device=x.device)
        for i in range(T):
            start = max(0, i - self.cycle + 1)
            mask[i, start:i+1] = 0.0

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.attn_scale
        scores = scores + mask.unsqueeze(0)
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, V)  # (B, T, D)

    def forward(self, x: torch.Tensor, hidden=None):
        """
        x: (batch, seq_len, dim)
        Returns: (batch, seq_len, dim), hidden_state
        """
        # GRU pass — sequential phoneme memory
        gru_out, hidden = self.gru(x, hidden)

        # Local attention pass — rhythmic context
        attn_out = self.local_attention(x)

        # Beat projection — blend both
        combined = torch.cat([gru_out, attn_out], dim=-1)
        out = self.beat_proj(combined)
        out = self.norm(out + x)  # residual like Chakra's recursive structure

        return out, hidden


# ─────────────────────────────────────────────
# TALA THREADING LAYER
# All 4 Tala threads running in parallel
# Output merged via Bandha (learned mixing)
# ─────────────────────────────────────────────

class TalaThreadingLayer(nn.Module):
    """
    The full Carnatic hyper-threading system.
    
    4 Tala threads (Khanda=5, Rupaka=6, Misra=7, Adi=8)
    run in PARALLEL on the same phoneme stream.
    
    Each thread captures a different temporal scale:
    - Khanda (5): clause boundaries, punctuation rhythm
    - Rupaka (6): phrase-level patterns
    - Misra (7): coarticulation, sound blending
    - Adi (8):   word-level phoneme sequences
    
    Outputs are merged by a learned Bandha mixer —
    like a conductor balancing 4 musicians.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # 4 Tala threads
        tala_names = list(TALAS.keys())
        self.threads = nn.ModuleList([
            TalaThread(dim, cycle, tala_names[i])
            for i, cycle in enumerate(TALA_CYCLES)
        ])

        # Bandha mixer — learned weighted combination of 4 threads
        # Inspired by Bandha (traversal key) in Siri Bhoovalaya
        # Small linear layer: 4*dim → dim
        self.bandha_mixer = nn.Linear(dim * len(TALA_CYCLES), dim, bias=False)
        self.bandha_norm = nn.LayerNorm(dim)

        # Gamaka position encoding
        self.gamaka_pe = GamakaPositionEncoding(dim)

    def forward(self, x: torch.Tensor):
        """
        x: (batch, seq_len, dim) — Chakra embeddings
        Returns: (batch, seq_len, dim) — threaded output
        """
        # Apply Gamaka position encoding first
        x = self.gamaka_pe(x)

        # Run all 4 Tala threads IN PARALLEL
        thread_outputs = []
        for thread in self.threads:
            out, _ = thread(x)
            thread_outputs.append(out)

        # Concatenate thread outputs along feature dim
        # (batch, seq_len, dim*4)
        merged = torch.cat(thread_outputs, dim=-1)

        # Bandha mixer: collapse 4 threads → 1 unified stream
        # (batch, seq_len, dim)
        out = self.bandha_mixer(merged)
        out = self.bandha_norm(out + x)  # residual connection

        return out


# ─────────────────────────────────────────────
# FULL EMBEDDING + THREADING STACK
# Chakra embedding → Tala threading
# This is the complete input processing pipeline
# ─────────────────────────────────────────────

class ChakraTalaEmbedding(nn.Module):
    """
    Complete input pipeline:
    Token IDs → Chakra Embeddings → Tala Threading → Rich representations
    
    This replaces the standard "embedding + positional encoding"
    used in vanilla transformers.
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, dim: int = 256,
                 num_tala_layers: int = 2, dropout: float = 0.1):
        super().__init__()

        # Chakra embedding table (64 × dim)
        # Tiny! 64*256 = 16,384 params vs baseline 1024*512 = 524,288
        self.chakra_embed = nn.Embedding(vocab_size, dim)

        # Stack of Tala threading layers
        self.tala_layers = nn.ModuleList([
            TalaThreadingLayer(dim)
            for _ in range(num_tala_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.dim = dim

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (batch, seq_len) — Chakra token IDs (0-63)
        Returns:   (batch, seq_len, dim)
        """
        # Chakra embedding lookup
        x = self.chakra_embed(token_ids)  # (B, T, dim)
        x = self.dropout(x)

        # Pass through Tala threading layers
        for tala_layer in self.tala_layers:
            x = tala_layer(x)

        return x

    def param_count(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        embed = sum(p.numel() for p in self.chakra_embed.parameters())
        tala = sum(p.numel() for p in self.tala_layers.parameters())
        return {
            "embedding": embed,
            "tala_threading": tala,
            "total": total,
            "size_mb": total * 4 / 1024 / 1024  # float32
        }


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("SHRUTI ARCHITECTURE — Tala Threading Layer Test")
    print("=" * 60)

    # Init tokenizer
    tok = ChakraTokenizer()

    # Test sentences
    sentences = [
        "the cat sat on the mat",
        "shruti is a sound in carnatic music",
    ]

    # Tokenize
    encoded = [tok.encode(s) for s in sentences]
    max_len = max(len(e) for e in encoded)

    # Pad to same length
    padded = [e + [tok.pad] * (max_len - len(e)) for e in encoded]
    token_ids = torch.tensor(padded)  # (2, max_len)

    print(f"\nInput shape: {token_ids.shape}")
    print(f"Vocab size:  {VOCAB_SIZE} (Chakra 64)")
    print(f"Tala cycles: {TALA_CYCLES} (Khanda, Rupaka, Misra, Adi)")
    print()

    # Build embedding stack
    dim = 256
    model = ChakraTalaEmbedding(
        vocab_size=VOCAB_SIZE,
        dim=dim,
        num_tala_layers=2
    )

    # Forward pass
    with torch.no_grad():
        output = model(token_ids)

    print(f"Output shape: {output.shape}  (batch=2, seq={max_len}, dim={dim})")
    print()

    # Parameter count
    params = model.param_count()
    print("Parameter breakdown:")
    print(f"  Chakra embedding : {params['embedding']:>10,}  params")
    print(f"  Tala threading   : {params['tala_threading']:>10,}  params")
    print(f"  TOTAL            : {params['total']:>10,}  params")
    print(f"  Size             : {params['size_mb']:.3f} MB (float32)")
    print()

    # Show Tala thread info
    print("Tala Threads:")
    for name, info in TALAS.items():
        print(f"  {info['color']} {name:8s} "
              f"cycle={info['beats']} beats  →  {info['role']}")

    print()
    print("✅ Chakra Core + Tala Threading: OPERATIONAL")
    print(f"   Budget remaining for RNN+Attention backbone: "
          f"{16 - params['size_mb']:.2f} MB")
