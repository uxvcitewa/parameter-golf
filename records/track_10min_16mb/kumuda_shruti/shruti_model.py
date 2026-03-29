"""
SHRUTI BACKBONE — Deep RNN + Attention
=======================================
The computational core of the Shruti architecture.

Sits on top of:
  Chakra Core (64 phoneme tokens)
  └── Tala Threading (4 parallel Carnatic rhythm threads)
      └── THIS: Deep RNN + Attention backbone
          └── Output head → next phoneme prediction

Design philosophy:
  Like a Carnatic vocalist — the Tala threads set the rhythmic
  framework, and the backbone improvises WITHIN that framework,
  using memory (RNN) and awareness (attention) simultaneously.

Budget: ~10MB for this entire backbone.
Target: Beat 1.2244 bpb on FineWeb validation set.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tala_threading import ChakraTalaEmbedding, TALA_CYCLES
from chakra_core import ChakraTokenizer, VOCAB_SIZE


# ─────────────────────────────────────────────
# BANDHA ATTENTION
# Position-aware attention inspired by
# Siri Bhoovalaya's Bandha traversal keys.
#
# Unlike standard attention where Q/K/V are
# position-blind, Bandha attention remaps
# queries based on their Tala phase —
# like a musician phrasing differently
# depending on where they are in the beat cycle.
# ─────────────────────────────────────────────

class BandhaAttention(nn.Module):
    """
    Multi-head causal attention with Bandha (Tala-phase) key remapping.

    For each attention head, we assign one of the 4 Tala cycles.
    The query is modulated by a phase-dependent gate —
    amplifying or dampening based on position in cycle.

    This gives each head a rhythmic "personality":
      Head group 0 → Khanda phase (5-beat awareness)
      Head group 1 → Rupaka phase (6-beat awareness)
      Head group 2 → Misra phase  (7-beat awareness)
      Head group 3 → Adi phase    (8-beat awareness)
    """

    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = math.sqrt(self.head_dim)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # Bandha phase gates — one per head
        # Small learnable scalar per (head, tala_cycle_position)
        # Maps: tala_cycle_max × n_heads scalar gates
        max_cycle = max(TALA_CYCLES)  # 8
        self.bandha_gate = nn.Parameter(
            torch.ones(n_heads, max_cycle)
        )

        # Assign each head a Tala cycle
        self.head_cycles = [
            TALA_CYCLES[i % len(TALA_CYCLES)]
            for i in range(n_heads)
        ]

    def _bandha_modulate(self, q: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Modulate queries by Tala phase gate.
        q: (batch, n_heads, seq_len, head_dim)
        """
        B, H, T, D = q.shape
        q = q.clone()  # avoid inplace on view
        for h in range(H):
            cycle = self.head_cycles[h]
            phases = torch.arange(T, device=q.device) % cycle
            gates = self.bandha_gate[h, :cycle]
            gate_vals = torch.sigmoid(gates[phases])
            q[:, h] = q[:, h] * gate_vals.view(1, T, 1)
        return q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, dim)"""
        B, T, D = x.shape

        # QKV projection
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.n_heads, self.head_dim)
                    .transpose(1, 2) for t in qkv]
        # q,k,v: (B, n_heads, T, head_dim)

        # Bandha modulation on queries
        q = self._bandha_modulate(q, T)

        # Causal attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device), diagonal=1
        )
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, v)  # (B, n_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


# ─────────────────────────────────────────────
# SHRUTI BLOCK
# One layer of the backbone.
# RNN first (memory) → Attention second (awareness)
# Like: remember the phoneme flow, then
#       become aware of long range patterns
# ─────────────────────────────────────────────

class ShrutiBlock(nn.Module):
    """
    One Shruti backbone block:

      Input
        │
        ├─ LayerNorm → GRU → residual    (memory pass)
        │
        ├─ LayerNorm → BandhaAttention → residual  (awareness pass)
        │
        └─ LayerNorm → FFN → residual    (synthesis pass)

    Weight tying between blocks: all ShrutiBlocks share
    their GRU weights (depth recurrence) — like Kumudendu's
    Chakra reusing the same 64 symbols across all chapters.
    """

    def __init__(self, dim: int, n_heads: int = 4,
                 ffn_mult: int = 2, shared_gru: nn.GRU = None):
        super().__init__()
        self.dim = dim

        # Layer norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        # GRU — shared across blocks (weight tying = free depth!)
        if shared_gru is not None:
            self.gru = shared_gru  # reuse weights
        else:
            self.gru = nn.GRU(dim, dim, batch_first=True)

        # Bandha attention
        self.attn = BandhaAttention(dim, n_heads)

        # FFN — small multiplier to stay within budget
        ffn_dim = dim * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim, bias=False),
            nn.GELU(),
            nn.Linear(ffn_dim, dim, bias=False),
        )

    def forward(self, x: torch.Tensor, hidden=None):
        """
        x: (batch, seq_len, dim)
        hidden: GRU hidden state (carried across blocks)
        """
        # Memory pass — GRU
        normed = self.norm1(x)
        gru_out, hidden = self.gru(normed, hidden)
        x = x + gru_out

        # Awareness pass — Bandha attention
        x = x + self.attn(self.norm2(x))

        # Synthesis pass — FFN
        x = x + self.ffn(self.norm3(x))

        return x, hidden


# ─────────────────────────────────────────────
# SHRUTI MODEL — Full architecture
# Chakra → Tala → Backbone → Output head
# ─────────────────────────────────────────────

class ShrutiModel(nn.Module):
    """
    Complete Shruti Language Model.

    Named after:
    - Shruti (श्रुति) = "that which is heard" in Sanskrit
    - Shruti = microtonal unit in Carnatic music (22 per octave)
    - Shruti = the Vedic oral tradition of preserved knowledge

    Architecture lineage:
    - Siri Bhoovalaya (9th c.) → 64-symbol Chakra Core
    - Carnatic Music theory   → Tala Threading + Gamaka PE
    - Modern ML               → GRU + Bandha Attention
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,  # 64
        dim: int = 128,
        n_backbone_layers: int = 6,
        n_heads: int = 4,
        ffn_mult: int = 2,
        n_tala_layers: int = 2,
        dropout: float = 0.1,
        tie_gru_weights: bool = True,  # depth recurrence
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size

        # ── Input stack ──
        self.embedding_stack = ChakraTalaEmbedding(
            vocab_size=vocab_size,
            dim=dim,
            num_tala_layers=n_tala_layers,
            dropout=dropout,
        )

        # ── Shared GRU (weight tying across all blocks) ──
        # This is our "Chakra reuse" — same weights, different passes
        # Gives us depth for free within the 16MB budget
        if tie_gru_weights:
            shared_gru = nn.GRU(dim, dim, batch_first=True)
        else:
            shared_gru = None

        # ── Backbone blocks ──
        self.blocks = nn.ModuleList([
            ShrutiBlock(
                dim=dim,
                n_heads=n_heads,
                ffn_mult=ffn_mult,
                shared_gru=shared_gru,
            )
            for _ in range(n_backbone_layers)
        ])

        # Store shared GRU separately so it's registered
        if tie_gru_weights:
            self.shared_gru = shared_gru

        # ── Output head ──
        self.output_norm = nn.LayerNorm(dim)
        self.output_head = nn.Linear(dim, vocab_size, bias=False)

        # ── Weight init ──
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "embed" in name:
                nn.init.normal_(p, 0, 0.02)
            elif p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "norm" in name:
                if "weight" in name:
                    nn.init.ones_(p)
                else:
                    nn.init.zeros_(p)

    def forward(self, token_ids: torch.Tensor,
                hidden_states=None) -> tuple:
        """
        token_ids:    (batch, seq_len)
        hidden_states: list of GRU hiddens per block (or None)

        Returns:
          logits:       (batch, seq_len, vocab_size)
          hidden_states: list of hidden states (for generation)
        """
        # Input stack: tokens → rich phoneme representations
        x = self.embedding_stack(token_ids)

        # Backbone: deep RNN + attention
        if hidden_states is None:
            hidden_states = [None] * len(self.blocks)

        new_hiddens = []
        for i, block in enumerate(self.blocks):
            x, h = block(x, hidden_states[i])
            new_hiddens.append(h)

        # Output head: dim → vocab_size logits
        x = self.output_norm(x)
        logits = self.output_head(x)

        return logits, new_hiddens

    def loss(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute cross entropy loss for language modeling.
        Input:  token_ids (batch, seq_len)
        Target: token_ids shifted by 1 (next token prediction)
        """
        logits, _ = self.forward(token_ids[:, :-1])
        targets = token_ids[:, 1:].contiguous()
        return F.cross_entropy(
            logits.view(-1, self.vocab_size),
            targets.view(-1),
            ignore_index=ChakraTokenizer().pad,
        )

    def bits_per_byte(self, token_ids: torch.Tensor,
                      text_bytes: int) -> float:
        """
        Compute bits-per-byte — the competition metric.
        Lower is better. Baseline = 1.2244 bpb.
        """
        with torch.no_grad():
            loss = self.loss(token_ids)
        # Convert nats to bits, normalize by bytes
        return (loss.item() * math.log2(math.e)) * token_ids.numel() / text_bytes

    def param_count(self) -> dict:
        groups = {
            "chakra_embedding": self.embedding_stack.chakra_embed,
            "tala_threading":   self.embedding_stack.tala_layers,
            "backbone_attn":    nn.ModuleList([b.attn for b in self.blocks]),
            "backbone_ffn":     nn.ModuleList([b.ffn for b in self.blocks]),
            "shared_gru":       getattr(self, "shared_gru", None),
            "output_head":      self.output_head,
        }
        counts = {}
        for name, mod in groups.items():
            if mod is not None:
                counts[name] = sum(p.numel() for p in mod.parameters())
        counts["total"] = sum(p.numel() for p in self.parameters())
        counts["size_mb"] = counts["total"] * 4 / 1024 / 1024
        return counts

    def __repr__(self):
        p = self.param_count()
        return (
            f"ShrutiModel(\n"
            f"  vocab={self.vocab_size} (Chakra-64)\n"
            f"  dim={self.dim}\n"
            f"  backbone_layers={len(self.blocks)}\n"
            f"  total_params={p['total']:,}\n"
            f"  size={p['size_mb']:.2f} MB\n"
            f")"
        )


# ─────────────────────────────────────────────
# QUICK TEST — no GPU needed
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("SHRUTI MODEL — Full Architecture Test")
    print("=" * 60)

    tok = ChakraTokenizer()

    # Build model
    model = ShrutiModel(
        vocab_size=VOCAB_SIZE,
        dim=128,
        n_backbone_layers=6,
        n_heads=4,
        ffn_mult=2,
        n_tala_layers=2,
        tie_gru_weights=True,
    )

    print(model)
    print()

    # Parameter breakdown
    p = model.param_count()
    print("Parameter breakdown:")
    for k, v in p.items():
        if k == "size_mb":
            print(f"  {'size':20s}: {v:.3f} MB")
        elif k == "total":
            print(f"  {'TOTAL':20s}: {v:>10,}")
        else:
            print(f"  {k:20s}: {v:>10,}")

    print()

    # Check if within 16MB budget
    size = p["size_mb"]
    budget = 16.0
    status = "✅ WITHIN BUDGET" if size < budget else "❌ OVER BUDGET"
    print(f"Budget: {size:.2f} MB / {budget:.1f} MB  {status}")
    print()

    # Forward pass test
    sentences = [
        "the cat sat on the mat",
        "shruti is a sound in music",
    ]
    encoded = [tok.encode(s) for s in sentences]
    max_len = max(len(e) for e in encoded)
    padded = [e + [tok.pad] * (max_len - len(e)) for e in encoded]
    token_ids = torch.tensor(padded)

    print(f"Input:  {token_ids.shape}  (batch=2, seq={max_len})")

    with torch.no_grad():
        logits, hiddens = model(token_ids)

    print(f"Output: {logits.shape}  (batch=2, seq={max_len}, vocab=64)")
    print()

    # Loss test
    with torch.no_grad():
        loss = model.loss(token_ids)
    print(f"Initial loss: {loss.item():.4f} nats")
    print(f"In bits/byte: {loss.item() * math.log2(math.e):.4f} bpb (untrained)")
    print()
    print("Target: beat 1.2244 bpb after training on FineWeb")
    print()
    print("✅ Shruti Model: FULLY OPERATIONAL")
    print()
    print("Architecture lineage:")
    print("  Kumudendu Muni (9th c. Karnataka) → Chakra-64 token basis")
    print("  Carnatic Tala system              → Parallel rhythm threading")
    print("  Siri Bhoovalaya Bandha keys       → Phase-aware attention")
    print("  Modern GRU + Attention            → Sequence modeling backbone")
