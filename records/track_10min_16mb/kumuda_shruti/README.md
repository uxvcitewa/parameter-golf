# 🎵 Shruti — A Language Model Rooted in Ancient Indian Knowledge

> *"Shruti" (श्रुति) — Sanskrit for "that which is heard"*  
> *Also the microtonal unit of Carnatic music. Also the Vedic oral tradition.*  
> *Everything about this model is in that word.*

---

## What is Shruti?

Shruti is a submission to the **OpenAI Parameter Golf Competition** —  
a language model that fits within **16MB**, trains in **under 10 minutes on 8×H100s**,  
and is evaluated on **bits-per-byte (bpb)** compression of FineWeb text.

But unlike every other submission, Shruti's architecture wasn't designed in a lab.  
It was designed by looking back **1,200 years** at one of the most extraordinary  
intellectual achievements in human history.

---

## The Inspiration — Siri Bhoovalaya (9th century, Karnataka)

In the 9th century CE, a Jain monk named **Kumudendu Muni** from Karnataka, India  
wrote a text called **Siri Bhoovalaya**.

It is unlike any other book ever written.

- Written entirely in **Kannada numerals (1–64)** — no letters, only numbers
- Arranged in **27×27 mathematical grids** called **Chakras**
- The same grid of numbers, read using **different traversal keys (Bandhas)**,  
  yields text in **700+ languages** — Kannada, Sanskrit, Tamil, Telugu, Marathi and more
- Contains an estimated **600,000 verses** across 56 chapters
- Uses **substitution, transposition, permutation, and steganography**  
  — every building block of modern cryptography

Kumudendu Muni built a **universal encoding system** where a compact set of 64 symbols  
could represent all of human language, with the meaning determined by the key used to read it.

Sound familiar?

**That is exactly what a language model does.**

Shruti is our attempt to build a modern neural architecture  
that embodies the same principle Kumudendu discovered 1,200 years ago.

---

## The Second Inspiration — Carnatic Music

Carnatic classical music (South Indian) has refined the mathematics of sound  
for over 2,000 years. Its key contributions to Shruti:

### Tala — Rhythmic Cycles
Carnatic music uses **Tala** (rhythmic cycles) of different lengths running simultaneously.  
Our model processes phoneme sequences through **4 parallel threads**, each with a  
different Tala cycle length:

| Tala | Beats | Linguistic role |
|------|-------|-----------------|
| Khanda | 5 | Clause boundaries |
| Rupaka | 6 | Phrase rhythm |
| Misra | 7 | Coarticulation |
| Adi | 8 | Word-level patterns |

These cycle lengths (5, 6, 7, 8) have LCM = **840** — meaning the threads  
stay out of phase for 840 tokens, giving maximum non-redundant coverage.

### Gamaka — Smooth Transitions
Carnatic Gamakas are ornamental slides between notes —  
the continuous flow between discrete pitches.  
Our **Gamaka Position Encoding** similarly encodes not just position,  
but *where each token falls within each Tala cycle* —  
giving the model rhythmic awareness of phoneme flow.

### Shruti — The Microtone
Carnatic music divides an octave into **22 Shrutis** (microtones)  
— far finer than Western music's 12 semitones.  
Our model similarly works at the **phoneme level** — finer than words or subwords —  
capturing the microtonal detail of language that BPE tokenizers discard.

---

## Architecture

```
Text Input
    │
    ▼
┌─────────────────────────────────────────────┐
│           CHAKRA CORE (Siri Bhoovalaya)      │
│                                             │
│  Hardcoded G2P rules → 64 phoneme tokens   │
│  Vocab size: 64 (Kumudendu's sacred number) │
│  vs baseline: 1024 tokens                  │
│  Embedding table: 98.4% smaller            │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│         TALA THREADING (Carnatic Music)      │
│                                             │
│  4 parallel GRU + Local Attention threads  │
│  Cycles: 5 | 6 | 7 | 8 (Khanda→Adi)       │
│  Gamaka positional encoding                │
│  Bandha mixer: learned thread combination  │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│        SHRUTI BACKBONE (Modern ML)           │
│                                             │
│  6× ShrutiBlocks (depth recurrence)        │
│  Each block: GRU → BandhaAttention → FFN   │
│  Shared GRU weights across all blocks      │
│  (like Chakra reusing 64 symbols)          │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
              Next phoneme logits
```

### Why phonemes instead of BPE?

| Tokenization | Vocab size | Embedding params |
|-------------|------------|-----------------|
| GPT-2 BPE | 50,257 | 38,597,376 |
| Baseline (competition) | 1,024 | 524,288 |
| **Shruti (phonemes)** | **64** | **8,192** |

Human languages have ~100 universal phonemes (IPA).  
English uses 44.  
**All meaning is downstream of sound.**  
Tokenizing at the sound level is linguistically correct —  
and dramatically more parameter-efficient.

---

## Parameter Budget

| Component | Params | Size |
|-----------|--------|------|
| Chakra embedding (64×128) | 8,192 | 0.03 MB |
| Tala threading (2 layers) | 1,578,496 | 6.02 MB |
| Shared GRU backbone | 98,688 | 0.38 MB |
| Bandha attention + FFN | 791,232 | 3.02 MB |
| Output head | 8,448 | 0.03 MB |
| **TOTAL** | **2,485,056** | **9.48 MB** |
| Code (~4 files) | — | ~0.08 MB |
| **ARTIFACT TOTAL** | — | **~9.56 MB / 16 MB ✅** |

---

## Key Innovations

### 1. Chakra-64 Tokenizer
A hardcoded grapheme-to-phoneme (G2P) engine converting English text  
to 64 IPA phoneme tokens. No learned tokenizer. No vocabulary file.  
Pure rules, ~50KB footprint.  
Includes 8 Kannada-specific phonemes (ಟ ಡ ಣ ಳ ಱ ಂ ಃ ೦) for future multilingual expansion.

### 2. Tala Threading
4 structurally distinct GRU+attention threads with coprime-ish cycle lengths.  
Unlike transformer attention heads (identical structure, learned differentiation),  
Tala threads are **born different** — capturing different temporal scales by design.

### 3. Bandha Attention
Multi-head causal attention where each head is assigned a Tala cycle.  
Queries are modulated by a learned phase gate — the head "knows"  
where it is in its rhythmic cycle and adjusts accordingly.  
Inspired directly by Siri Bhoovalaya's Bandha traversal keys.

### 4. Depth Recurrence via Weight Tying
All 6 backbone GRU layers **share weights** — like Kumudendu's Chakra  
reusing the same 64 symbols across all 56 chapters.  
This gives us computational depth essentially for free within the 16MB budget.

### 5. Gamaka Position Encoding
Blends standard sinusoidal PE with Tala-phase encoding —  
each token's position is encoded both absolutely and  
relative to each of the 4 Tala cycles simultaneously.

---

## Running Shruti

```bash
# Install dependencies
pip install torch datasets

# Verify setup (no training)
python train.py --dry-run

# Local test (CPU, small config)
python train.py --local

# Full competition run (8×H100, ~10 min)
python train.py
```

### File structure
```
shruti/
├── chakra_core.py      # Chakra-64 tokenizer (Siri Bhoovalaya)
├── tala_threading.py   # Carnatic Tala threading + Gamaka PE
├── shruti_model.py     # Full Shruti model
└── train.py            # Training loop (FineWeb, bpb metric)
```

---

## The Deeper Point

Modern LLMs tokenize language into subword units — artifacts of statistical  
frequency in training data. They have no notion of *sound*, *rhythm*, or *phonetic structure*.

Kumudendu Muni understood in the 9th century what linguists formalized in the 20th:  
**all language is downstream of a small, universal set of sounds.**  
His 64 symbols weren't arbitrary — they were a phonetic basis set for human communication.

Carnatic music's Tala system understood something else:  
**language has rhythm, and rhythm has structure.**  
Different rhythmic cycles capture different levels of linguistic organization —  
from phoneme coarticulation to clause boundaries.

Shruti attempts to encode both of these insights into a neural architecture  
that fits in 16MB and trains in 10 minutes.

Whether it beats the baseline or not —  
the idea is 1,200 years old and came from Karnataka.

---

## Author

Built for the **OpenAI Model Craft: Parameter Golf** competition (2026).  
Inspired by **Kumudendu Muni** — Jain monk, mathematician, polyglot, cryptographer.  
Largely overlooked by history. Not anymore.

---

*"The same 64 symbols. Different keys. All of language."*  
*— Siri Bhoovalaya, 9th century CE*
