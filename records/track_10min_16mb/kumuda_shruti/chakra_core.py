"""
CHAKRA CORE — Inspired by Siri Bhoovalaya (Kumudendu Muni, 9th century)
========================================================================
The atomic unit of our "Shruti" architecture.

Just as Kumudendu's 64-symbol Chakra grid encodes all language through
mathematical traversal (Bandha), our Chakra Core maps all English phonemes
to 64 base tokens — with room to expand to Kannada and other languages.

Architecture role: CPU Core (everything else threads through this)
"""

# ─────────────────────────────────────────────
# THE 64-SYMBOL CHAKRA
# Organized like Kumudendu's grid:
# Rows = phoneme class, Cols = variants
# ─────────────────────────────────────────────

CHAKRA_64 = {

    # ── VOWELS (16 symbols) ── Row 0-1
    # Short vowels
    0:  ("ɪ",  "short_i",   "bit"),
    1:  ("ɛ",  "short_e",   "bet"),
    2:  ("æ",  "short_a",   "bat"),
    3:  ("ʌ",  "short_u",   "but"),
    4:  ("ɒ",  "short_o",   "bot"),
    5:  ("ʊ",  "short_oo",  "book"),
    6:  ("ə",  "schwa",     "about"),
    7:  ("ɜ",  "er",        "bird"),
    # Long vowels
    8:  ("iː", "long_i",    "beet"),
    9:  ("eɪ", "long_a",    "bait"),
    10: ("ɑː", "long_ah",   "bar"),
    11: ("ɔː", "long_aw",   "bore"),
    12: ("uː", "long_oo",   "boot"),
    # Diphthongs
    13: ("aɪ", "diph_ai",   "bite"),
    14: ("aʊ", "diph_ow",   "bout"),
    15: ("ɔɪ", "diph_oi",   "boy"),

    # ── STOPS (8 symbols) ── Row 2
    16: ("p",  "stop_p",    "pat"),
    17: ("b",  "stop_b",    "bat"),
    18: ("t",  "stop_t",    "tap"),
    19: ("d",  "stop_d",    "dap"),
    20: ("k",  "stop_k",    "cat"),
    21: ("g",  "stop_g",    "gap"),
    22: ("ʔ",  "glottal",   "uh-oh"),
    23: ("ʈ",  "retro_t",   "kannada_T"),   # Kannada ಟ

    # ── FRICATIVES (12 symbols) ── Row 3-4
    24: ("f",  "fric_f",    "fat"),
    25: ("v",  "fric_v",    "vat"),
    26: ("θ",  "fric_th",   "thin"),
    27: ("ð",  "fric_dh",   "then"),
    28: ("s",  "fric_s",    "sat"),
    29: ("z",  "fric_z",    "zap"),
    30: ("ʃ",  "fric_sh",   "ship"),
    31: ("ʒ",  "fric_zh",   "vision"),
    32: ("h",  "fric_h",    "hat"),
    33: ("x",  "fric_kh",   "loch"),
    34: ("ɣ",  "fric_gh",   "kannada_gh"),  # Kannada ಘ
    35: ("ç",  "fric_hy",   "huge"),

    # ── NASALS (6 symbols) ── Row 5
    36: ("m",  "nasal_m",   "mat"),
    37: ("n",  "nasal_n",   "nat"),
    38: ("ŋ",  "nasal_ng",  "sing"),
    39: ("ɲ",  "nasal_ny",  "kannada_ny"),  # Kannada ಞ
    40: ("ɳ",  "nasal_rn",  "kannada_N"),   # Kannada ಣ
    41: ("ṃ",  "anusvara",  "kannada_M"),   # Kannada ಂ

    # ── LIQUIDS (6 symbols) ── Row 6
    42: ("l",  "liquid_l",  "lat"),
    43: ("r",  "liquid_r",  "rat"),
    44: ("ɾ",  "flap_r",    "butter"),
    45: ("ɭ",  "retro_l",   "kannada_L"),   # Kannada ಳ
    46: ("ɽ",  "retro_r",   "kannada_R"),   # Kannada ಱ
    47: ("ɬ",  "lat_fric",  "welsh_ll"),

    # ── AFFRICATES (4 symbols) ── Row 7
    48: ("tʃ", "affr_ch",   "chip"),
    49: ("dʒ", "affr_j",    "jet"),
    50: ("ts", "affr_ts",   "bits"),
    51: ("dz", "affr_dz",   "adze"),

    # ── SEMIVOWELS (4 symbols) ── Row 8
    52: ("w",  "semi_w",    "wet"),
    53: ("j",  "semi_y",    "yet"),
    54: ("ɥ",  "semi_hy",   "french_u"),
    55: ("ɰ",  "semi_ɰ",    "korean_eu"),

    # ── SPECIAL / CONTROL TOKENS (8 symbols) ── Row 9
    # These are like Kumudendu's control symbols in the Chakra
    56: ("[SOS]",   "start",      "start of sequence"),
    57: ("[EOS]",   "end",        "end of sequence"),
    58: ("[PAD]",   "pad",        "padding"),
    59: ("[UNK]",   "unknown",    "unknown sound"),
    60: ("[STRESS]","stress",     "primary stress marker"),
    61: ("[BOUND]", "boundary",   "word boundary"),
    62: ("[PAUSE]", "pause",      "short pause / comma"),
    63: ("[STOP]",  "full_stop",  "sentence end / period"),
}

# ─────────────────────────────────────────────
# REVERSE LOOKUP
# ─────────────────────────────────────────────
IPA_TO_ID = {v[0]: k for k, v in CHAKRA_64.items()}
NAME_TO_ID = {v[1]: k for k, v in CHAKRA_64.items()}
VOCAB_SIZE = 64  # The sacred number — Kumudendu's Chakra


# ─────────────────────────────────────────────
# HARDCODED G2P ENGINE
# Grapheme-to-Phoneme — English rules only
# Tiny footprint, no ML needed (~50KB)
# ─────────────────────────────────────────────

# Digraph rules (must be checked BEFORE single char rules)
DIGRAPH_MAP = {
    "sh": 30,   # ʃ  — ship
    "ch": 48,   # tʃ — chip
    "th": 26,   # θ  — thin (voiced 'th' handled separately)
    "dh": 27,   # ð  — then
    "ph": 24,   # f  — phone
    "wh": 52,   # w  — what
    "ng": 38,   # ŋ  — sing
    "zh": 31,   # ʒ  — vision
    "gh": 34,   # ɣ  — ghost (silent in some cases)
    "ck": 20,   # k  — back
    "qu": 20,   # k  — queen (simplified)
    "ee": 8,    # iː — bee
    "ea": 8,    # iː — meat (simplified)
    "oo": 12,   # uː — boot
    "ou": 14,   # aʊ — out
    "ow": 14,   # aʊ — cow
    "oi": 15,   # ɔɪ — boy
    "oy": 15,   # ɔɪ — toy
    "ai": 9,    # eɪ — rain
    "ay": 9,    # eɪ — day
    "au": 11,   # ɔː — haul
    "aw": 11,   # ɔː — law
    "ew": 12,   # uː — few
    "ue": 12,   # uː — blue
    "ie": 13,   # aɪ — pie
    "igh": 13,  # aɪ — night
}

# Single character map (fallback)
CHAR_MAP = {
    "a": 2,    # æ  — cat
    "e": 1,    # ɛ  — bet
    "i": 0,    # ɪ  — bit
    "o": 4,    # ɒ  — bot
    "u": 3,    # ʌ  — but
    "b": 17,
    "c": 20,   # k sound default
    "d": 19,
    "f": 24,
    "g": 21,
    "h": 32,
    "j": 49,
    "k": 20,
    "l": 42,
    "m": 36,
    "n": 37,
    "p": 16,
    "q": 20,
    "r": 43,
    "s": 28,
    "t": 18,
    "v": 25,
    "w": 52,
    "x": 28,   # ks → simplified to s
    "y": 53,
    "z": 29,
    " ": 61,   # word boundary
    ".": 63,   # sentence end
    ",": 62,   # pause
}

# Voiced 'th' contexts — "the", "this", "that", "they", etc.
VOICED_TH_WORDS = {"the", "this", "that", "they", "them",
                   "their", "there", "these", "those", "though",
                   "then", "than", "thus", "thereof"}


# ─────────────────────────────────────────────
# CHAKRA TOKENIZER CLASS
# ─────────────────────────────────────────────

class ChakraTokenizer:
    """
    Converts English text → 64-symbol Chakra token IDs.
    
    Inspired by Siri Bhoovalaya's principle:
    The same underlying symbols encode all language —
    the traversal key (Bandha) determines the meaning.
    """

    def __init__(self):
        self.vocab_size = VOCAB_SIZE
        self.sos = NAME_TO_ID["start"]
        self.eos = NAME_TO_ID["end"]
        self.pad = NAME_TO_ID["pad"]
        self.unk = NAME_TO_ID["unknown"]
        self.stress = NAME_TO_ID["stress"]
        self.boundary = NAME_TO_ID["boundary"]

    def _text_to_phoneme_ids(self, text: str) -> list[int]:
        """Hardcoded G2P: text → phoneme token IDs"""
        text = text.lower().strip()
        tokens = []
        words = text.split()

        for word in words:
            # Check voiced 'th' words
            use_voiced_th = word in VOICED_TH_WORDS

            i = 0
            # Try trigraph first (igh)
            while i < len(word):
                matched = False

                # Trigraph
                if i + 2 < len(word):
                    trigraph = word[i:i+3]
                    if trigraph in DIGRAPH_MAP:
                        tokens.append(DIGRAPH_MAP[trigraph])
                        i += 3
                        matched = True

                # Digraph
                if not matched and i + 1 < len(word):
                    digraph = word[i:i+2]
                    # Handle voiced vs unvoiced th
                    if digraph == "th":
                        tokens.append(27 if use_voiced_th else 26)
                        i += 2
                        matched = True
                    elif digraph in DIGRAPH_MAP:
                        tokens.append(DIGRAPH_MAP[digraph])
                        i += 2
                        matched = True

                # Single char
                if not matched:
                    char = word[i]
                    if char in CHAR_MAP:
                        tokens.append(CHAR_MAP[char])
                    else:
                        tokens.append(self.unk)
                    i += 1

            # Word boundary after each word
            tokens.append(self.boundary)

        return tokens

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Full encode pipeline: text → Chakra IDs"""
        tokens = self._text_to_phoneme_ids(text)
        if add_special_tokens:
            tokens = [self.sos] + tokens + [self.eos]
        return tokens

    def decode(self, ids: list[int]) -> str:
        """Chakra IDs → IPA phoneme string (for inspection)"""
        return " ".join(
            CHAKRA_64[i][0] if i in CHAKRA_64 else "?"
            for i in ids
            if i not in (self.sos, self.eos, self.pad)
        )

    def token_name(self, id: int) -> str:
        return CHAKRA_64.get(id, ("?", "unknown", "?"))[1]

    def __repr__(self):
        return f"ChakraTokenizer(vocab_size={self.vocab_size}, inspired_by='Siri Bhoovalaya')"


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    tok = ChakraTokenizer()
    print(tok)
    print()

    test_sentences = [
        "the cat sat on the mat",
        "ship shape shore",
        "this and that",
        "shruti is a sound",
    ]

    for sentence in test_sentences:
        ids = tok.encode(sentence)
        ipa = tok.decode(ids)
        print(f"INPUT : {sentence}")
        print(f"IDs   : {ids}")
        print(f"IPA   : {ipa}")
        print(f"Tokens: {len(ids)} (compression: {len(sentence.split())} words → {len(ids)} phoneme tokens)")
        print()
