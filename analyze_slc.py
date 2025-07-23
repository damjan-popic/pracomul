#!/usr/bin/env python
"""
Analyze SLC-style dialogue corpus (≈108k words)
==============================================

Fixes in this version (v5)
--------------------------
* **No LexicalRichness attr errors** – `root_ttr` is computed manually; all LR calls are wrapped in `safe()`.
* **Zero-length turns handled** – all metrics become 0.0 safely.
* **Sentence statistics added** – per turn: `sent_count`, `sent_len_mean`, `sent_len_median`, `sent_len_std`, `sent_len_min`, `sent_len_max` (token counts).
* **Long format output** – `turns_long.csv` with generic columns (`age`, `gender`, `nationality`, `university`, `spanish_level`, `mother_tongue`) so you can aggregate “all Slovenians” irrespective of A/B.
* **Robust aggregations** – only numeric columns are aggregated; no more "unsupported operand type(s) for +: 'method'".
* **Still no YAML/env nonsense** – just run: `python analyze_slc.py data/corpus.txt results`.

Outputs
-------
results/
    turns.csv                  # wide: keeps both A/B meta
    turns_long.csv             # long: single set of attrs per row
    tokens.csv                 # one token per row
    aggregates_overall.csv
    aggregates_by_doc.csv
    aggregates_by_speaker.csv  # A vs B
    aggregates_by_<attr>.csv   # nationality, university, age, gender, etc.
    aggregates_by_country_combo.csv  # doc-level A_B nationality pair

"""

from __future__ import annotations
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from lexicalrichness import LexicalRichness
import spacy
from tqdm import tqdm

# ------------------------- Regexes -------------------------
DOC_OPEN_RE = re.compile(r"<doc\s+([^>]+)>")
ATTR_RE = re.compile(r"(\w+)=\"([^\"]*)\"")
TEXT_OPEN_RE = re.compile(r"<text>")
TEXT_CLOSE_RE = re.compile(r"</text>")
TAG_SPK_RE = re.compile(r"<(A|B)>")
TAG_SPK_CLOSE_RE = re.compile(r"</(A|B)>")

ASTERISK_SPLIT_RE = re.compile(r"([^*]+)\*([^*]+)")
PAREN_EVENT_RE = re.compile(r"\([^)]*\)")  # (RISAS), etc.
ARROW_RE = re.compile(r"<+|>+|→|←|↗|↘|↔|↕")
MULTISPACE_RE = re.compile(r"\s+")

# Pragmatic feature heuristics (add more if needed)
PRAG_FEATURES = {
    "laughter": re.compile(r"\(risas\)", re.I),
    "filled_pause": re.compile(r"\b(e+hm+|mm+|aa+a+)\b", re.I),
    "thanks": re.compile(r"\bgracias\b", re.I),
    "question_marks": re.compile(r"\?"),
}

# ------------------------- Data classes -------------------------
@dataclass
class Turn:
    doc_id: str
    speaker: str  # 'A' or 'B'
    turn_id: int
    raw_text: str
    normalized_text: str
    token_count: int
    type_count: int
    ttr: float
    root_ttr: float
    maas: float
    hdd: float
    mtld: float
    mtld_ma_wrap: float
    msttr_50: float
    # Sentence stats
    sent_count: int
    sent_len_mean: float
    sent_len_median: float
    sent_len_std: float
    sent_len_min: int
    sent_len_max: int
    # Pragmatic counts
    laughter: int
    filled_pause: int
    thanks: int
    question_marks: int
    # Generic meta for this speaker
    age: str
    gender: str
    mother_tongue: str
    spanish_level: str
    nationality: str
    university: str
    # Full doc-level meta (kept for reference)
    age_A: str
    gender_A: str
    mother_tongue_A: str
    spanish_level_A: str
    nationality_A: str
    university_A: str
    age_B: str
    gender_B: str
    mother_tongue_B: str
    spanish_level_B: str
    nationality_B: str
    university_B: str

@dataclass
class TokenRow:
    doc_id: str
    speaker: str
    turn_id: int
    idx: int
    token: str
    lemma: str
    pos: str
    tag: str
    dep: str
    is_alpha: bool
    sent_id: int

# ------------------------- Helpers -------------------------
def clean_text(s: str) -> str:
    s = PAREN_EVENT_RE.sub(" ", s)
    s = ARROW_RE.sub(" ", s)
    # replace informal*formal with the formal side
    def repl(m):
        informal, formal = m.group(1).strip(), m.group(2).strip()
        return formal if formal else informal
    s = ASTERISK_SPLIT_RE.sub(repl, s)
    s = MULTISPACE_RE.sub(" ", s).strip()
    return s


def lexical_stats(text: str) -> Dict[str, float | int]:
    if not text:
        return dict(token_count=0, type_count=0, ttr=0.0, root_ttr=0.0, maas=0.0,
                    hdd=0.0, mtld=0.0, mtld_ma_wrap=0.0, msttr_50=0.0)
    lr = LexicalRichness(text)
    words = getattr(lr, "words", 0)
    types = getattr(lr, "terms", 0)
    if words == 0:
        return dict(token_count=0, type_count=0, ttr=0.0, root_ttr=0.0, maas=0.0,
                    hdd=0.0, mtld=0.0, mtld_ma_wrap=0.0, msttr_50=0.0)

    def safe(fn, default=0.0):
        try:
            return fn() if callable(fn) else fn
        except Exception:
            return default

    draws = min(42, max(1, words - 1))  # HDD constraint
    ttr_val = safe(getattr(lr, "ttr", 0.0))
    root_ttr_val = types / np.sqrt(words) if words else 0.0

    return dict(
        token_count=words,
        type_count=types,
        ttr=ttr_val,
        root_ttr=root_ttr_val,
        maas=safe(getattr(lr, "maas", 0.0)),
        hdd=safe(lambda: lr.hdd(draws=draws)),
        mtld=safe(getattr(lr, "mtld", 0.0)),
        mtld_ma_wrap=safe(getattr(lr, "mtld_ma_wrap", 0.0)),
        msttr_50=safe(lambda: lr.msttr(segment_window=50)),
    )


def sentence_stats(doc) -> Dict[str, float | int]:
    if doc is None or len(doc) == 0:
        return dict(sent_count=0, sent_len_mean=0.0, sent_len_median=0.0,
                    sent_len_std=0.0, sent_len_min=0, sent_len_max=0)
    lengths = [len([t for t in sent if not t.is_space]) for sent in doc.sents]
    if not lengths:
        return dict(sent_count=0, sent_len_mean=0.0, sent_len_median=0.0,
                    sent_len_std=0.0, sent_len_min=0, sent_len_max=0)
    arr = np.array(lengths)
    return dict(
        sent_count=len(arr),
        sent_len_mean=float(arr.mean()),
        sent_len_median=float(np.median(arr)),
        sent_len_std=float(arr.std(ddof=0)),
        sent_len_min=int(arr.min()),
        sent_len_max=int(arr.max()),
    )


def pragmatic_counts(text: str) -> Dict[str, int]:
    return {k: len(v.findall(text)) for k, v in PRAG_FEATURES.items()}


def parse_attrs(attr_str: str) -> Dict[str, str]:
    return {k: v for k, v in ATTR_RE.findall(attr_str)}

# ------------------------- Core parsing -------------------------
def process_turns(doc_attrs: Dict[str, str], nlp, text_block: str) -> Tuple[List[Turn], List[TokenRow]]:
    turns: List[Turn] = []
    tokens: List[TokenRow] = []

    current_speaker = None
    current_text_parts: List[str] = []
    turn_idx = 0

    def flush_turn():
        nonlocal turn_idx, current_text_parts, current_speaker
        if current_speaker is None:
            return
        raw = MULTISPACE_RE.sub(" ", " ".join(current_text_parts).strip())
        norm = clean_text(raw)
        lex = lexical_stats(norm)
        prag = pragmatic_counts(raw)
        doc = nlp(norm) if norm else None
        sent_stats = sentence_stats(doc)

        pref = current_speaker  # 'A' or 'B'
        def g(field: str):
            return doc_attrs.get(f"{field}_{pref}", "")

        if doc is not None:
            for i, tok in enumerate(doc):
                tokens.append(TokenRow(
                    doc_id=doc_attrs.get("id", ""),
                    speaker=current_speaker,
                    turn_id=turn_idx,
                    idx=i,
                    token=tok.text,
                    lemma=tok.lemma_,
                    pos=tok.pos_,
                    tag=tok.tag_,
                    dep=tok.dep_,
                    is_alpha=tok.is_alpha,
                    sent_id=tok.sent.start if tok.sent is not None else -1,
                ))

        t = Turn(
            doc_id=doc_attrs.get("id", ""),
            speaker=current_speaker,
            turn_id=turn_idx,
            raw_text=raw,
            normalized_text=norm,
            token_count=lex["token_count"],
            type_count=lex["type_count"],
            ttr=lex["ttr"],
            root_ttr=lex["root_ttr"],
            maas=lex["maas"],
            hdd=lex["hdd"],
            mtld=lex["mtld"],
            mtld_ma_wrap=lex["mtld_ma_wrap"],
            msttr_50=lex["msttr_50"],
            sent_count=sent_stats["sent_count"],
            sent_len_mean=sent_stats["sent_len_mean"],
            sent_len_median=sent_stats["sent_len_median"],
            sent_len_std=sent_stats["sent_len_std"],
            sent_len_min=sent_stats["sent_len_min"],
            sent_len_max=sent_stats["sent_len_max"],
            laughter=prag["laughter"],
            filled_pause=prag["filled_pause"],
            thanks=prag["thanks"],
            question_marks=prag["question_marks"],
            age=g("age"),
            gender=g("gender"),
            mother_tongue=g("mother_tongue"),
            spanish_level=g("spanish_level"),
            nationality=g("nationality"),
            university=g("university"),
            age_A=doc_attrs.get("age_A", ""),
            gender_A=doc_attrs.get("gender_A", ""),
            mother_tongue_A=doc_attrs.get("mother_tongue_A", ""),
            spanish_level_A=doc_attrs.get("spanish_level_A", ""),
            nationality_A=doc_attrs.get("nationality_A", ""),
            university_A=doc_attrs.get("university_A", ""),
            age_B=doc_attrs.get("age_B", ""),
            gender_B=doc_attrs.get("gender_B", ""),
            mother_tongue_B=doc_attrs.get("mother_tongue_B", ""),
            spanish_level_B=doc_attrs.get("spanish_level_B", ""),
            nationality_B=doc_attrs.get("nationality_B", ""),
            university_B=doc_attrs.get("university_B", ""),
        )
        turns.append(t)
        turn_idx += 1
        current_text_parts = []
        current_speaker = None

    pos = 0
    while True:
        open_match = TAG_SPK_RE.search(text_block, pos)
        if not open_match:
            break
        spk = open_match.group(1)
        close_match = TAG_SPK_CLOSE_RE.search(text_block, open_match.end())
        if not close_match:
            break
        content = text_block[open_match.end():close_match.start()]
        if current_speaker is None:
            current_speaker = spk
        elif current_speaker != spk:
            flush_turn()
            current_speaker = spk
        current_text_parts.append(content)
        pos = close_match.end()

    flush_turn()
    return turns, tokens

# ------------------------- Aggregations -------------------------
AGG_FUNCS = {
    "token_count": "sum",
    "type_count": "sum",
    "ttr": "mean",
    "root_ttr": "mean",
    "maas": "mean",
    "hdd": "mean",
    "mtld": "mean",
    "mtld_ma_wrap": "mean",
    "msttr_50": "mean",
    "sent_count": "sum",
    "sent_len_mean": "mean",
    "sent_len_median": "mean",
    "sent_len_std": "mean",
    "sent_len_min": "mean",
    "sent_len_max": "mean",
    "laughter": "sum",
    "filled_pause": "sum",
    "thanks": "sum",
    "question_marks": "sum",
}

def aggregate_and_save(turns_df: pd.DataFrame, out_dir: Path) -> None:
    num_cols = turns_df.select_dtypes(include=["number"]).columns
    agg_map = {k: v for k, v in AGG_FUNCS.items() if k in num_cols}

    turns_df[list(agg_map.keys())].agg(agg_map).to_csv(out_dir / "aggregates_overall.csv")
    turns_df.groupby("doc_id")[list(agg_map.keys())].agg(agg_map).to_csv(out_dir / "aggregates_by_doc.csv")
    turns_df.groupby("speaker")[list(agg_map.keys())].agg(agg_map).to_csv(out_dir / "aggregates_by_speaker.csv")

    def by(attr: str):
        if attr in turns_df.columns:
            turns_df.groupby(attr)[list(agg_map.keys())].agg(agg_map).to_csv(out_dir / f"aggregates_by_{attr}.csv")

    for attr in ["nationality", "university", "age", "gender", "spanish_level", "mother_tongue"]:
        by(attr)

    if {"nationality_A", "nationality_B"}.issubset(turns_df.columns):
        combo = turns_df.assign(country_combo=lambda d: d["nationality_A"] + "_" + d["nationality_B"])
        combo.groupby("country_combo")[list(agg_map.keys())].agg(agg_map).to_csv(out_dir / "aggregates_by_country_combo.csv")

# ------------------------- Reading -------------------------
def read_corpus(corpus_path: Path) -> Tuple[List[Turn], List[TokenRow]]:
    with corpus_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    nlp = spacy.load("es_core_news_md", disable=["ner"])
    if "lemmatizer" in nlp.pipe_names:
        nlp.enable_pipe("lemmatizer")
    if "senter" not in nlp.pipe_names and "parser" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    turns_all: List[Turn] = []
    tokens_all: List[TokenRow] = []

    inside_text = False
    current_doc_attrs: Dict[str, str] = {}
    text_buffer: List[str] = []

    total_docs = sum(1 for ln in lines if DOC_OPEN_RE.search(ln))
    pbar = tqdm(total=total_docs, desc="Docs")

    for ln in lines:
        doc_match = DOC_OPEN_RE.search(ln)
        if doc_match:
            current_doc_attrs = parse_attrs(doc_match.group(1))
            continue
        if TEXT_OPEN_RE.search(ln):
            inside_text = True
            text_buffer = []
            continue
        if TEXT_CLOSE_RE.search(ln):
            inside_text = False
            text_block = "".join(text_buffer)
            t, tk = process_turns(current_doc_attrs, nlp, text_block)
            turns_all.extend(t)
            tokens_all.extend(tk)
            pbar.update(1)
            continue
        if inside_text:
            text_buffer.append(ln)

    pbar.close()
    return turns_all, tokens_all

# ------------------------- Main -------------------------
def main(corpus_path: Path, out_dir: Path) -> None:
    print(f"Reading corpus from {corpus_path} ...")
    out_dir.mkdir(parents=True, exist_ok=True)

    turns, tokens = read_corpus(corpus_path)

    turns_df = pd.DataFrame([asdict(t) for t in turns])
    tokens_df = pd.DataFrame([asdict(tk) for tk in tokens])

    turns_df.to_csv(out_dir / "turns.csv", index=False)
    tokens_df.to_csv(out_dir / "tokens.csv", index=False)

    long_cols = [
        "doc_id", "speaker", "turn_id", "raw_text", "normalized_text",
        "token_count", "type_count", "ttr", "root_ttr", "maas", "hdd", "mtld", "mtld_ma_wrap", "msttr_50",
        "sent_count", "sent_len_mean", "sent_len_median", "sent_len_std", "sent_len_min", "sent_len_max",
        "laughter", "filled_pause", "thanks", "question_marks",
        "age", "gender", "mother_tongue", "spanish_level", "nationality", "university",
        "nationality_A", "nationality_B"
    ]
    turns_long = turns_df[long_cols].copy()
    turns_long.to_csv(out_dir / "turns_long.csv", index=False)

    aggregate_and_save(turns_long, out_dir)

    print("Done. Files written to", out_dir)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python analyze_slc.py <corpus.txt> <out_dir>")
        sys.exit(1)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
