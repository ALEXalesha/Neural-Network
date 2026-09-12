"""Общее для обучающих скриптов: загрузка датасетов с Hugging Face и токенизация."""
import re

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

# Та же токенизация, что в alexgpt/app_server.py
TOKEN_RE = re.compile(r"\w+(?:[-'’]\w+)*|[^\w\s]")


def tokenize(text):
    return TOKEN_RE.findall(text)


def load_parquet(repo, filename, revision=None):
    return pq.read_table(hf_hub_download(repo, filename, repo_type="dataset", revision=revision)).to_pylist()


def build_vocab(token_lists, max_size, min_freq=2):
    from collections import Counter
    cnt = Counter(t for toks in token_lists for t in toks)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for w, c in cnt.most_common(max_size - 2):
        if c < min_freq:
            break
        vocab[w] = len(vocab)
    return vocab
