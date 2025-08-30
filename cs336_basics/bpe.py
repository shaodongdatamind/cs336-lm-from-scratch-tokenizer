from __future__ import annotations

import os
from dataclasses import dataclass
from collections import Counter, defaultdict
import multiprocessing as mp
import regex as re
from typing import BinaryIO, Iterable, Iterator
import time
import pickle
import psutil
import numpy as np


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def count_chunk(
    start: int, end: int, path: str, pat: str, specials: list[str]
) -> Counter:
    """Count pre-tokens within a file slice [start, end). Minimal helper for multiprocessing."""
    counter = Counter()
    with open(path, "rb") as fh:
        fh.seek(start)
        raw = fh.read(end - start)
    text = raw.decode("utf-8", errors="ignore")
    # Normalize newlines so Windows CRLF does not introduce stray \r tokens
    # This ensures reproducible tokenization across platforms
    text = text.replace("\r\n", "\n").replace("\r", "")
    specials_set = set(specials)
    if specials:
        split_pat = "|".join(
            re.escape(tok) for tok in specials
        )  # escape special tokens since some have "|" in them
        segments = re.split(split_pat, text)
    else:
        segments = [text]
    for segment in segments:
        if not segment:
            continue
        for match in re.finditer(pat, segment):
            token_text = match.group(0)
            token_bytes = token_text.encode("utf-8")
            seq = tuple(bytes([b]) for b in token_bytes)
            counter[seq] += 1
    return counter


BytePair = tuple[bytes, bytes]
Vocab = dict[int, bytes]


@dataclass
class BPETokenizer:

    def __init__(
        self,
        vocab: Vocab,
        merges: list[BytePair],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.bytes_to_id = {b: i for i, b in vocab.items()}
        # Ranks: lower index = higher priority during merging
        self.ranks = {pair: i for i, pair in enumerate(merges)}

    @classmethod
    def from_files(
        cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] = None
    ):
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special
        tokens.
        """
        import pickle

        with open(vocab_filepath, "rb") as vf:
            vocab: dict[int, bytes] = pickle.load(vf)
        with open(merges_filepath, "rb") as mf:
            merges: list[BytePair] = pickle.load(mf)
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Encode text to token ids using BPE merges.
        """
        text = text.replace("\r\n", "\n").replace("\r", "")

        # Pattern for GPT-2 style pretokenization
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        # Split text into segments on special tokens
        if self.special_tokens:
            # keep specials as separate segments; prefer longer matches first
            toks = sorted(self.special_tokens, key=len, reverse=True)
            split_pat = "(" + "|".join(re.escape(tok) for tok in toks) + ")"
            segments = re.split(split_pat, text)
        else:
            segments = [text]

        ids = []
        for seg in segments:
            if not seg:
                continue
            if self.special_tokens and seg in self.special_tokens:
                ids.append(self.bytes_to_id[seg.encode("utf-8")])
                continue
            for match in re.finditer(PAT, seg):
                token_bytes = match.group(0).encode("utf-8")
                seq = [bytes([b]) for b in token_bytes]

                while True:
                    best_i = -1
                    best_rank = None
                    for i in range(len(seq) - 1):
                        r = self.ranks.get((seq[i], seq[i + 1]))
                        if r is not None and (best_rank is None or r < best_rank):
                            best_rank = r
                            best_i = i
                    if best_i == -1:
                        break
                    seq[best_i : best_i + 2] = [seq[best_i] + seq[best_i + 1]]
                ids.extend(self.bytes_to_id[b] for b in seq)

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs.
        This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
        """
        for chunk in iterable:
            for _id in self.encode(chunk):
                yield _id

    def decode(self, ids: list[int]) -> str:
        """
        Decode token ids back to text (UTF-8).
        """
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")


# Training API expected by tests
def train_bpe(
    input_path: str | os.PathLike = "data/TinyStoriesV2-GPT4-valid.txt",
    vocab_size: int = 1000,
    special_tokens: list[str] = ["<|endoftext|>", "<|startoftext|>"],
) -> tuple[Vocab, list[BytePair]]:
    """
    Learn BPE merges from corpus:
    - Initialize vocab with all single bytes (0..255) plus special tokens appended.
    - Pretokenize and build word frequency counts.
    - Repeatedly count adjacent pair frequencies, select the most frequent pair
      (break ties lexicographically), apply merge, and continue until the
      requested size is reached.
    - Return (id_to_bytes, merges).
    Keep this simple; implement the details yourself.
    """
    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    # Pattern for GPT-2 style pretokenization
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # Pretokenize and build word frequency counts
    # Chunked pretokenization (serial), align to the required split token
    split_token_bytes = b"<|endoftext|>"
    with open(input_path, "rb") as fbin:
        num_chunks = 100 # max(1, os.cpu_count())
        boundaries = find_chunk_boundaries(fbin, num_chunks, split_token_bytes)
    spans = list(zip(boundaries[:-1], boundaries[1:]))
    pretokenized_counter = Counter()
    if spans:
        cpu = max(1, os.cpu_count() - 2)
        workers = min(len(spans), max(1, cpu))
        if workers > 1:
            with mp.Pool(processes=workers) as pool:
                parts = pool.starmap(
                    count_chunk,
                    [(s, e, input_path, PAT, special_tokens) for s, e in spans],
                )
            for c in parts:
                pretokenized_counter.update(c)
        else:
            # Single span or single worker fallback
            for s, e in spans:
                pretokenized_counter.update(
                    count_chunk(s, e, input_path, PAT, special_tokens)
                )

    pair_counts = Counter()
    pair_index = defaultdict(set)  # pair -> words (token tuples) that contain the pair
    for word_seq, freq in pretokenized_counter.items():
        if len(word_seq) < 2:
            continue
        for pair in zip(word_seq, word_seq[1:]):
            pair_counts[pair] += freq
            pair_index[pair].add(word_seq)

    merges = []
    while len(vocab) < vocab_size:
        # pair_counts = Counter()
        # for word_seq, freq in pretokenized_counter.items():
        #     if len(word_seq) < 2:
        #         continue
        #     for i in range(len(word_seq) - 1):
        #         pair_counts[(word_seq[i], word_seq[i + 1])] += freq

        if not pair_counts:
            break

        # Select most frequent pair; break ties by lexicographically greatest pair
        max_count = max(pair_counts.values())
        candidates = [pair for pair, cnt in pair_counts.items() if cnt == max_count]
        best_pair = max(candidates)

        # Record merge and add merged token to vocab
        merges.append(best_pair)
        merged_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = merged_token

        # Find all word sequences that contain the best pair
        affected_words = pair_index.pop(best_pair, set())
        if not affected_words:
            continue

        updates = {}
        for word_seq in affected_words:
            freq = pretokenized_counter.pop(word_seq, 0)
            if freq == 0:
                continue
            # remove old pair contribution
            for pair in zip(word_seq, word_seq[1:]):
                pair_counts[pair] -= freq
                if pair_counts[pair] <= 0:
                    pair_counts.pop(pair, None)
                pair_index[pair].discard(word_seq)
            # merge occurrences of best_pair in word_seq
            merged_seq = []
            i = 0
            while i < len(word_seq):
                if (
                    i + 1 < len(word_seq)
                    and word_seq[i] == best_pair[0]
                    and word_seq[i + 1] == best_pair[1]
                ):
                    merged_seq.append(merged_token)
                    i += 2
                else:
                    merged_seq.append(word_seq[i])
                    i += 1
            merged_seq = tuple(merged_seq)
            updates[merged_seq] = updates.get(merged_seq, 0) + freq

        for w_new, freq in updates.items():
            prev_freq = pretokenized_counter.get(w_new, 0)
            for pair in zip(w_new, w_new[1:]):
                pair_counts[pair] += freq
                pair_index[pair].add(w_new)
            pretokenized_counter[w_new] = freq + prev_freq

    return vocab, merges


def tokenize_file_to_npy(
    tokenizer: BPETokenizer,
    input_path: str,
    output_npy_path: str,
    dtype: str = "uint16",
) -> int:
    """
    Tokenize a text file using the provided tokenizer and write tokens to a .npy file.
    Uses a memory-efficient two-pass approach:
    1) Count tokens
    2) Write tokens into an open_memmap .npy array

    Returns the number of tokens written.
    """
    dt = np.dtype(dtype)

    # Pass 1: count
    num_tokens = 0
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in tokenizer.encode_iterable(f):
            num_tokens += 1

    # Pass 2: write to .npy via open_memmap
    arr = np.lib.format.open_memmap(output_npy_path, mode="w+", dtype=dt, shape=(num_tokens,))
    i = 0
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        for tid in tokenizer.encode_iterable(f):
            arr[i] = tid
            i += 1
    # ensure file is flushed/closed
    del arr
    return num_tokens


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train BPE and serialize vocab/merges."
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        type=str,
        default=str(os.path.join("data", "TinyStoriesV2-GPT4-train.txt")),
        help="Path to training text file",
    )
    parser.add_argument(
        "--vocab-size",
        dest="vocab_size",
        type=int,
        default=10000,
        help="Target vocab size (including specials)",
    )
    parser.add_argument(
        "--special",
        dest="special_tokens",
        action="append",
        default=None,
        help="Special token to include (may be specified multiple times)",
    )
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        type=str,
        default="data",
        help="Directory to write serialized outputs",
    )
    parser.add_argument(
        "--prefix",
        dest="prefix",
        type=str,
        default="tinystories_bpe",
        help="Filename prefix for outputs",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss
    start_time = time.time()

    # Default special tokens if none provided
    specials = (
        args.special_tokens if args.special_tokens is not None else ["<|endoftext|>"]
    )
    if "<|endoftext|>" not in specials:
        specials.append("<|endoftext|>")

    vocab, merges = train_bpe(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=specials,
    )

    elapsed_s = time.time() - start_time
    rss_after = process.memory_info().rss
    peak_bytes = getattr(process.memory_info(), "peak_wset", None)

    vocab_path = os.path.join(args.out_dir, f"{args.prefix}_vocab.pkl")
    merges_path = os.path.join(args.out_dir, f"{args.prefix}_merges.pkl")
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)

    # Compute longest token by byte length
    longest_token_bytes = max(vocab.values(), key=len)
    longest_token_len = len(longest_token_bytes)
    longest_token_text = longest_token_bytes.decode("utf-8", errors="replace")

    print("BPE training complete")
    print(f"Input: {args.input_path}")
    print(
        f"Vocab size: {len(vocab)} (target {args.vocab_size}) | Merges learned: {len(merges)}"
    )
    print(f"Special tokens: {specials}")
    print(f"Serialized to: {vocab_path} and {merges_path}")
    print(f"Elapsed: {elapsed_s/3600:.3f} hours ({elapsed_s:.1f} seconds)")
    print(f"Memory RSS before: {rss_before/1e9:.3f} GB | after: {rss_after/1e9:.3f} GB")
    if peak_bytes is not None:
        print(f"Peak working set (Windows): {peak_bytes/1e9:.3f} GB")
    print(f"Longest token length (bytes): {longest_token_len}")
    print(f"Longest token (decoded): {longest_token_text!r}")
