from __future__ import annotations

import os
from dataclasses import dataclass
from collections import Counter, defaultdict
import multiprocessing as mp
import regex as re
from typing import BinaryIO, Iterable, Iterator


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

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


def count_chunk(start: int, end: int, path: str, pat: str, specials: list[str]) -> Counter:
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
        split_pat = "|".join(re.escape(tok) for tok in specials) # escape special tokens since some have "|" in them
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

    def __init__(self, vocab: Vocab, merges: list[BytePair], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.bytes_to_id = {b: i for i, b in vocab}
        # Ranks: lower index = higher priority during merging
        self.ranks = {pair: i for i, pair in enumerate(merges)}

    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str]=None):
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
             # don't forget () to keep the special tokens as a segment
            split_pat = "(" + "|".join(re.escape(tok) for tok in self.special_tokens) + ")"
            segments = re.split(split_pat, text)
        else:
            segments = [text]

        ids = []
        for seg in segments:
            if not seg:
                continue
            if seg in self.special_tokens:
                ids.append(self.bytes_to_id[seg.encode("utf-8")])
                continue
            for match in re.finditer(PAT, seg):
                token_bytes = match.group(0).encode("utf-8")
                seq = [bytes([b]) for b in token_bytes]

                while True:
                    best_i = -1
                    best_rank = None
                    for i in range(len(seq) - 1):
                        r = self.ranks.get((seq[i], seq[i+1]))
                        if r is not None and (best_rank is None or r < best_rank):
                            best_rank = r
                            best_i = i
                    if best_i == -1:
                        break
                    seq[best_i:best_i+2] = [seq[best_i] + seq[best_i+1]]
                ids.extend(self.bytes_to_id[b] for b in seq)

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
        cpu = os.cpu_count() or 1
        num_chunks = max(1, cpu)
        boundaries = find_chunk_boundaries(fbin, num_chunks, split_token_bytes)
    spans = list(zip(boundaries[:-1], boundaries[1:]))
    pretokenized_counter = Counter()
    if spans:
        cpu = os.cpu_count() or 1
        workers = min(len(spans), max(1, cpu))
        if workers > 1:
            with mp.Pool(processes=workers) as pool:
                parts = pool.starmap(count_chunk, [(s, e, input_path, PAT, special_tokens) for s, e in spans])
            for c in parts:
                pretokenized_counter.update(c)
        else:
            # Single span or single worker fallback
            for s, e in spans:
                pretokenized_counter.update(count_chunk(s, e, input_path, PAT, special_tokens))

    pair_counts = Counter()
    pair_index = defaultdict(set) # pair -> words (token tuples) that contain the pair
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
                if i + 1 < len(word_seq) and word_seq[i] == best_pair[0] and word_seq[i + 1] == best_pair[1]:
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
    
if __name__ == "__main__":
    vocab, merges = train_bpe()
    print(vocab)
    print(merges)