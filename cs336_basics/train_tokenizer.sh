uv run python -m cs336_basics.bpe \
  --input data/TinyStoriesV2-GPT4.txt \
  --vocab-size 10000 \
  --special "<|endoftext|>" \
  --out-dir ./artifacts \
  --prefix TinyStoriesV2-GPT4

uv run python -m cs336_basics.bpe \
  --input data/owt.txt \
  --vocab-size 32000 \
  --special "<|endoftext|>" \
  --out-dir ./artifacts \
  --prefix owt



# # load tokenizer (for training)
# from cs336_basics.bpe import BPETokenizer
# tok = BPETokenizer.from_files(
#     vocab_filepath="artifacts/owt_vocab.pkl",
#     merges_filepath="artifacts/owt_merges.pkl",
#     special_tokens=["<|endoftext|>"],
# )

# # generate tokenized data (.npy)
# import numpy as np
# from cs336_basics.bpe import BPETokenizer
# tok = BPETokenizer.from_files("artifacts/owt_vocab.pkl","artifacts/owt_merges.pkl",["<|endoftext|>"])
# with open("owt_train.txt","r",encoding="utf-8",errors="ignore") as f, open("owt_train_tokens.bin","wb") as out:
#     for tid in tok.encode_iterable(f):
#         out.write(np.uint16(tid).tobytes())

# # During training
# tokens = np.memmap("owt_train_tokens.bin", dtype=np.uint16, mode="r")

# from cs336_basics.training_utils import get_batch
# x, y = get_batch(tokens, batch_size=32, context_length=256, device="cuda:0")
