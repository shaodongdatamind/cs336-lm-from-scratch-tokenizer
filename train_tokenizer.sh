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


