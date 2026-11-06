
import cProfile
import pstats
import time
import psutil
import os
import json
from tokenizer import train_bpe

input_path = "../data/TinyStoriesV2-GPT4-train.txt"
vocab_size = 50257
special_tokens = ["<|endoftext|>"]

process = psutil.Process(os.getpid())

profiler = cProfile.Profile()

start = time.perf_counter()
profiler.enable()

tokenizer = train_bpe(input_path, vocab_size, special_tokens, 1000)

profiler.disable()
end = time.perf_counter()

print(f"\n总训练时间: {end - start:.2f} 秒")
print(f"词表大小: {len(tokenizer.vocab)}")

mem_info = process.memory_info()
print(f"当前 RSS 内存: {mem_info.rss / 1024**3:.4f} GB")

print("\n===== Cumulative Top 30 =====")
stats = pstats.Stats(profiler)
stats.strip_dirs()
stats.sort_stats("cumulative").print_stats(30)

print("\n===== Tottime Top 30 =====")
stats.sort_stats("tottime").print_stats(30)

stats.dump_stats("bpe_training_profile.prof")

# =========================
# 保存 vocab 和 merges
# =========================

output_dir = "../output/run_bpe_train_on_tinystories_output"
os.makedirs(output_dir, exist_ok=True)

# 1. 保存 vocab
readable_vocab = {
    token_id: list(token_bytes)
    for token_id, token_bytes in tokenizer.vocab.items()
}

vocab_path = os.path.join(output_dir, "vocab.json")

with open(vocab_path, "w", encoding="utf-8") as f:
    json.dump(readable_vocab, f, indent=4)

# 2. 保存 merges
merges_path = os.path.join(output_dir, "merges.txt")

with open(merges_path, "w", encoding="utf-8") as f:
    for p1, p2 in tokenizer.merges:
        s1 = ",".join(map(str, p1))
        s2 = ",".join(map(str, p2))
        f.write(f"{s1} {s2}\n")

print("\n模型文件保存完成：")
print(f"Vocab:  {vocab_path}")
print(f"Merges: {merges_path}")
