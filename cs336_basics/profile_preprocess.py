import time
from pathlib import Path
from collections import Counter

from tokenizer import pre_tokenization

input_path = "../data/TinyStoriesV2-GPT4-train.txt"
special_tokens = ["<|endoftext|>"]


def process_bytes(text):
    word_counts = Counter()

    for word in pre_tokenization(text, special_tokens):
        if word in special_tokens:
            continue

        token = tuple(bytes([b]) for b in word.encode("utf-8"))
        word_counts[token] += 1

    return word_counts


def process_int(text):
    word_counts = Counter()

    for word in pre_tokenization(text, special_tokens):
        if word in special_tokens:
            continue

        token = tuple(word.encode("utf-8"))
        word_counts[token] += 1

    return word_counts


def main():
    size = Path(input_path).stat().st_size
    print(f"文件大小：{size / 1024**3:.2f} GiB", flush=True)

    # 一次性读取，保证两种方式使用相同文本
    start = time.perf_counter()

    with open(input_path, "rb") as f:
        text = f.read().decode("utf-8", errors="ignore")

    print(
        f"读取和解码耗时：{time.perf_counter() - start:.2f} 秒",
        flush=True,
    )

    print("\n开始测试 bytes tuple...", flush=True)
    start = time.perf_counter()
    bytes_counts = process_bytes(text)
    bytes_time = time.perf_counter() - start

    print(f"bytes 耗时：{bytes_time:.2f} 秒", flush=True)
    print(f"bytes 不同词数：{len(bytes_counts):,}", flush=True)

    print("\n开始测试 int tuple...", flush=True)
    start = time.perf_counter()
    int_counts = process_int(text)
    int_time = time.perf_counter() - start

    print(f"int 耗时：{int_time:.2f} 秒", flush=True)
    print(f"int 不同词数：{len(int_counts):,}", flush=True)

    # 将整数 token 转回 bytes token，检查结果是否完全一致
    normalized_int_counts = {
        tuple(bytes([b]) for b in word): count
        for word, count in int_counts.items()
    }

    assert bytes_counts == normalized_int_counts, "词频不一致！"

    print("\n正确性检查通过：两种方式的词频完全一致")
    print(f"加速比：{bytes_time / int_time:.2f} 倍")


if __name__ == "__main__":
    main()