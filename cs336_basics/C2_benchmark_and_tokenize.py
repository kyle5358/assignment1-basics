
import os
import time
import numpy as np
from tokenizer import Tokenizer


# =========================
# 1. 路径配置
# =========================

TRAIN_PATH = "../data/TinyStoriesV2-GPT4-train.txt"
VALID_PATH = "../data/TinyStoriesV2-GPT4-valid.txt"

VOCAB_PATH = "../output/run_bpe_train_on_tinystories_output/vocab.json"
MERGES_PATH = "../output/run_bpe_train_on_tinystories_output/merges.txt"

OUTPUT_DIR = "../output/tokenized"

SPECIAL_TOKENS = ["<|endoftext|>"]

# 吞吐量测试：处理约 100 MB
BENCHMARK_BYTES = 100 * 1024 * 1024


# =========================
# 2. 按 endoftext 切分文档
# =========================

def iter_document_segments(path, chunk_size=4 * 1024 * 1024):

    delimiter = b"<|endoftext|>"
    buffer = bytearray()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            buffer.extend(chunk)

            while True:
                pos = buffer.find(delimiter)

                if pos == -1:
                    break

                end = pos + len(delimiter)

                # 保留文档结束标记
                yield bytes(buffer[:end])

                del buffer[:end]

    # 处理最后没有 endoftext 的内容
    if buffer:
        yield bytes(buffer)


# =========================
# 3. (c) 测量吞吐量
# =========================

def benchmark_throughput(path, tokenizer):

    processed_bytes = 0
    total_tokens = 0
    encode_seconds = 0.0

    for segment in iter_document_segments(path):

        text = segment.decode("utf-8")

        start = time.perf_counter()
        ids = tokenizer.encode(text)
        encode_seconds += time.perf_counter() - start

        processed_bytes += len(segment)
        total_tokens += len(ids)

        # 达到测试规模后停止
        if processed_bytes >= BENCHMARK_BYTES:
            break

    throughput = processed_bytes / encode_seconds

    # Pile 按 825 GB = 825 * 10^9 bytes 估算
    pile_seconds = 825 * 10**9 / throughput

    print("\n========== (c) 吞吐量测试 ==========")
    print(f"处理字节数: {processed_bytes:,}")
    print(f"生成 Token 数: {total_tokens:,}")
    print(f"编码耗时: {encode_seconds:.2f} 秒")
    print(f"吞吐量: {throughput / 1024**2:.2f} MB/s")
    print(f"估算处理 Pile 耗时: {pile_seconds / 3600:.2f} 小时")
    print(f"估算处理 Pile 耗时: {pile_seconds / 86400:.2f} 天")


# =========================
# 4. 统计完整数据集的 Token 数
# =========================

def count_tokens(path, tokenizer):

    total_tokens = 0

    for segment in iter_document_segments(path):

        text = segment.decode("utf-8")
        ids = tokenizer.encode(text)

        total_tokens += len(ids)

    return total_tokens


# =========================
# 5. (d) 编码并保存为 uint16
# =========================

def encode_to_npy(path, output_path, tokenizer):

    print(f"\n开始处理: {path}")

    # 第一次遍历：统计 Token 总数
    print("第一遍：统计 Token 数量...")
    start = time.perf_counter()

    total_tokens = count_tokens(path, tokenizer)

    print(f"Token 总数: {total_tokens:,}")
    print(f"统计耗时: {time.perf_counter() - start:.2f} 秒")

    # 创建 NumPy 磁盘映射数组
    # 不需要一次性把整个数据集放入内存
    print("第二遍：编码并写入 NumPy 数组...")

    arr = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.uint16,
        shape=(total_tokens,)
    )

    offset = 0
    start = time.perf_counter()

    for segment in iter_document_segments(path):

        text = segment.decode("utf-8")
        ids = tokenizer.encode(text)

        end = offset + len(ids)

        arr[offset:end] = ids

        offset = end

    arr.flush()
    del arr

    elapsed = time.perf_counter() - start

    if offset != total_tokens:
        raise RuntimeError(
            f"Token 数量不一致: 预统计 {total_tokens}, 实际写入 {offset}"
        )

    print(f"写入完成: {output_path}")
    print(f"实际 Token 数: {offset:,}")
    print(f"编码写入耗时: {elapsed:.2f} 秒")


# =========================
# 6. 主函数
# =========================

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载训练好的 Tokenizer
    tokenizer = Tokenizer.from_files(
        VOCAB_PATH,
        MERGES_PATH,
        SPECIAL_TOKENS
    )

    print("词表大小:", len(tokenizer.vocab))

    # 检查 uint16 是否足够
    max_id = max(tokenizer.vocab)

    if max_id > np.iinfo(np.uint16).max:
        raise ValueError(
            f"最大 Token ID 为 {max_id}，无法使用 uint16"
        )

    print("最大 Token ID:", max_id)
    print("可以使用 uint16")

    # (c) 测量吞吐量
    benchmark_throughput(TRAIN_PATH, tokenizer)

    # (d) 编码训练集
    encode_to_npy(
        TRAIN_PATH,
        os.path.join(OUTPUT_DIR, "tinystories_train.npy"),
        tokenizer
    )

    # (d) 编码验证集
    encode_to_npy(
        VALID_PATH,
        os.path.join(OUTPUT_DIR, "tinystories_valid.npy"),
        tokenizer
    )


if __name__ == "__main__":
    main()
