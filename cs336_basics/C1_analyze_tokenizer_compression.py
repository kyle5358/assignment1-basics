
import json
import random
from tokenizer import Tokenizer


# =========================
# 1. 路径配置
# =========================

INPUT_PATH = "../data/TinyStoriesV2-GPT4-train.txt"

VOCAB_PATH = "../output/run_bpe_train_on_tinystories_output/vocab.json"
MERGES_PATH = "../output/run_bpe_train_on_tinystories_output/merges.txt"

OUTPUT_PATH = "../output/run_bpe_train_on_tinystories_output/tinystories_sample_ids.json"

SPECIAL_TOKENS = ["<|endoftext|>"]
NUM_SAMPLES = 10


# =========================
# 2. 按 endoftext 切分文档
# =========================

def iter_documents(path, chunk_size=4 * 1024 * 1024):
    delimiter = b"<|endoftext|>"
    buffer = b""

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            buffer += chunk

            while True:
                pos = buffer.find(delimiter)

                if pos == -1:
                    break

                doc = buffer[:pos].strip()
                buffer = buffer[pos + len(delimiter):]

                if doc:
                    yield doc

    # 处理文件末尾的文档
    doc = buffer.strip()
    if doc:
        yield doc


# =========================
# 3. 随机抽取 10 份文档
# =========================

def sample_documents(path, n=10, seed=42):
    rng = random.Random(seed)

    samples = []
    total_docs = 0

    # Reservoir Sampling：不需要将整个数据集读入内存
    for doc in iter_documents(path):
        total_docs += 1

        if len(samples) < n:
            samples.append(doc)
        else:
            j = rng.randrange(total_docs)
            if j < n:
                samples[j] = doc

    if len(samples) < n:
        raise ValueError(f"文档数量不足 {n} 份，实际只有 {total_docs} 份")

    print(f"数据集总文档数: {total_docs}")
    print(f"随机抽取文档数: {len(samples)}")

    return samples


# =========================
# 4. 编码并计算压缩比
# =========================

def encode_and_report(samples, tokenizer):

    records = []

    total_bytes = 0
    total_tokens = 0

    print("\n========== TinyStories 编码结果 ==========")

    for i, doc_bytes in enumerate(samples, 1):

        text = doc_bytes.decode("utf-8")

        # 编码为整数 Token ID
        ids = tokenizer.encode(text)

        byte_count = len(doc_bytes)
        token_count = len(ids)

        ratio = byte_count / token_count if token_count else 0

        total_bytes += byte_count
        total_tokens += token_count

        print(f"\n文档 {i}")
        print(f"原始字节数: {byte_count}")
        print(f"Token 数量: {token_count}")
        print(f"压缩比: {ratio:.4f} bytes/token")
        print(f"Token ID 前 50 个: {ids[:50]}")

        records.append({
            "document": i,
            "byte_count": byte_count,
            "token_count": token_count,
            "compression_ratio": ratio,
            "token_ids": ids
        })

    # 所有文档合并计算总体压缩比
    overall_ratio = (
        total_bytes / total_tokens if total_tokens else 0
    )

    print("\n========== 汇总结果 ==========")
    print(f"总字节数: {total_bytes}")
    print(f"总 Token 数: {total_tokens}")
    print(f"总体压缩比: {overall_ratio:.4f} bytes/token")

    # 保存完整 Token ID
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    print(f"\n完整 Token ID 已保存至: {OUTPUT_PATH}")


# =========================
# 5. 主函数
# =========================

def main():

    # 加载 TinyStories 10K Tokenizer
    tokenizer = Tokenizer.from_files(
        VOCAB_PATH,
        MERGES_PATH,
        SPECIAL_TOKENS
    )

    print(f"Tokenizer 词表大小: {len(tokenizer.vocab)}")

    # 抽样
    samples = sample_documents(INPUT_PATH, NUM_SAMPLES)

    # 编码并统计
    encode_and_report(samples, tokenizer)


if __name__ == "__main__":
    main()
