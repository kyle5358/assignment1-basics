# class BPEtokenizer:
#     '''
#     输入input_path(包含training data), vocab_size(int), special_tokens(list[str]).
#
#     返回vocab(dict[int, bytes], int为原ID, bytes为token bytes),
#
#     和merges(list[tuple[bytes, bytes]],记录所有被合并的bytes).
#     '''
#     def __init__(self, vocab=None, merges=None,special_tokens=None):
#         self.vocab = vocab if vocab is not None else {i:bytes(i) for i in ranges(256)}
#         self.special_tokens = special_tokens if special_tokens is not None else []
#         self.merges = merges if merges is not None else []
#
#     def train(self, input_path, vocab_size, special_tokens):

import regex as re
import os
from multiprocessing import Pool
import multiprocessing
from pretokenization_example import find_chunk_boundaries

# 预编译正则
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

from collections import Counter


#预分词 处理special token
def pre_tokenization(text,special_tokens):
    if not special_tokens:
        return PAT.findall(text)

    #special tokens 列表按长度降序
    sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
    #转义特殊token，构造匹配正则
    special_pats="("+"|".join(re.escape(s) for s in sorted_special_tokens)+")"
    #用特殊字符分割文本成块
    parts=re.split(special_pats,text)

    result=[]
    for part in parts:
        if not part:
            continue
        if part in special_tokens:
            result.append(part)
        else:
            result.extend(PAT.findall(part))
    return result

#预分词 并统计词频
def process_text(text,special_tokens=None):
    special_tokens = special_tokens or []
    word_counts = Counter()

    for word in pre_tokenization(text,special_tokens):
        #如果是特殊字符就跳过不统计
        if word in special_tokens:
            continue
        token=tuple(bytes([a]) for a in word.encode("utf8"))
        word_counts[token] += 1
    return dict(word_counts)

# print(process_text("hello hello world"))
# #{(b'h', b'e', b'l', b'l', b'o'): 1, (b' ', b'h', b'e', b'l', b'l', b'o'): 1, (b' ', b'w', b'o', b'r', b'l', b'd'): 1}
#整个文本切块，并行处理，汇总词频
# def counting_init(input_path,special_tokens=None):
#     with open(input_path, "rb") as f:
#         text=f.read().decode("utf-8", errors="ignore")
#     return process_text(text,special_tokens)
from collections import Counter
import multiprocessing
from pretokenization_example import find_chunk_boundaries

def counting_init(input_path,specical_tokens,num_chunks=4):
    with open(input_path,"rb") as f:
        boundaries = find_chunk_boundaries(
            f,
            num_chunks,
            split_special_token = b"<|endoftext|>"
        )
          # specical_tokens[0].encode("utf8")
    #为每个文本块创建任务
    tasks=[]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        tasks.append((input_path,start,end,specical_tokens))
    #并行处理文本块
    with multiprocessing.Pool() as pool:
        results=pool.starmap(process_chunk,tasks)

    #汇总词频
    final_word_counts=Counter()
    for result in results:
        final_word_counts.update(result)

    return dict(final_word_counts)
#处理一个文本块 每个进程
def process_chunk(input_path,start,end,special_tokens):
    with open(input_path,'rb') as f:
        f.seek(start)
        text = f.read(end - start).decode('utf-8',errors='ignore')
        return process_text(text,special_tokens)#返回词频

#word_count 统计文本中单词的频率 {tuple(letter),count}
#pairs_count
#volcabu
#merge
# input_path="../data/test.txt"
# vocab_size = 1257
# special_tokens = ["<|endoftext|>"]

def train_bpe(input_path, vocab_size, special_tokens):

    vocab= {i:bytes([i]) for i in range(256)}

    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    #初始化word_count
    word_counts= counting_init(input_path,special_tokens)

    pair_counts = count_pairs(word_counts)
    merges = []

    while len(vocab)<vocab_size:
        if not pair_counts:#有可能都合并完了，每个单词就一个token
            break

        max_pair = max(
            pair_counts,
            key=lambda pair: (pair_counts[pair], pair)#def get_key(pair):return (pair_counts[pair], pair)
        )

        word_counts = update_word_counts(word_counts, max_pair)

        vocab[len(vocab)]=max_pair[0] + max_pair[1]
        merges.append(max_pair)#max_pair==(max_pair[0], max_pair[1])   eg('a','b')

        pair_counts = count_pairs(word_counts)
    return vocab,merges


def count_pairs(word_counts):
    pair_counts={}
    for word, count in word_counts.items():
        for pair in zip(word[:-1], word[1:]):
            pair_counts[pair] = pair_counts.get(pair, 0) + count
    return pair_counts
#print(count_pairs({(b'h', b'e', b'l', b'l', b'o'):2}))

def update_word_counts(word_counts, max_pair):
    new_word_counts = {}
    for word, count in word_counts.items():
        new_word = merge_single_word(word, max_pair)
        new_word_counts[new_word] = new_word_counts.get(new_word, 0) + count
    return new_word_counts

#word tuple格式
def merge_single_word(word, max_pair):
    new_word = []
    i = 0
    while i+1 < len(word):
        if (word[i],word[i+1])==max_pair:
            new_word.append(word[i]+word[i+1])
            i=i+2
        else:
            new_word.append(word[i])
            i=i+1
    if i==len(word)-1:
        new_word.append(word[i])
    return tuple(new_word)

def counting_init_single(input_path, special_tokens):
    with open(input_path, "rb") as f:
        text = f.read().decode("utf-8", errors="ignore")

    return process_text(text, special_tokens)


# single = counting_init_single(input_path, special_tokens)
# parallel = counting_init(input_path, special_tokens)
#
# print(single == parallel)
if __name__ == "__main__":
    # input_path = "../data/TinyStoriesV2-GPT4-train.txt"
    # vocab_size = 50257
    input_path = "../data/test.txt"
    vocab_size = 1257

    special_tokens = ["<|endoftext|>"]
    print(train_bpe(input_path, vocab_size, special_tokens))

# def train_bpe(input_path, vocab_size, special_tokens,num_chunks=1000):
#     tokenizer=BPETokenizer()
#     tokenizer.train(
#         input_path=input_path,
#         vocab_size=vocab_size,
#         special_tokens=special_tokens,
#         num_chunks=1000,
#         new_id=None,
#     )
#     return tokenizer
#     vocab, merges = run_train_bpe()


