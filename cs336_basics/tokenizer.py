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
class BPETokenizer:
    def __init__(self, vocab, merges):
        self.vocab = vocab
        self.merges = merges

import regex as re
# 预编译正则
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

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
    special_set = set(special_tokens)#更快
    for part in parts:
        if not part:
            continue
        if part in special_set:
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
        # token=tuple(bytes([a]) for a in word.encode("utf8"))
        # token = tuple(word.encode("utf8"))
        # word_counts[token] += 1
        token_ids = tuple(word.encode("utf8"))
        word_counts[token_ids] += 1
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
from cs336_basics.pretokenization_example import find_chunk_boundaries
import time
def counting_init(input_path,special_tokens,num_chunks=1000):
    # start_time = time.time()
    with open(input_path,"rb") as f:
        boundaries = find_chunk_boundaries(
            f,
            num_chunks,
            split_special_token = b"<|endoftext|>"
        )
          # specical_tokens[0].encode("utf8")

    # print(f"切块完成，耗时 {time.time() - start_time:.2f} 秒", flush=True)
    # print(f"实际分块数：{len(boundaries) - 1}", flush=True)
    # print(f"各块大小（MB）：", flush=True)
    # print([
    #     round((end - start) / 1024 / 1024, 2)
    #     for start, end in zip(boundaries[:-1], boundaries[1:])
    # ], flush=True)

    #为每个文本块创建任务
    tasks=[]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        tasks.append((input_path,start,end,special_tokens))
    #增加空任务检查 避免任务只有几个时却启动大量空闲进程
    if not tasks:
        return {}

    process_count = min(multiprocessing.cpu_count(), len(tasks))

    #并行处理文本块
    #with multiprocessing.Pool() as pool:
    with multiprocessing.Pool(processes=process_count ) as pool:
        results=pool.starmap(process_chunk,tasks)

    #汇总词频
    final_word_counts=Counter()
    for result in results:
        final_word_counts.update(result)

    return dict(final_word_counts)

import os
import time
#处理一个文本块 每个进程
def process_chunk(input_path,start,end,special_tokens):
    # with open(input_path,'rb') as f:
    #     f.seek(start)
    #     text = f.read(end - start).decode('utf-8',errors='ignore')
    #     return process_text(text,special_tokens)#返回词频
    # t0 = time.time()
    # chunk_mb = (end - start) / 1024 / 1024
    #
    # print(
    #     f"[PID {os.getpid()}] 开始处理 {chunk_mb:.2f} MB",
    #     flush=True
    # )

    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")

    #t1 = time.time()
    result = process_text(text, special_tokens)
    #t2 = time.time()

    # print(
    #     f"[PID {os.getpid()}] 完成："
    #     f"读取解码 {t1 - t0:.2f}s，"
    #     f"预分词统计 {t2 - t1:.2f}s，"
    #     f"词种数 {len(result)}",
    #     flush=True
    # )

    return result
#word_count 统计文本中单词的频率 {tuple(letter),count}
#pairs_count
#volcabu
#merge
# input_path="../data/test.txt"
# vocab_size = 1257
# special_tokens = ["<|endoftext|>"]

def train_bpe(input_path, vocab_size, special_tokens,num_chunks=1000):
    changed_pairs = set()
    vocab= {i:bytes([i]) for i in range(256)}

    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    #初始化word_count
    word_counts= counting_init(input_path,special_tokens,num_chunks)

    pair_counts = count_pairs(word_counts)
    pair_to_words = build_pair_to_words(word_counts)
    merges = []
    merge_id_pairs = []

    # 新增：当前最高频率，以及达到最高频率的候选 pair
    #候选集合初始化
    current_max_freq = max(pair_counts.values(), default=0)
    candidates = { #满足条件的pair集合
        pair
        for pair, count in pair_counts.items()
        if count == current_max_freq
    }

    while len(vocab)<vocab_size:
        if not pair_counts:#有可能都合并完了，每个单词就一个token
            print("训练提前结束：pair_counts 已经为空")
            break
        # 把上一轮变化的 pair 加入候选集合
        candidates.update(changed_pairs)
        changed_pairs.clear()

        max_pair, candidates, current_max_freq = select_best_pair(
            pair_counts,
            candidates,
            current_max_freq,
            vocab
        )
        if max_pair is None:
            print("训练提前结束，没有可合并的pair")
            break

        # #word_counts = update_word_counts(word_counts, max_pair)
        # update_pair_counts(word_counts,pair_counts,pair_to_words,max_pair,changed_pairs)
        # # vocab[len(vocab)]=max_pair[0] + max_pair[1]
        # vocab[len(vocab)] = byte(max_pair)
        # merges.append(max_pair)#max_pair==(max_pair[0], max_pair[1])   eg('a','b')
        new_token_id=len(vocab)
        left_id,right_id = max_pair
        vocab[new_token_id]=vocab[left_id]+vocab[right_id]
        update_pair_counts(word_counts, pair_counts, pair_to_words, max_pair,new_token_id, changed_pairs)
        merge_id_pairs.append(max_pair)

    merges = [
        (vocab[left_id], vocab[right_id])
        for left_id, right_id in merge_id_pairs
    ]
    print(f"最终词表大小: {len(vocab)}")
    print(f"实际合并次数: {len(merges)}")
        #pair_counts = count_pairs(word_counts)
    return BPETokenizer(vocab, merges)

def select_best_pair(pair_counts,candidates,current_max_freq,vocab):
    #更新pair_counts后，删掉已经不存在的pair
    candidates={
        pair
        for pair in candidates
        if pair in pair_counts
    }
    # 候选集合为空，或者候选中的最高频率低于当前记录
    local_max=max((pair_counts[pair] for pair in candidates),default=0)
    if local_max<current_max_freq:
        current_max_freq = max(pair_counts.values(), default=0)
        candidates = {  # 满足条件的pair集合
            pair
            for pair, count in pair_counts.items()
            if count == current_max_freq
        }
    else:
        # 只保留候选集合中达到当前最高频率的 pair
        current_max_freq = local_max

        candidates = {
            pair for pair in candidates
            if pair_counts[pair] == current_max_freq
        }
    if not candidates:
        return None,candidates,current_max_freq
    best_pair = max(
        candidates,
        key=lambda pair: (
            pair_counts[pair],
            vocab[pair[0]],
            vocab[pair[1]],
        ),
    )
    return best_pair,candidates,current_max_freq

def count_pairs(word_counts):
    pair_counts={}
    for word, count in word_counts.items():
        for pair in zip(word[:-1], word[1:]):
            pair_counts[pair] = pair_counts.get(pair, 0) + count
    return pair_counts
#print(count_pairs({(b'h', b'e', b'l', b'l', b'o'):2}))

#建立反向索引
def build_pair_to_words(word_counts):
    pair_to_words={}
    for word in word_counts:
        for pair in set(zip(word[:-1], word[1:])):#这里可以去重
            if pair not in pair_to_words:
                pair_to_words[pair] = set()
            pair_to_words[pair].add(word)
    return pair_to_words


def update_pair_to_words(pair_to_words,old_word,new_word):
    #删掉旧单词索引
    for pair in set(zip(old_word[:-1], old_word[1:])):
        if pair in pair_to_words:
            pair_to_words[pair].remove(old_word)

            if not pair_to_words[pair]:#键盘对应的集合为空
                del pair_to_words[pair]

        # words=pair_to_words.get(pair) #words={(d,o,g),(c,a,t)}
        # if words is not None: #判断有没有找到这个 pair 对应的集合。
        #     words.discard(old_word)
        #
        #     if not words:# 判断集合里面还有没有单词(word之前只有一个单词，现在被删为空了)
        #         del pair_to_words[pair]
        #
        # if words is not None:
        #     words.discard(old_word)
        #
        #     if word is None:
        #         del pair_to_words[pair]

    for pair in set(zip(new_word[:-1], new_word[1:])):
        if pair not in pair_to_words:
            pair_to_words[pair] = set()
        pair_to_words[pair].add(new_word)

def update_pair_counts(word_counts,pair_counts,pair_to_words,max_pair,new_token_id,changed_pairs=None):
    affected_pairs=list(pair_to_words.get(max_pair,set()))
    for old_word in affected_pairs:
        counts=word_counts[old_word]
        #扣除旧单词对所有相邻pair的贡献
        for pair in zip(old_word[:-1], old_word[1:]):
            pair_counts[pair]-=counts
            if changed_pairs is not None:
                changed_pairs.add(pair)

            if pair_counts[pair]==0:
                del pair_counts[pair]
        #2 合并旧单词
        new_word=merge_single_word(old_word,max_pair,new_token_id)
        #3 更新反向索引
        update_pair_to_words(pair_to_words,old_word,new_word)
        #4 更新单词词频
        del word_counts[old_word]
        word_counts[new_word]=word_counts.get(new_word,0)+counts

        #5 加上新单词对所有相邻pair的贡献
        for pair in zip(new_word[:-1], new_word[1:]):
            pair_counts[pair]=pair_counts.get(pair,0)+counts

            if changed_pairs is not None:
                changed_pairs.add(pair)

def update_word_counts(word_counts, max_pair):
    new_word_counts = {}
    for word, count in word_counts.items():
        new_word = merge_single_word(word, max_pair)
        new_word_counts[new_word] = new_word_counts.get(new_word, 0) + count
    return new_word_counts

#word tuple格式
def merge_single_word(word, max_pair,new_token_id):
    new_word = []
    i = 0
    while i+1 < len(word):
        if (word[i],word[i+1])==max_pair:
            new_word.append(new_token_id)
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

############encoding and decoding
def merge_tokens(tokens,pair):
    new_tokens=[]
    i=0

    while i<len(tokens):
        if(i+1<len(tokens) and (tokens[i],tokens[i+1])==pair):
            new_tokens.append(tokens[i]+tokens[i+1])
            i+=2
        else:
            new_tokens.append(tokens[i])
            i+=1

    return new_tokens

import json

class Tokenizer:
    def __init__(self, vocab, merges,special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.token_to_id = {
            token_bytes: token_id
            for token_id, token_bytes in self.vocab.items()
        }
        self.special_tokens=special_tokens or []

        next_id=max(self.vocab,default=-1)+1

        for token in self.special_tokens:
            token_bytes=token.encode('utf8')

            if token_bytes not in self.token_to_id:
                self.vocab[next_id]=token_bytes
                self.token_to_id[token_bytes]=next_id
                next_id += 1
        self.bpe_ranks={bpe:id for id,bpe in enumerate(self.merges)}

    def _encode_piece(self,text):
        text_bytes=text.encode('utf8')
        #如果整个单词都在vocab
        if text_bytes in self.token_to_id:
            return [self.token_to_id[text_bytes]]

        tokens=[bytes([i]) for i in text_bytes]

        while len(tokens)>1:
            pairs=[(tokens[i],tokens[i+1]) for i in range(len(tokens)-1)]
            best_pair=min(pairs,key=lambda p:self.bpe_ranks.get(p,float('inf')))

            if best_pair not in self.bpe_ranks:
                break

            tokens=merge_tokens(tokens,best_pair)
        return[self.token_to_id[token] for token in tokens]

    def encode(self,text):
        result=[]
        special_set=set(self.special_tokens)

        pieces=pre_tokenization(text,self.special_tokens)
        for piece in pieces:
            if piece in special_set:
                token_bytes = piece.encode("utf-8")
                result.append(self.token_to_id[token_bytes])
            else:
                result.extend(self._encode_piece(piece))
        return result

    def decode(self,ids):
        text_bytes=b"".join(self.vocab[token_id] for token_id in ids)
        return text_bytes.decode("utf-8",errors="replace")

    def encode_iterable(self,iterable):
        for text in iterable:
            yield from self.encode(text)

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):

        # 1. 读取 vocab
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)

            # JSON value 是整数列表，需要恢复成 bytes
            vocab = {
                int(k): bytes(v)
                for k, v in raw_vocab.items()
            }

        # 2. 读取 merges
        merges = []

        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                if len(parts) == 2:
                    p0 = bytes(map(int, parts[0].split(",")))
                    p1 = bytes(map(int, parts[1].split(",")))

                    merges.append((p0, p1))


        # 3. 创建 Tokenizer 对象
        return cls(vocab, merges, special_tokens) #等价于return Tokenizer(vocab, merges, special_tokens)，但是可以用与子类的共享函数灵活
    
if __name__ == "__main__":
    vocab = {
        0: b"h",
        1: b"e",
        2: b"l",
        3: b"o",
        4: b"he",
        5: b"ll",
        6: b"llo",
    }

    merges = [
        (b"h", b"e"),
        (b"l", b"l"),
        (b"ll", b"o"),
    ]

    tokenizer = Tokenizer(vocab, merges)

    text = "hello"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    print("编码结果:", ids)
    print("解码结果:", decoded)

    assert decoded == text
    # input_path = "../data/TinyStoriesV2-GPT4-train.txt"
    # #vocab_size = 50257
    # # input_path = "../data/test.txt"
    # vocab_size = 257
    #
    # special_tokens = ["<|endoftext|>"]
    # print(train_bpe(input_path, vocab_size, special_tokens))

    # word_counts = counting_init_single(input_path, special_tokens)
    # pair_counts = count_pairs(word_counts)
    # pair_to_words = build_pair_to_words(word_counts)
    #
    # max_pair = max(
    #     pair_counts,
    #     key=lambda pair: (pair_counts[pair], pair)
    # )
    #
    # update_pair_counts(
    #     word_counts, pair_counts, pair_to_words, max_pair
    # )
    #
    # # 验证增量 pair 统计是否正确
    # assert pair_counts == count_pairs(word_counts)
    #
    # # 验证反向索引是否正确
    # assert pair_to_words == build_pair_to_words(word_counts)
    #
    # print("单轮增量更新验证通过！")

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

