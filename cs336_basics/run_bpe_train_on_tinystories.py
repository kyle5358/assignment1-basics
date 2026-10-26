# import cProfile
# import pstats
# from tokenizer import train_bpe
#
# input_path = "../data/TinyStoriesV2-GPT4-valid.txt"
# vocab_size = 50257
# #vocab_size = 300
#
# special_tokens = ["<|endoftext|>"]
#
# profiler = cProfile.Profile()
# profiler.enable()
#
# train_bpe(input_path, vocab_size, special_tokens,20)
#
# profiler.disable()
# stats = pstats.Stats(profiler)
# stats.sort_stats("cumulative").print_stats(30)


import cProfile
import pstats
import time
from tokenizer import train_bpe

input_path = "../data/TinyStoriesV2-GPT4-train.txt"
vocab_size = 50257
# input_path = "../data/TinyStoriesV2-GPT4-train-100M-test.txt"
# vocab_size = 257
special_tokens = ["<|endoftext|>"]

profiler = cProfile.Profile()

start = time.perf_counter()
profiler.enable()

train_bpe(input_path, vocab_size, special_tokens, 1000)

profiler.disable()
end = time.perf_counter()

print(f"\n总训练时间: {end - start:.2f} 秒")

stats = pstats.Stats(profiler)
stats.strip_dirs()
stats.sort_stats("cumulative").print_stats(30)
