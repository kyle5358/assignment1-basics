import cProfile
import pstats
from tokenizer import train_bpe

input_path = "../data/TinyStoriesV2-GPT4-valid.txt"
vocab_size = 50257
special_tokens = ["<|endoftext|>"]

profiler = cProfile.Profile()
profiler.enable()

train_bpe(input_path, vocab_size, special_tokens)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats("cumulative").print_stats(30)