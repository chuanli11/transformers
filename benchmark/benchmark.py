import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, dir_path + '/../src')

from transformers import PyTorchBenchmark, PyTorchBenchmarkArguments, BertConfig

list_models = [
        "bert-base-uncased",
        "bert-large-uncased",
        "roberta-base",
        "roberta-large",
        "distilbert-base-uncased",
        "gpt2",
        "gpt2-medium",
        "gpt2-large"]


args = PyTorchBenchmarkArguments(
        memory=False, training=True, inference=True, models=list_models, batch_sizes=[32, 64, 128], sequence_lengths=[64], save_to_csv=True)

benchmark = PyTorchBenchmark(args)
results = benchmark.run()

