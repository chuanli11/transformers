import sys
sys.path.insert(1, '/home/ubuntu/projects/transformers//src')

from transformers import PyTorchBenchmark, PyTorchBenchmarkArguments, BertConfig

# list_models = [
#         "bert-base-uncased",
#         "bert-large-uncased",
#         "roberta-base",
#         "roberta-large",
#         "distilbert-base-uncased",
#         "gpt2",
#         "gpt2-medium",
#         "gpt2-large"]

list_models = ["bert-base-uncased"]

args = PyTorchBenchmarkArguments(
        memory=True, training=True, inference=True, models=list_models, batch_sizes=[32, 64], sequence_lengths=[64], save_to_csv=True)

benchmark = PyTorchBenchmark(args)
results = benchmark.run()

