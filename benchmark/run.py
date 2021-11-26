import sys
import os
from multiprocessing import Queue

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, dir_path + '/../src')

from typing import Callable
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


from transformers.models.auto.configuration_auto import AutoConfig

inference = True
train = True
memory = False
speed = True
model_names = ["bert-base-uncased"]
batch_sizes = [32, 64, 128]
sequence_lengths = [64]
precision = "fp32"
repeat = 2
number = 10
number_warmup = 5
num_gpu = 2
optimize = True
backend = 'nccl'

config_dict = {
    model_name: AutoConfig.from_pretrained(model_name) for model_name in model_names
}

def setup(rank: int, world_size: int, backend: str):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run_ddp(ddp_func: Callable[[], None], world_size: int, number: int, model_name: str, batch_size: int, sequence_length: int, optimize: bool, backend: str):
    tqueue = mp.get_context('spawn').SimpleQueue()
    mp.spawn(ddp_func,
             args=(world_size, number, model_name, batch_size, sequence_length, optimize, backend, tqueue),
             nprocs=world_size,
             join=True)
    return tqueue.get()


def inference_func(number: int, model_name: str, batch_size: int, sequence_length: int):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    config = config_dict[model_name]

    has_model_class_in_config = (
        hasattr(config, "architectures")
        and isinstance(config.architectures, list)
        and len(config.architectures) > 0
    )

    if has_model_class_in_config:
        try:
            model_class = config.architectures[0]
            transformers_module = __import__("transformers", fromlist=[model_class])
            model_cls = getattr(transformers_module, model_class)
            model = model_cls(config)
        except ImportError:
            raise ImportError(
                f"{model_class} does not exist. If you just want to test the pretrained model, you might want to set `--only_pretrain_model` or `args.only_pretrain_model=True`."
            )

        model.eval()
        model.to(device)        

        # encoder-decoder has vocab size saved differently
        vocab_size = config.vocab_size if hasattr(config, "vocab_size") else config.encoder.vocab_size
        input_ids = torch.randint(vocab_size, (batch_size, sequence_length), dtype=torch.long, device=device)

        if precision == "fp16":
            if not device.type == 'cuda':
                raise ValueError("Mixed precision is possible only for GPU.")
            # amp seems to have memory leaks so that memory usage
            # is measured using .half() for now https://github.com/NVIDIA/apex/issues/439
            model.half()

        inference_model = model

        def encoder_decoder_forward():
            
            with torch.no_grad():
                for i_batch in range(number_warmup):
                    outputs = inference_model(input_ids, decoder_input_ids=input_ids)
                t0 = time.time()
                for i_batch in range(number):
                    outputs = inference_model(input_ids, decoder_input_ids=input_ids)
                torch.cuda.current_stream().synchronize()
                t1 = time.time()
            return t1 - t0

        def encoder_forward():
            with torch.no_grad():
                for i_batch in range(number_warmup):
                    outputs = inference_model(input_ids)
                t0 = time.time()
                for i_batch in range(number):
                    outputs = inference_model(input_ids)
                torch.cuda.current_stream().synchronize()
                t1 = time.time()
            return t1 - t0

        func = encoder_decoder_forward if config.is_encoder_decoder else encoder_forward
        
        return func()


def train_func(rank: int, num_gpu: int, number: int, model_name: str, batch_size: int, sequence_length: int, optimize: bool, backend: str, tqueue: Queue):

    setup(rank, num_gpu, backend)

    config = config_dict[model_name]

    has_model_class_in_config = (
        hasattr(config, "architectures")
        and isinstance(config.architectures, list)
        and len(config.architectures) > 0
    )

    if has_model_class_in_config:
        try:
            model_class = config.architectures[0]
            transformers_module = __import__("transformers", fromlist=[model_class])
            model_cls = getattr(transformers_module, model_class)
            model = model_cls(config)
        except ImportError:
            raise ImportError(
                f"{model_class} does not exist. If you just want to test the pretrained model, you might want to set `--only_pretrain_model` or `args.only_pretrain_model=True`."
            )

        model.to(rank)
        model = DDP(model, device_ids=[rank])

        if optimize:
            optimizer = optim.SGD(model.parameters(), lr=0.001)

        # encoder-decoder has vocab size saved differently
        vocab_size = config.vocab_size if hasattr(config, "vocab_size") else config.encoder.vocab_size
        input_ids = torch.randint(vocab_size, (batch_size, sequence_length), dtype=torch.long).to(rank)

        if precision == "fp16":
            if torch.cuda.device_count() < rank + 1:
                raise ValueError("Mixed precision is possible only for GPU.")
            # amp seems to have memory leaks so that memory usage
            # is measured using .half() for now https://github.com/NVIDIA/apex/issues/439
            model.half()

        def compute_loss_and_backprob_encoder():

            for i_batch in range(number_warmup):
                loss = model(input_ids, labels=input_ids)[0]
                loss.backward()
                if optimize:
                    optimizer.step()

            if rank == 0:
                t0 = time.time()

            for i_batch in range(number):
                loss = model(input_ids, labels=input_ids)[0]
                loss.backward()
                if optimize:
                    optimizer.step()
            
            if rank == 0:
                t1 = time.time()
                tqueue.put(t1 - t0)

        def compute_loss_and_backprob_encoder_decoder():

            for i_batch in range(number_warmup):
                loss = model(input_ids, labels=input_ids)[0]
                loss.backward()
                if optimize:
                    optimizer.step()

            if rank == 0:
                t0 = time.time()

            for i_batch in range(number):
                loss = model(input_ids, decoder_input_ids=input_ids, labels=input_ids)[0]
                loss.backward()
                if optimize:
                    optimizer.step()

            if rank == 0:
                t1 = time.time()
                tqueue.put(t1 - t0)

        func = (
            compute_loss_and_backprob_encoder_decoder
            if config.is_encoder_decoder
            else compute_loss_and_backprob_encoder
        )
        
        func()

    cleanup()

if __name__ == "__main__":
    for c, model_name in enumerate(model_names):
        print(f"{c + 1} / {len(model_names)}")

        model_dict = {
            "bs": batch_sizes,
            "ss": sequence_lengths,
            "result": {i: {} for i in batch_sizes},
        }

        for batch_size in batch_sizes:
            for sequence_length in sequence_lengths:
                
                if inference:
                    for i_run in range(repeat):
                        try:
                            t = inference_func(
                                number,
                                model_name,
                                batch_size,
                                sequence_length
                            )
                        except:
                            t = 0
                            print(f"BS: {batch_size}, Sequence Length: {sequence_length}, {model_name} didn't work for inference in {precision}. Maybe OOM")
                        print(t)

                if train:
                    for i_run in range(repeat):
                        try:
                            t = run_ddp(
                                train_func, 
                                num_gpu,
                                number, 
                                model_name, 
                                batch_size, 
                                sequence_length, 
                                optimize,
                                backend
                            )
                        except:
                            t = 0
                            print(f"BS: {batch_size}, Sequence Length: {sequence_length}, {model_name} didn't work for {num_gpu} DDP training in {precision}. Maybe OOM")
                        print(t)

