import sys
sys.path.insert(1, '/home/ubuntu/projects/transformers/src')
import timeit
import time
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


from transformers.models.auto.configuration_auto import AutoConfig

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
inference = False
train = True
memory = False
speed = True
model_names = ["bert-base-uncased"]
batch_sizes = [32]
sequence_lengths = [64]
fp16 = False
repeat = 5
number = 100
num_gpu = 2
optimize = False

config_dict = {
    model_name: AutoConfig.from_pretrained(model_name) for model_name in model_names
}

def prepare_inference_func(model_name: str, batch_size: int, sequence_length: int):
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

        if fp16:
            if not device.type == 'cuda':
                raise ValueError("Mixed precision is possible only for GPU.")
            # amp seems to have memory leaks so that memory usage
            # is measured using .half() for now https://github.com/NVIDIA/apex/issues/439
            model.half()

        inference_model = model

        def encoder_decoder_forward():
            with torch.no_grad():
                outputs = inference_model(input_ids, decoder_input_ids=input_ids)
            return outputs

        def encoder_forward():
            with torch.no_grad():
                outputs = inference_model(input_ids)
            return outputs

        func = encoder_decoder_forward if config.is_encoder_decoder else encoder_forward
        
        return func


def prepare_train_func(rank: int, num_gpu: int, number: int, model_name: str, batch_size: int, sequence_length: int, optimize: bool):

    dist.init_process_group(backend="nccl", init_method='env://', rank=rank, world_size=num_gpu)

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
        model.train()
        model = DDP(model, device_ids=[rank])

        if optimize:
            optimizer = optim.SGD(model.parameters(), lr=0.001)

        # encoder-decoder has vocab size saved differently
        vocab_size = config.vocab_size if hasattr(config, "vocab_size") else config.encoder.vocab_size
        input_ids = torch.randint(vocab_size, (batch_size, sequence_length), dtype=torch.long, device=device)

        if fp16:
            if not device.type == 'cuda':
                raise ValueError("Mixed precision is possible only for GPU.")
            # amp seems to have memory leaks so that memory usage
            # is measured using .half() for now https://github.com/NVIDIA/apex/issues/439
            model.half()

        def compute_loss_and_backprob_encoder():
            if rank == 0:
                t0 = time.time()

            for i_batch in range(number):
                loss = model(input_ids, labels=input_ids)[0]
                loss.backward()
                if optimize:
                    optimizer.step()
            
            if rank == 0:
                torch.cuda.current_stream().synchronize()
                t1 = time.time()
                print(t1 - t0)

        def compute_loss_and_backprob_encoder_decoder():
            if rank == 0:
                t0 = time.time()

            for i_batch in range(number):
                loss = model(input_ids, decoder_input_ids=input_ids, labels=input_ids)[0]
                loss.backward()
                if optimize:
                    optimizer.step()

            if rank == 0:
                torch.cuda.current_stream().synchronize()
                t1 = time.time()
                print(t1 - t0)

        func = (
            compute_loss_and_backprob_encoder_decoder
            if config.is_encoder_decoder
            else compute_loss_and_backprob_encoder
        )
        
        func()

def main():
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
                    if speed:
                        func = prepare_inference_func(model_name, batch_size, sequence_length)
                        for i_run in range(repeat):
                            t0 = time.time()
                            for i_batch in range(number):
                                func()
                            torch.cuda.current_stream().synchronize()
                            t1 = time.time()
                            print(t1 - t0)

                if train:
                    if speed:
                        mp.spawn(prepare_train_func,
                            args=(num_gpu, number, model_name, batch_size, sequence_length, optimize),
                            nprocs=num_gpu,
                            join=True)
                    
if __name__=="__main__":
    main()
