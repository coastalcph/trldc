import collections, logging, math, numpy, os, torch
from collections.abc import Mapping
from dainlp.utils.files import remove_checkpoints
from dainlp.utils.tensors import nested_detach


logger = logging.getLogger(__name__)


'''[2022-Mar-10] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer.py#L1528
    A state_dict is simply a Python dictionary object that maps each layer to its parameter tensor'''
def load_state_dict_in_model(model, state_dict):
    load_result = model.load_state_dict(state_dict, strict=False)
    if len(load_result.missing_keys) != 0:
        assert set(load_result.missing_keys) == set(model._keys_to_ignore_on_save)
        model.tie_weights()
    assert len(load_result.unexpected_keys) == 0


'''[2022-Mar-10] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer.py#L2005
                 https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer.py#L2086
    Save the model (tokenizer, training_args) using save_pretrained or torch.save
    Will only save from the main process.
    Do not consider TPU, SageMaker (Amazon), Sharded (Microsoft), DeepSpeed (Microsoft); do not push to the Hub'''
def save_model(model, output_dir, tokenizer, args):
    if not args.should_save: return
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving model checkpoint to {output_dir}")
    if hasattr(unwrap_model(model), "save_pretrained"):
        model = unwrap_model(model)
        model.save_pretrained(output_dir, state_dict=model.state_dict())
    elif hasattr(model, "save_pretrained"):
        model.save_pretrained(output_dir)
    else:
        torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
    torch.save(args, os.path.join(output_dir, "training_args.bin"))


'''[2022-Mar-10] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer.py#L566
    Don't consider group_by_length'''
def get_train_sampler(train_dataset, args):
    assert isinstance(train_dataset, collections.abc.Sized)
    if args.world_size <= 1:
        generator = torch.Generator()
        generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        return torch.utils.data.RandomSampler(train_dataset, generator=generator)
    else:
        return torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.world_size,
                                                               rank=args.process_index, seed=args.seed)


'''[2022-Mar-10] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer.py#L630
    train_dataset has __len__ and it is not a datasets.Dataset'''
def get_train_dataloader(train_dataset, collate_fn, args):
    train_sampler = get_train_sampler(train_dataset, args)
    return torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, sampler=train_sampler,
                                       collate_fn=collate_fn, pin_memory=True)


'''[2022-Mar-10] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer_pt_utils.py#L641'''
class ShardSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size=1, num_processes=1, process_index=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_processes = num_processes
        self.process_index = process_index
        self.total_batch_size = batch_size * num_processes
        num_batches = math.ceil(len(dataset) / self.total_batch_size)
        self.total_num_samples = num_batches * self.total_batch_size

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        while len(indices) < self.total_num_samples:
            indices += indices[: (self.total_num_samples - len(indices))]

        result = []
        for start in range(self.batch_size * self.process_index, self.total_num_samples, self.total_batch_size):
            result += indices[start:start + self.batch_size]
        return iter(result)

    def __len__(self):
        return self.total_num_samples // self.num_processes


'''[2022-Mar-10] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer.py#L705
                 https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer.py#L676
                 https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer.py#L752'''
def get_eval_dataloader(eval_dataset, collate_fn, args):
    if args.world_size <= 1:
        eval_sampler = torch.utils.data.SequentialSampler(eval_dataset)
    else:
        eval_sampler = ShardSampler(eval_dataset, batch_size=args.per_device_eval_batch_size,
                                    num_processes=args.world_size, process_index=args.process_index)
    return torch.utils.data.DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                       collate_fn=collate_fn, pin_memory=True)


'''[2022-Mar-10] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/modeling_utils.py#L2261'''
def unwrap_model(model):
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


'''[2022-Mar-10] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer.py#L989'''
def wrap_model(model, args, training=True):
    if unwrap_model(model) is not model: # already wrapped
        return model
    assert args.n_gpu > 0
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    if not training:
        return model
    if args.local_rank != -1:
        kwargs = {"find_unused_parameters": True}
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank, **kwargs)
    return model


'''[2022-Mar-10] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer.py#L1869
    Take care of a tensor or a nested list/dictionary of tensors'''
def prepare_input(data, device):
    if isinstance(data, Mapping):
        return type(data)({k: prepare_input(v, device) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(prepare_input(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        kwargs = dict(device=device)
        return data.to(**kwargs)
    return data


'''[2022-Mar-10] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer.py#L1613
    Do not save optimizer and scheduler; do not save RNG state'''
def save_checkpoint(model, tokenizer, args, state, metrics=None):
    checkpoint_folder = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
    # TODO: self.store_flos()
    save_model(model, checkpoint_folder, tokenizer=tokenizer, args=args)
    if metrics is not None and args.metric_for_best_model is not None:
        metric_to_check = args.metric_for_best_model
        if not metric_to_check.startswith("eval_"): metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics[metric_to_check]

        operator = numpy.greater if args.greater_is_better else numpy.less
        if (state.best_metric is None or state.best_model_checkpoint is None
                or operator(metric_value, state.best_metric)):
            state.best_metric = metric_value
            state.best_model_checkpoint = checkpoint_folder

    if args.should_save:
        state.save_to_json(os.path.join(checkpoint_folder, "trainer_state.json"))
        remove_checkpoints(args.output_dir, prefix="checkpoint", save_total_limit=args.save_total_limit,
                           best_model_checkpoint=state.best_model_checkpoint)


'''[2022-Mar-10] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer.py#L2521'''
def prediction_step(model, inputs, device):
    assert "labels" in inputs
    inputs = prepare_input(inputs, device)
    golds = nested_detach(inputs["labels"])
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            outputs = model(**inputs)
        loss = outputs["loss"].mean().detach()
        if "logits" not in outputs: return {"loss": loss}
        logits = outputs["logits"]
    logits = nested_detach(logits)
    return {"loss": loss, "logits": logits, "golds": golds}


'''[2022-Mar-10] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer.py#L1913
                 do_grad_scaling is True, using fp16 and amp'''
def training_step(model, inputs, scaler, args):
    model.train()
    inputs = prepare_input(inputs, args.device)

    with torch.cuda.amp.autocast():
        loss = model(**inputs)["loss"]

    if args.n_gpu > 1:
        loss = loss.mean()
    if args.gradient_accumulation_steps > 1:
        loss = loss / args.gradient_accumulation_steps
    scaler.scale(loss).backward()
    return loss.detach()
