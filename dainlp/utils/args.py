import dataclasses, logging, time, torch
import os
from argparse import ArgumentParser
from dataclasses import dataclass, field, asdict
from enum import Enum
from functools import cached_property
from dainlp.training.optimizer import OptimizerNames
from dainlp.training.scheduler import SchedulerType
from dainlp.training.callback import IntervalStrategy
from transformers.file_utils import torch_required


logger = logging.getLogger(__name__)


'''[TODO] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/training_args.py#L1321'''
class ParallelMode(Enum):
    DISTRIBUTED = "distributed"
    NOT_PARALLEL = "not_parallel"
    NOT_DISTRIBUTED = "not_distributed"


'''[2022-Feb-18] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/hf_argparser.py#L44'''
class HfArgumentParser(ArgumentParser):
    def __init__(self, dataclass_types):
        super(HfArgumentParser, self).__init__()
        assert type(dataclass_types) in [list, tuple]
        self.dataclass_types = dataclass_types
        for dtype in self.dataclass_types:
            self._add_dataclass_arguments(dtype)

    def _add_dataclass_arguments(self, dtype):
        assert not hasattr(dtype, "_argument_group_name")
        for field in dataclasses.fields(dtype):
            if not field.init:
                continue
            assert field.default is not dataclasses.MISSING
            field_name = f"--{field.name}"
            kwargs = field.metadata.copy()
            assert not isinstance(field.type, str)
            if isinstance(field.type, type) and issubclass(field.type, Enum):
                kwargs["choices"] = [x.value for x in field.type]
                kwargs["type"] = type(kwargs["choices"][0])
                kwargs["default"] = field.default
                self.add_argument(field_name, **kwargs)
            elif field.type is bool:
                if field.default is True:
                    self.add_argument(f"--no_{field.name}", action="store_false", dest=field.name, **kwargs)
                else:
                    self.add_argument(f"--{field.name}", action="store_true", dest=field.name, **kwargs)
            else:
                kwargs["type"] = field.type
                kwargs["default"] = field.default
                self.add_argument(field_name, **kwargs)

    def parse_args_into_dataclasses(self):
        namespace, _ = self.parse_known_args()
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype) if f.init}
            values = {k: v for k, v in vars(namespace).items() if k in keys}
            for k in keys:
                delattr(namespace, k)
            outputs.append(dtype(**values))
        assert len(namespace.__dict__) == 0
        return (*outputs, )


'''[TODO]'''
@dataclass
class ArgumentsBase:
    task_name: str = field(default=None)
    dataset_name: str = field(default=None)
    output_metrics_filepath: str = field(default=None)

    model_dir: str = field(default=None)
    config_dir: str = field(default=None)
    tokenizer_dir: str = field(default=None)

    logging_strategy: IntervalStrategy = field(default="steps")
    logging_steps: int = field(default=50)

    seed: int = field(default=42)
    local_rank: int = field(default=-1)
    _n_gpu: int = field(init=False, repr=False, default=-1)

    init_args_time: float = field(default=None)
    complete_running_time: str = field(default=None)

    bf16: bool = field(default=False)
    fp16: bool = field(default=True)

    skip_memory_metrics: bool = field(default=True)


'''[TODO]'''
@dataclass
class TrainingArguments:
    train_filepath: str = field(default=None)
    dev_filepath: str = field(default=None)
    test_filepath: str = field(default=None)
    cache_dir: str = field(default=None)
    label_filepath: str = field(default=None)
    output_dir: str = field(default=None)

    per_device_train_batch_size: int = field(default=16)
    per_device_eval_batch_size: int = field(default=16)
    gradient_accumulation_steps: int = field(default=1)

    learning_rate: float = field(default=5e-5)
    weight_decay: float = field(default=0.0)
    max_grad_norm: float = field(default=1.0)
    num_train_epochs: float = field(default=3.0)
    max_steps: int = field(default=-1)
    lr_scheduler_type: SchedulerType = field(default="linear_with_warmup")
    warmup_ratio: float = field(default=0.1)
    warmup_steps: int = field(default=0)

    save_strategy: IntervalStrategy = field(default="no")
    save_steps: int = field(default=None)
    save_total_limit: int = field(default=3)
    evaluation_strategy: IntervalStrategy = field(default="no")
    eval_steps: int = field(default=None)

    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default=None)
    greater_is_better: bool = field(default=None)
    optim: OptimizerNames = field(default="adamw_hf")


    @property
    def train_batch_size(self):
        return self.per_device_train_batch_size * max(1, self.n_gpu)

    @property
    def should_save(self):
        return self.local_process_index == 0

    @property
    def eval_batch_size(self):
        return self.per_device_eval_batch_size * max(1, self.n_gpu)


'''[TODO]'''
@dataclass
class TestArguments:
    output_predictions_filepath: str = field(default=None)


'''[TODO]'''
@dataclass
class TextArguments:
    max_seq_length: int = field(default=512)
    do_lower_case: bool = field(default=False)


'''[20220330]'''
@dataclass
class Arguments(ArgumentsBase, TrainingArguments, TestArguments, TextArguments):
    def __post_init__(self):
        self.init_args_time = time.time()
        env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if env_local_rank != -1 and env_local_rank != self.local_rank:
            self.local_rank = env_local_rank

        if self.config_dir is None: self.config_dir = self.model_dir
        if self.tokenizer_dir is None: self.tokenizer_dir = self.model_dir

        self.logging_strategy = IntervalStrategy(self.logging_strategy)
        if self.logging_strategy == IntervalStrategy.STEPS:
            assert self.logging_steps > 0

        self.evaluation_strategy = IntervalStrategy(self.evaluation_strategy)
        if self.evaluation_strategy == IntervalStrategy.STEPS:
            assert self.eval_steps > 0
        if self.dev_filepath is not None:
            assert self.evaluation_strategy != IntervalStrategy.NO

        self.save_strategy = IntervalStrategy(self.save_strategy)
        self.lr_scheduler_type = SchedulerType(self.lr_scheduler_type)

        if self.load_best_model_at_end and self.train_filepath is not None:
            assert self.evaluation_strategy == self.save_strategy
            if self.evaluation_strategy == IntervalStrategy.STEPS:
                assert self.save_steps % self.eval_steps == 0
            assert self.metric_for_best_model is not None
        if self.metric_for_best_model is not None:
            assert self.greater_is_better is not None

        self.optim = OptimizerNames(self.optim)


    def __str__(self):
        self_as_dict = asdict(self)
        attrs_as_str = [f"{k}={v}" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}({'; '.join(attrs_as_str)})"

    __repr__ = __str__

    @cached_property
    @torch_required
    def _setup_devices(self):
        assert torch.cuda.is_available()
        if self.local_rank == -1:
            device = torch.device("cuda:0")
            self._n_gpu = torch.cuda.device_count()
        else:
            torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1

        assert device.type == "cuda"
        torch.cuda.set_device(device)
        return device

    @property
    @torch_required
    def device(self):
        return self._setup_devices

    @property
    @torch_required
    def n_gpu(self):
        _ = self._setup_devices
        return self._n_gpu

    @property
    @torch_required
    def parallel_mode(self):
        if self.local_rank != -1:
            return ParallelMode.DISTRIBUTED
        elif self.n_gpu > 1:
            return ParallelMode.NOT_DISTRIBUTED
        else:
            return ParallelMode.NOT_PARALLEL

    @property
    @torch_required
    def world_size(self):
        if self.local_rank != -1:
            return torch.distributed.get_world_size()
        return 1

    @property
    @torch_required
    def process_index(self):
        if self.local_rank != -1:
            return torch.distributed.get_rank()
        return 0

    @property
    @torch_required
    def local_process_index(self):
        if self.local_rank != -1:
            return self.local_rank
        return 0

    @property
    def should_log(self):
        return self.local_process_index == 0


'''[20220330]'''
@dataclass
class ArgumentsForLongformer(Arguments):
    local_size: int = field(default=512)

'''[20220330]'''
@dataclass
class ArgumentsForHiTransformer(Arguments):
    segment_length: int = field(default=64)
    max_num_segments: int = field(default=256)
    add_cls_each_segment: bool = field(default=False)
    do_use_stride: bool = field(default=False)
    do_use_label_wise_attention: bool = field(default=False)