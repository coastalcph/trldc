import logging, torch
from transformers.optimization import AdamW
from torch.optim import SGD
from transformers.file_utils import ExplicitEnum


logger = logging.getLogger(__name__)


'''[2022-Mar-10] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/training_args.py#L73'''
class OptimizerNames(ExplicitEnum):
    ADAMW_HF = "adamw_hf"
    SGD = "sgd"


'''[2022-Mar-10] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer_pt_utils.py#L996'''
def get_parameter_names(model, skipped_types):
    result = []
    for name, child in model.named_children():
        result += [f"{name}.{n}" for n in get_parameter_names(child, skipped_types) if
                   not isinstance(child, tuple(skipped_types))]
    result += list(model._parameters.keys())
    return result


'''[2022-Mar-10] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer.py#L844'''
def get_optimizer_cls_and_kwargs(args):
    optimizer_kwargs = {"lr": args.learning_rate}
    if args.optim == OptimizerNames.ADAMW_HF:
        adam_kwargs = {"betas": (0.9, 0.999), "eps": 1e-8}
        optimizer_cls = AdamW
        optimizer_kwargs.update(adam_kwargs)
    elif args.optim == OptimizerNames.SGD:
        sgd_kwargs = {"momentum": 0.9}
        optimizer_cls = SGD
        optimizer_kwargs.update(sgd_kwargs)
    else:
        raise ValueError(args.optim)
    return optimizer_cls, optimizer_kwargs


'''[2022-Mar-10] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer.py#L806'''
def create_optimizer(model, args):
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [n for n in decay_parameters if "bias" not in n]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if n in decay_parameters], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if n not in decay_parameters], "weight_decay": 0.0}]
    optimizer_cls, optimizer_kwargs = get_optimizer_cls_and_kwargs(args)
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    return optimizer


'''[TODO]'''
def get_SGD_optimizer(model, learning_rate, weight_decay):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    return optimizer


'''[TODO] https://github.com/princeton-nlp/PURE/blob/main/run_entity.py#L206'''
def get_discriminative_AdamW_optimizer(model, bert_lr, task_lr):
    params = list(model.named_parameters())
    grouped_parameters = []
    grouped_parameters.append({"params": [p for n, p in params if "bert" in n]})
    grouped_parameters.append({"params": [p for n, p in params if "bert" not in n], "lr": task_lr})
    return AdamW(grouped_parameters, lr=bert_lr)