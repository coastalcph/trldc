import numpy, torch


'''[2022-Feb-17] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer_pt_utils.py#L67'''
def pad_and_concatenate(tensor1, tensor2, padding_index=-100):
    """Concatenate `tensor1` and `tensor2` on the first axis (e.g., representing batch size)
    and apply padding on the second axis (e.g., representing sequence length)"""
    if len(tensor1.shape) == 1 or tensor1.shape[1] == tensor2.shape[1]:
        return torch.cat((tensor1, tensor2), dim=0)

    new_shape = (tensor1.shape[0] + tensor2.shape[0], max(tensor1.shape[1], tensor2.shape[1])) + tensor1.shape[2:]
    result = tensor1.new_full(new_shape, padding_index)
    result[:tensor1.shape[0], : tensor1.shape[1]] = tensor1
    result[tensor1.shape[0]:, : tensor2.shape[1]] = tensor2
    return result


'''[2022-Feb-17] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer_pt_utils.py#L97'''
def nested_concat(tensor1, tensor2, padding_index=-100):
    assert type(tensor1) == type(tensor2)
    if isinstance(tensor1, (list, tuple)):
        return type(tensor1)(nested_concat(t1, t2, padding_index) for t1, t2 in zip(tensor1, tensor2))
    elif isinstance(tensor1, torch.Tensor):
        return pad_and_concatenate(tensor1, tensor2, padding_index)
    else:
        raise NotImplemented


'''[2022-Feb-17] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer_pt_utils.py#L148'''
def nested_detach(tensors):
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)
    return tensors.detach()


'''[2022-Feb-17] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer_pt_utils.py#L135'''
def nested_numpify(tensors):
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_numpify(t) for t in tensors)
    t = tensors.cpu()
    assert t.type != torch.bfloat16
    return t.numpy()


'''[2022-Feb-17] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer_pt_utils.py#L331'''
def nested_truncate(tensors, limit):
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_truncate(t, limit) for t in tensors)
    return tensors[:limit]


'''[2022-Feb-17] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer_utils.py#L502'''
def denumpify_detensorize(metrics):
    """Call `.item()`"""
    if isinstance(metrics, (list, tuple)):
        return type(metrics)(denumpify_detensorize(t) for t in metrics)
    elif isinstance(metrics, dict):
        return type(metrics)({k: denumpify_detensorize(v) for k, v in metrics.items()})
    elif isinstance(metrics, torch.Tensor) and metrics.numel() == 1:
        return metrics.item()
    elif isinstance(metrics, numpy.generic):
        return metrics.item()
    return metrics


'''[2022-Mar-10] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer.py#L2471'''
def nested_gather(tensors, local_rank):
    if tensors is None: return
    if local_rank != -1:
        tensors = distributed_concat(tensors)
    return tensors


'''[2022-Mar-10] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer_pt_utils.py#L185'''
def distributed_broadcast_scalars(scalars, device):
    try:
        tensorized_scalar = torch.tensor(scalars).to(device)
        output_tensors = [tensorized_scalar.clone() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensors, tensorized_scalar)
        concat = torch.cat(output_tensors, dim=0)
        return concat
    except AssertionError:
        raise AssertionError("Not using distributed training")


'''[2022-Mar-10] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer_pt_utils.py'''
def distributed_concat(tensor):
    try:
        if isinstance(tensor, (tuple, list)):
            return type(tensor)(distributed_concat(t) for t in tensor)
        output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
        output_tensors = [t if len(t.shape) > 0 else t[None] for t in output_tensors]
        torch.distributed.all_gather(output_tensors, tensor)
        concat = torch.cat(output_tensors, dim=0)
        return concat
    except AssertionError:
        raise AssertionError("Not using distributed training")


'''[2022-Feb-17] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer.py#L2489'''
def pad_across_processes(tensors, padding_index=-100):
    """Pad the tensors to the same size so that they can safely be gathered"""
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(pad_across_processes(t, padding_index) for t in tensors)
    elif isinstance(tensors, dict):
        return type(tensors)({k: pad_across_processes(v, padding_index) for k, v in tensors.items()})
    else:
        assert isinstance(tensors, torch.Tensor)

        if len(tensors.shape) < 2:
            return tensors

        size = torch.tensor(tensors.shape, device=tensors.device)[None].cpu()
        max_size = max(s[1] for s in size)
        if tensors.shape[1] == max_size:
            return tensors

        old_size = tensors.shape
        new_size = list(old_size)
        new_size[1] = max_size
        new_tensor = tensors.new_zeros(tuple(new_size)) + padding_index
        new_tensor[:, :old_size[1]] = tensors
        return tensors
