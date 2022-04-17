import os, psutil, torch


'''[TODO] https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/modeling_utils.py#L121'''
def get_parameter_device(parameter):
    try:
        return next(parameter.parameters()).device
    except StopIteration:
        raise NotImplementedError


'''[TODO] https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/modeling_utils.py#L136'''
def get_parameter_dtype(parameter):
    try:
        return next(parameter.parameters()).dtype
    except StopIteration:
        raise NotImplementedError


'''[TODO] https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/modeling_utils.py#L151'''
class ModuleUtilsMixin:
    @staticmethod
    def _hook_rss_memory_pre_forward(module):
        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        module.mem_rss_pre_forward = mem.rss
        return None

    @staticmethod
    def _hook_rss_memory_post_forward(module):
        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        module.mem_rss_post_forward = mem.rss
        mem_rss_diff = module.mem_rss_post_forward - module.mem_rss_pre_forward
        module.mem_rss_diff = mem_rss_diff + (module.mem_rss_diff if hasattr(module, "mem_rss_diff") else 0)
        return None

    def reset_memory_hooks_state(self):
        for module in self.modules():
            module.mem_rss_diff = 0
            module.mem_rss_post_forward = 0
            module.mem_rss_pre_forward = 0

    def add_memory_hooks(self):
        for module in self.modules():
            module.register_forward_pre_hook(self._hook_rss_memory_pre_forward)
            module.register_forward_hook(self._hook_rss_memory_post_forward)
        self.reset_memory_hooks_state()

    @property
    def device(self):
        return get_parameter_device(self)

    @property
    def dtype(self):
        return get_parameter_dtype(self)

    def invert_attention_mask(self, encoder_attention_mask):
        raise NotImplementedError

    def create_extended_attention_mask_for_decoder(self, input_shape, attention_mask, device):
        raise NotImplementedError

    def get_extended_attention_mask(self, attention_mask, input_shape, device):
        # 1 for tokens to attend to, 0 for tokens to ignore
        if attention_mask.dim() == 3:  # [batch size, from sequence length, to sequence length]
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:  # [batch size, sequence length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError

        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        # 0 for tokens to attend to, -10000.0 for tokens to ignore
        # this makes sense when we add it to the raw scores before the softmax
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        assert head_mask is None
        head_mask = [None] * num_hidden_layers
        return head_mask

    def num_parameters(self, only_trainable=False, exclude_embeddings=False):
        if exclude_embeddings:
            embedding = [f"{n}.weight" for n, m in self.named_modules() if isinstance(m, torch.nn.Embedding)]
            non_embedding = [p for n, p in self.named_parameters() if n not in embedding]
            return sum(p.numel() for p in non_embedding if p.requires_grad or not only_trainable)
        else:
            return sum(p.numel() for p in self.parameters() if p.requires_grad or not only_trainable)

    def estimate_tokens(self, input_dict):
        assert "input_ids" in input_dict
        return input_dict["input_ids"].numel()

    def floating_point_ops(self, input_dict, exclude_embeddings=True):
        return 6 * self.estimate_tokens(input_dict) * self.num_parameters(exclude_embeddings=exclude_embeddings)