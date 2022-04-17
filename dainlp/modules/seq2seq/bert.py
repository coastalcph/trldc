import logging, os, re, torch, transformers
from dainlp.modules import ModuleUtilsMixin
from dainlp.modules.seq2seq.utils import TransformerLayerList
from dainlp.modules.embeddings.bert import BertEmbeddings
from transformers import BertConfig


logger = logging.getLogger(__name__)


# [2021-07-08] https://github.com/huggingface/transformers/blob/v4.8.2/src/transformers/modeling_utils.py#L407
class PreTrainedModel(torch.nn.Module, ModuleUtilsMixin):
    config_class = None
    base_model_prefix = ""
    _keys_to_ignore_on_load_missing = None
    _keys_to_ignore_on_load_unexpected = None
    _keys_to_ignore_on_save = None
    is_parallelizable = False

    @property
    def dummy_inputs(self):
        return {"input_ids": torch.tensor([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])}

    def __init__(self, config):
        super(PreTrainedModel, self).__init__()
        self.config = config
        self.name_or_path = config.name_or_path

    @property
    def base_model(self):
        return getattr(self, self.base_model_prefix, self)

    def get_input_embeddings(self):
        base_model = getattr(self, self.base_model_prefix, self)
        assert base_model is not self
        return base_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        base_model = getattr(self, self.base_model_prefix, self)
        assert base_model is not self
        base_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return None
        
    def tie_weights(self):
        output_embeddings = self.get_output_embeddings()
        if output_embeddings is not None and self.config.tie_word_embeddings:
            self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        logger.info("Tie the weights: input embeddings and output embeddings")
        output_embeddings.weight = input_embeddings.weight
        if getattr(output_embeddings, "bias", None) is not None:
            pad_length = output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0]
            output_embeddings.bias.data = torch.nn.functional.pad(output_embeddings.bias.data, (0, pad_length))
        if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
            output_embeddings.out_features = input_embeddings.num_embeddings

    def resize_token_embeddings(self, new_num_tokens=None):
        model_embeds = self._resize_token_embeddings(new_num_tokens)
        if new_num_tokens is None: return model_embeds
        self.config.vocab_size = new_num_tokens
        self.vocab_size = new_num_tokens
        self.tie_weights()
        return model_embeds

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)
        if self.get_output_embeddings() is not None and not self.config.tie_word_embeddings:
            raise NotImplemented
        return self.get_input_embeddings()

    def _get_resized_embeddings(self, old_embeddings, new_num_tokens=None):
        if new_num_tokens is None: return old_embeddings
        old_num_tokens, embedding_dim = old_embeddings.weight.size()
        if old_num_tokens == new_num_tokens: return old_embeddings

        new_embeddings = torch.nn.Embedding(new_num_tokens, embedding_dim).to(self.device,
                                                                              dtype=old_embeddings.weight.dtype)
        self._init_weights(new_embeddings)
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[:num_tokens_to_copy, :]
        return new_embeddings

    def save_pretrained(self, save_directory, state_dict=None):
        assert not os.path.isfile(save_directory)
        os.makedirs(save_directory, exist_ok=True)
        assert not hasattr(self, "module")
        model_to_save = self
        model_to_save.config.architectures = [model_to_save.__class__.__name__]
        model_to_save.config.save_pretrained(save_directory)
        if state_dict is None: state_dict = model_to_save.state_dict()
        if self._keys_to_ignore_on_save is not None:
            state_dict = {k: v for k, v in state_dict.items() if k not in self._keys_to_ignore_on_save}
        torch.save(state_dict, os.path.join(save_directory, "pytorch_model.bin"))
        logger.info(f"\tModel weights saved in {save_directory}")

    def init_weights(self):
        return

    @classmethod
    def from_pretrained(cls, pretrained_dir, config):
        assert os.path.isdir(pretrained_dir)
        config.name_or_path = pretrained_dir
        with transformers.modeling_utils.no_init_weights(_enable=True):
            model = cls(config)

        weights_file = os.path.join(pretrained_dir, "pytorch_model.bin")
        if os.path.isfile(weights_file):
            logger.info(f"Loading weights file {weights_file}")
            state_dict = torch.load(weights_file, map_location="cpu")
            model = cls._load_state_dict_into_model(model, state_dict, pretrained_dir)
            model.tie_weights()
        else:
            logger.info(f"Weights are randomly initialized weights")
        model.eval()
        return model

    @classmethod
    def _load_state_dict_into_model(cls, model, state_dict, pretrained_dir, _fast_init=True):
        old_keys, new_keys = [], []
        for key in state_dict.keys():
            new_key = None
            if "gamma" in key: new_key = key.replace("gamma", "weight")
            if "beta" in key: new_key = key.replace("beta", "bias")
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        expected_keys = list(model.state_dict().keys())
        loaded_keys = list(state_dict.keys())
        prefix = model.base_model_prefix

        has_prefix_module = any(s.startswith(prefix) for s in loaded_keys)
        expects_prefix_module = any(s.startswith(prefix) for s in expected_keys)
        remove_prefix = not has_prefix_module and expects_prefix_module # copy model without prefix to model with prefix
        add_prefix = has_prefix_module and not expects_prefix_module # copy model with prefix to model without prefix
        if remove_prefix:
            expected_keys = [".".join(s.split(".")[1:]) if s.startswith(prefix) else s for s in expected_keys]
        elif add_prefix:
            expected_keys = [".".join([prefix, s]) for s in expected_keys]

        missing_keys = list(set(expected_keys) - set(loaded_keys))
        unexpected_keys = list(set(loaded_keys) - set(expected_keys))
        if cls._keys_to_ignore_on_load_missing is not None:
            for pat in cls._keys_to_ignore_on_load_missing:
                missing_keys = [k for k in missing_keys if re.search(pat, k) is None]
        if cls._keys_to_ignore_on_load_unexpected is not None:
            for pat in cls._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

        unintialized_modules = model.retrieve_modules_from_names(missing_keys, add_prefix, remove_prefix)
        for module in unintialized_modules:
            model._init_weights(module)

        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None: state_dict._metadata = metadata

        error_msgs = []
        # PyTorch's _load_from_state_dict does not copy parameters in a module's descendants, so need this recursion
        def load(module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(state_dict, prefix, local_metadata, True, [], [], error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        start_prefix = ""
        model_to_load = model
        if not hasattr(model, cls.base_model_prefix) and has_prefix_module:
            start_prefix = cls.base_model_prefix + "."
        if hasattr(model, cls.base_model_prefix) and not has_prefix_module:
            model_to_load = getattr(model, cls.base_model_prefix)
        load(model_to_load, prefix=start_prefix)

        if len(unexpected_keys) > 0:
            unexpected_keys = ["Unexpected keys:"] + unexpected_keys
            logger.info("\n\t".join(unexpected_keys))
        if len(missing_keys) > 0:
            missing_keys = ["Missing keys:"] + missing_keys
            logger.info("\n\t".join(missing_keys))
        if len(error_msgs) > 0: logger.info("\n\t".join(error_msgs))
        return model

    def retrieve_modules_from_names(self, names, add_prefix=False, remove_prefix=False):
        module_keys = set([".".join(k.split(".")[:-1]) for k in names])
        # torch.nn.ParameterList uses names like 'bert.special_embeddings.0
        module_keys = module_keys.union(set([".".join(k.split(".")[:-2]) for k in names if k[-1].isdigit()]))
        retrieved_modules = []
        for name, module in self.named_modules():
            if remove_prefix:
                name = ".".join(name.split(".")[1:]) if name.startswith(self.base_model_prefix) else name
            elif add_prefix:
                name = ".".join([self.base_model_prefix, name]) if len(name) > 0 else self.base_model_prefix
            if name in module_keys:
                retrieved_modules.append(module)
        return retrieved_modules


# [TODO] https://github.com/huggingface/transformers/blob/v4.5.0/src/transformers/models/bert/modeling_bert.py#L694
class BertPreTrainedModel(PreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "bert"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


# [2021-07-29] https://github.com/huggingface/transformers/blob/v4.9.1/src/transformers/models/bert/modeling_bert.py#L842
class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config.vocab_size, config.hidden_size, config.pad_token_id,
                                         config.max_position_embeddings, config.type_vocab_size,
                                         config.layer_norm_eps, config.hidden_dropout_prob,
                                         config.position_embedding_type)
        self.encoder = TransformerLayerList(config)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        batch_size, seq_length = input_ids.size()
        if attention_mask is None: attention_mask = torch.ones((batch_size, seq_length), device=input_ids.device)
        if token_type_ids is None:
            token_type_ids = self.embeddings.token_type_ids[:, :seq_length].expand(batch_size, seq_length)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask)
        extended_head_mask = [None] * self.config.num_hidden_layers
        embedding_output = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(embedding_output, attention_mask=extended_attention_mask,
                                       head_mask=extended_head_mask)
        return encoder_outputs # (batch_size, sequence_length, hidden_size)