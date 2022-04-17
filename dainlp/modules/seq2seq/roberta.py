import torch, transformers
from dainlp.modules.seq2seq.bert import PreTrainedModel
from dainlp.modules.seq2seq.utils import TransformerLayerList
from dainlp.modules.embeddings.roberta import RobertaEmbeddings


# [2021-07-08] https://github.com/huggingface/transformers/blob/v4.8.2/src/transformers/models/roberta/modeling_roberta.py#L580
class RobertaPreTrainedModel(PreTrainedModel):
    config_class = transformers.RobertaConfig
    base_model_prefix = "roberta"

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


# [2021-07-08; 2021-08-03]
# https://github.com/huggingface/transformers/blob/v4.8.2/src/transformers/models/roberta/modeling_roberta.py#L679
class RobertaModel(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super(RobertaModel, self).__init__(config)
        self.config = config
        self.embeddings = RobertaEmbeddings(config.vocab_size, config.hidden_size, config.pad_token_id,
                                            config.max_position_embeddings, config.type_vocab_size,
                                            config.layer_norm_eps, config.hidden_dropout_prob)
        self.encoder = TransformerLayerList(config)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None):
        input_shape = input_ids.size()
        if attention_mask is None: attention_mask = torch.ones(input_shape, device=input_ids.device)
        if token_type_ids is None: token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, input_ids.device)
        head_mask = self.get_head_mask(None, self.config.num_hidden_layers)
        hidden_states = self.embeddings(input_ids, token_type_ids=token_type_ids, position_ids=position_ids)
        hidden_states = self.encoder(hidden_states, attention_mask=extended_attention_mask, head_mask=head_mask)
        return hidden_states