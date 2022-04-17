import torch
from dainlp.modules.embeddings.bert import BertEmbeddings


# [2021-06-15; 2021-10-29]
# https://github.com/huggingface/transformers/blob/v4.12.0/src/transformers/models/roberta/modeling_roberta.py#L1559
def create_position_ids_from_input_ids(input_ids, padding_idx):
    '''Replace non-padding symbols with their position numbers'''
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
    return incremental_indices.long() + padding_idx


# [2021-06-15; 2021-08-03; 2021-10-29]
# https://github.com/huggingface/transformers/blob/v4.12.0/src/transformers/models/roberta/modeling_roberta.py#L70
class RobertaEmbeddings(BertEmbeddings):
    def __init__(self, vocab_size, hidden_size, pad_token_id, max_position_embeddings, type_vocab_size,
                 layer_norm_eps, hidden_dropout_prob, position_embedding_type="absolute"):
        super(RobertaEmbeddings, self).__init__(vocab_size, hidden_size, pad_token_id, max_position_embeddings,
                                                type_vocab_size, layer_norm_eps, hidden_dropout_prob,
                                                position_embedding_type)
        self.padding_idx = pad_token_id
        self.position_embeddings = torch.nn.Embedding(max_position_embeddings, hidden_size,
                                                      padding_idx=self.padding_idx)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        if position_ids is None:
            position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx).to(input_ids.device)
        return super(RobertaEmbeddings, self).forward(input_ids, token_type_ids=token_type_ids,
                                                      position_ids=position_ids)