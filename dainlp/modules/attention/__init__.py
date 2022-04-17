import math, torch


'''[Mar-30-2022] 
https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/bert/modeling_bert.py#L226'''
class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout,
                 position_embedding_type="absolute", output_dim=None):
        super(MultiHeadSelfAttention, self).__init__()
        assert hidden_size % num_attention_heads == 0

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = torch.nn.Linear(hidden_size, self.all_head_size)
        self.key = torch.nn.Linear(hidden_size, self.all_head_size)
        self.value = torch.nn.Linear(hidden_size, self.all_head_size)

        self.dropout = torch.nn.Dropout(dropout)

        if position_embedding_type != "absolute":
            raise NotImplementedError

        self.out = None if output_dim is None else torch.nn.Linear(self.all_head_size, output_dim)
        self.out_dropout = None if output_dim is None else torch.nn.Dropout(dropout)

    def transpose_for_scores(self, x): # [bs, sq, dim]
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3) # [bs, number of attention heads, sq, attention head size]

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # [bs, number of attention heads, sq, sq]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None: # value in attention_mask will be 0 or -10000.0
            attention_scores = attention_scores + attention_mask
        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores) # [bs, number of attention heads, sq, sq]
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # [bs, number of attention heads, sq, attention head size]
        context_layer = torch.matmul(attention_probs, value_layer)
        # [bs, sq, number of attention heads, attention head size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        final_shape = context_layer.size()[:-2] + (self.all_head_size, )
        outputs = context_layer.view(final_shape)

        if self.out is not None:
            outputs = self.out_dropout(self.out(outputs))
        return outputs # [bs, sq, dim]
