import torch
from dainlp.modules.seq2seq.roberta import RobertaPreTrainedModel, RobertaModel
from dainlp.modules.utils import get_sinusoidal_embeddings
from dainlp.modules.seq2vec.lwan import MullenbachModel


'''[2022-Mar-13]'''
class Model(RobertaPreTrainedModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        # pre-trained RoBERTa encoder, being able to take care of no longer than 512 wordpieces
        self.roberta = RobertaModel(config)
        # Init sinusoidal positional embeddings
        self.seg_pos_embeddings = torch.nn.Embedding(config.max_segments + 1, config.hidden_size, padding_idx=0,
                                                     _weight=get_sinusoidal_embeddings(config.max_segments + 1,
                                                                                       config.hidden_size,
                                                                                       padding_idx=0))
        self.seg_encoder = torch.nn.Transformer(d_model=config.hidden_size, nhead=config.num_attention_heads,
                                                batch_first=True, dim_feedforward=config.intermediate_size,
                                                activation=config.hidden_act, dropout=config.hidden_dropout_prob,
                                                layer_norm_eps=config.layer_norm_eps, num_encoder_layers=2,
                                                num_decoder_layers=0).encoder

        if config.do_use_label_wise_attention:
            self.lwan = MullenbachModel(config.hidden_size, config.num_labels)
        else:
            self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
            self.out_proj = torch.nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        # Example
        # Batch of 4 documents: (batch_size, n_segments, max_segment_length) --> (4, 64, 128)
        # Squash samples and segments into a single axis (batch_size * n_segments, max_segment_length) --> (256, 128)
        batch_size, n_segments, max_segment_length = input_ids.size()
        input_ids_reshape = input_ids.contiguous().view(-1, input_ids.size(-1))
        attention_mask_reshape = attention_mask.contiguous().view(-1, attention_mask.size(-1))
        if token_type_ids is None:
            token_type_ids_reshape = None
        else:
            token_type_ids_reshape = token_type_ids.contiguous().view(-1, token_type_ids.size(-1))

        # Encode segments with RoBERTa --> (256, 128, 768)
        encoder_outputs = self.roberta(input_ids=input_ids_reshape, attention_mask=attention_mask_reshape,
                                       token_type_ids=token_type_ids_reshape)

        # Reshape back to (batch_size, n_segments, max_segment_length, output_size) --> (4, 64, 128, 768)
        encoder_outputs = encoder_outputs.contiguous().view(batch_size, n_segments, max_segment_length,
                                                            self.config.hidden_size)

        # Gather CLS outputs per segment --> (4, 64, 768)
        encoder_outputs = encoder_outputs[:, :, 0]

        # Infer real segments, i.e., mask paddings
        # seg_mask = (torch.sum(input_ids, 2) != max_segment_length).to(input_ids.dtype)
        seg_mask = (torch.sum(attention_mask, 2) != 0).to(input_ids.dtype)
        # Infer and collect segment positional embeddings
        seg_positions = torch.arange(1, n_segments + 1).to(input_ids.device) * seg_mask
        # Add segment positional embeddings to segment inputs
        encoder_outputs += self.seg_pos_embeddings(seg_positions)
        # Encode segments with segment-wise transformer
        seg_encoder_outputs = self.seg_encoder(encoder_outputs)

        if self.config.do_use_label_wise_attention:
            logits = self.lwan(seg_encoder_outputs)
        else:
            # Max pooling contextual segmental representations into document representation
            outputs, _ = torch.max(seg_encoder_outputs, 1)
            hidden_states = torch.tanh(self.dense(self.dropout(outputs)))
            logits = self.out_proj(self.dropout(hidden_states))

        if self.config.task_name == "singlelabel":
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        else:
            assert self.config.task_name == "multilabel"
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}