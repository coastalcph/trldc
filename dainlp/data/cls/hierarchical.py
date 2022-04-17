import math, torch


'''[20220330]'''
class Collator:
    def __init__(self, tokenizer, segment_length=128, max_num_segments=256, do_use_stride=False,
                 add_cls_each_segment=False, task_name="singlelabel"):
        self.cls_token_id = tokenizer.cls_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.segment_length = segment_length
        self.add_cls_each_segment = add_cls_each_segment
        self.effective_segment_length = segment_length - 1 if add_cls_each_segment else segment_length
        self.max_num_segments = max_num_segments
        self.stride = int(segment_length / 8) if do_use_stride else 0
        self.task_name = task_name

    def convert_labels(self, features):
        if self.task_name == "singlelabel":
            return torch.tensor([f["labels"][0] for f in features], dtype=torch.int64)
        elif self.task_name == "multilabel":
            return torch.tensor([f["labels"] for f in features], dtype=torch.float)
        raise NotImplementedError

    def __call__(self, features):
        max_seq_length = max([len(f["input_ids"]) for f in features])
        max_num_segments = math.ceil((max_seq_length - 2) / (self.effective_segment_length - self.stride))
        max_num_segments = min(max_num_segments, self.max_num_segments)

        batch = {"input_ids": [], "attention_mask": []}
        for f in features:
            input_ids, attention_mask = [], []
            start = 1 # due to the first [CLS] token
            for _ in range(max_num_segments):
                end = start + self.effective_segment_length
                if start > (len(f["input_ids"]) - self.stride):
                    # too short, add only pading tokens
                    input_ids += [self.pad_token_id] * self.segment_length
                    attention_mask += [0] * self.segment_length
                elif end > len(f["input_ids"]):
                    # the last segment
                    segment_ids = f["input_ids"][start:]
                    if self.add_cls_each_segment:
                        segment_ids = [self.cls_token_id] + segment_ids
                    attention_mask += [1] * len(segment_ids) + [0] * (self.segment_length - len(segment_ids))
                    input_ids += segment_ids
                    input_ids += [self.pad_token_id] * (self.segment_length - len(segment_ids))
                else:
                    if self.add_cls_each_segment:
                        input_ids += [self.cls_token_id] + f["input_ids"][start:end]
                    else:
                        input_ids += f["input_ids"][start:end]
                    attention_mask += [1] * self.segment_length
                assert len(input_ids) == len(attention_mask)
                start = end - self.stride

            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)
        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        batch["input_ids"] = batch["input_ids"].view(len(features), max_num_segments, self.segment_length)
        batch["attention_mask"] = batch["attention_mask"].view(len(features), max_num_segments, self.segment_length)
        batch["labels"] = self.convert_labels(features)

        return batch