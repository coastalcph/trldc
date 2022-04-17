import json, logging, os, torch
from filelock import FileLock


logger = logging.getLogger(__name__)


'''[20220329]'''
class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, filepath, args, tokenizer=None, split=None, label2idx=None):
        assert (label2idx is not None) or (split == "train")
        self.args = args

        if args.cache_dir is not None:
            os.makedirs(args.cache_dir, exist_ok=True)
            with FileLock(os.path.join(args.cache_dir, f"{split}.lock")):
                if os.path.exists(os.path.join(args.cache_dir, f"{split}.features")):
                    self.examples = torch.load(os.path.join(args.cache_dir, f"{split}.examples"))
                    self.label2idx = torch.load(os.path.join(args.cache_dir, f"{split}.label2idx"))
                    self.features = torch.load(os.path.join(args.cache_dir, f"{split}.features"))
                    logger.info(f"Loading {len(self.features)} examples from cached directory {args.cache_dir}")
                else:
                    self.load_from_filepath(filepath, tokenizer, label2idx)
                    torch.save(self.examples, os.path.join(args.cache_dir, f"{split}.examples"))
                    torch.save(self.label2idx, os.path.join(args.cache_dir, f"{split}.label2idx"))
                    torch.save(self.features, os.path.join(args.cache_dir, f"{split}.features"))
        else:
            self.load_from_filepath(filepath, tokenizer, label2idx)

    def load_from_filepath(self, filepath, tokenizer, label2idx):
        self.examples = [json.loads(l.strip()) for l in open(filepath).readlines()]
        self.label2idx = label2idx if label2idx is not None else self.build_label2idx_from_examples()
        self.features = self.convert_examples_to_features(self.examples, tokenizer)
        logger.info(f"Loading {len(self.features)} examples from file {filepath}")

    def build_label2idx_from_examples(self):
        labels = set()
        for e in self.examples:
            if self.args.task_name == "multilabel":
                labels = labels.union(set(e["labels"]))
            elif self.args.task_name == "singlelabel":
                labels.add(e["label"])
        return {l: i for i, l in enumerate(sorted(labels))}

    def get_example_label(self, example):
        if self.args.task_name == "singlelabel":
            return [self.label2idx[example["label"]]]
        elif self.args.task_name == "multilabel":
            label_ids = [0] * len(self.label2idx)
            for l in example["labels"]:
                label_ids[self.label2idx[l]] = 1
            return label_ids
        raise ValueError(f"Unknown task: {self.args.task_name}")

    def convert_examples_to_features(self, examples, tokenizer, text_field="text"):
        features = []
        for example in examples:
            text = example[text_field]
            if self.args.do_lower_case: text = text.lower()
            outputs = tokenizer(text, padding=False, truncation=True, max_length=self.args.max_seq_length)
            feature = {"input_ids": outputs["input_ids"], "labels": self.get_example_label(example)}
            features.append(feature)

        if len(examples) > 0:
            logger.info(examples[0])
            logger.info(features[0])

        return features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]


'''[20220329]'''
class Collator:
    def __init__(self, tokenizer, max_seq_length, task_name="singlelabel"):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.task_name = task_name

    def __call__(self, features):
        max_seq_length = max([len(f["input_ids"]) for f in features])
        max_seq_length = min(max_seq_length, self.max_seq_length)
        batch = self.tokenizer.pad(features, padding=True, max_length=max_seq_length)
        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}

        assert self.task_name in ["singlelabel", "multilabel"]
        if self.task_name == "singlelabel":
            batch["labels"] = torch.tensor([f["labels"][0] for f in features], dtype=torch.int64)
        else:
            batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.float)
        return batch


