import numpy as np, torch
from dainlp.utils import pad_sequences


'''[2022-Jan-16] https://github.com/LorrinWWW/Pyramid/blob/master/layers/indexings.py#L169'''
class CharVocab:
    def __init__(self):
        self.char2idx = {"[PAD]": 0, "[UNK]": 1}
        self.can_update = True

    def get_char_idx(self, char):
        if char in self.char2idx:
            return self.char2idx[char]
        elif self.can_update:
            self.char2idx[char] = len(self.char2idx)
            return self.char2idx[char]
        else:
            return self.char2idx["[UNK]"]

    def get_indices(self, sentences, depth=3):
        # 3: sentences; 2: tokens; 1: chars; 0: char
        if depth > 0:
            return [self.get_indices(sentence, depth=depth - 1) for sentence in sentences]
        return self.get_char_idx(sentences)

    def __call__(self, sentences):
        indices = self.get_indices(sentences)
        max_num_tokens_per_sentence = max(len(s) for s in indices)
        max_num_chars_per_token = max(max(len(t) for t in s) for s in indices)

        dummy_token = [0] * max_num_chars_per_token
        for i, tokens in enumerate(indices):
            for j, chars in enumerate(tokens):
                tokens[j] = chars + [0] * (max_num_chars_per_token - len(chars))
            indices[i] = tokens + [dummy_token] * (max_num_tokens_per_sentence - len(tokens))

        return np.array(indices, dtype="long")


'''[2022-Jan-16] https://github.com/LorrinWWW/Pyramid/blob/master/layers/indexings.py#L30'''
class TokenVocab:
    def __init__(self, token2idx={"[PAD]": 0, "[MASK]": 1, "[CLS]": 2, "[SEP]": 3, "[UNK]": 4},
                 can_update=True, cased=False):
        self.token2idx = token2idx
        self.can_update = can_update
        self.cased = cased

    def get_token_idx(self, token):
        if not self.cased: token = token.lower()

        if token in self.token2idx:
            return self.token2idx[token]
        elif self.can_update:
            self.token2idx[token] = len(self.token2idx)
            return self.token2idx[token]
        else:
            return self.token2idx["[UNK]"]

    def get_indices(self, sentences, depth=2):
        # 2: sentences; 1: tokens; 0: token
        if depth > 0:
            return [self.get_indices(sentence, depth=depth - 1) for sentence in sentences]
        return self.get_token_idx(sentences)

    def __call__(self, sentences):
        indices = self.get_indices(sentences)
        indices = pad_sequences(indices, max_length=None, dtype="long", value=0)
        return indices


'''[TODO] https://github.com/LorrinWWW/Pyramid/blob/master/layers/indexings.py#L30'''
class LabelVocab:
    def __init__(self, label2idx={"O": 0}):
        self.label2idx = label2idx
        self.idx2label = {}
        self.update()

    def update(self):
        if len(self.label2idx) != len(self.idx2label):
            self.idx2label = {v: k for k, v in self.label2idx.items()}

    def get_label(self, idx, default_label="O"):
        self.update()
        return self.idx2label.get(idx, default_label)

    def get_label_idx(self, label):
        if label in self.label2idx:
            return self.label2idx[label]
        else:
            self.label2idx[label] = len(self.label2idx)
            return self.label2idx[label]

    def get_indices(self, sentences, depth=2):
        # 2: sentences; 1: tokens; 0: token
        if depth > 0:
            return [self.get_indices(sentence, depth=depth - 1) for sentence in sentences]
        return self.get_label_idx(sentences)

    def get_labels(self, indices, default_label="O"):
        if isinstance(indices, str):
            return indices
        elif isinstance(indices, int) or (type(indices).__module__ == "numpy" and indices.shape == ()):
            return self.get_label(indices, default_label)
        else:
            assert hasattr(indices, "__iter__")
            return [self.get_labels(i) for i in indices]

    def __call__(self, sentences):
        indices = self.get_indices(sentences)
        indices = pad_sequences(indices, max_length=None, dtype="long", value=0)
        return indices

    @staticmethod
    def convert_to_tensors(vocab, labels):
        labels = np.array(labels)
        tensor_list = []
        for i in range(labels.shape[1]):
            tensor_list.append(torch.from_numpy(vocab(labels[:, i])))
        return tensor_list