import logging, numpy, random, torch


logger = logging.getLogger(__name__)


'''[2022-Feb-17] https://github.com/LorrinWWW/Pyramid/blob/master/utils/data.py#L327'''
def pad_sequences(sequences, max_length=None, dtype="int32", value=0):
    batch_size = len(sequences)
    if max_length is None:
        max_length = max([len(s) for s in sequences])

    sample_shape = numpy.asarray(sequences[0]).shape[1:]
    padded_sequences = numpy.full((batch_size, max_length) + sample_shape, value, dtype=dtype)
    for i, s in enumerate(sequences):
        assert len(s) > 0
        trunc_s = numpy.asarray(s[:max_length], dtype=dtype)
        padded_sequences[i, :len(trunc_s)] = trunc_s

    return padded_sequences


'''[2022-Feb-17] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer_utils.py#L50'''
def set_seed(seed=52):
    """Fix the random seed for reproducibility"""
    if seed < 0: return
    logger.debug(f"Random seed: {seed}")
    # os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cuda.matmul.allow_tf32 = False