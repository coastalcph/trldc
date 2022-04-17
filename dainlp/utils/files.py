import json, logging, numpy, os, re, shutil
from pathlib import Path


logger = logging.getLogger(__name__)


'''[2021-Dec-31]'''
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(NumpyJSONEncoder, self).default(obj)


'''[2021-Aug-18]'''
def make_sure_parent_dir_exists(filepath):
    parent_dir = os.path.dirname(filepath)
    os.makedirs(parent_dir, exist_ok=True)


'''[2021-Aug-18]'''
def write_list_to_json_file(data, filepath):
    make_sure_parent_dir_exists(filepath)

    with open(filepath, "w") as f:
        for i in data:
            f.write(f"{json.dumps(i, cls=NumpyJSONEncoder)}\n")


'''[2021-Aug-19]'''
def write_object_to_json_file(data, filepath, sort_keys=False):
    make_sure_parent_dir_exists(filepath)
    json.dump(data, open(filepath, "w"), indent=2, sort_keys=sort_keys, default=lambda o: "Unknown")


# [2021-08-29]
def move_best_checkpoint(output_dir, best_checkpoint):
    output_dir = os.path.abspath(output_dir)
    best_checkpoint = os.path.abspath(best_checkpoint)
    for filename in os.listdir(best_checkpoint):
        shutil.move(f"{best_checkpoint}/{filename}", f"{output_dir}/{filename}")
    for checkpoint in os.listdir(output_dir):
        if checkpoint.startswith("checkpoint"): shutil.rmtree(f"{output_dir}/{checkpoint}")


'''[2022-Mar-10] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer.py#L2146'''
def remove_checkpoints(output_dir, prefix, save_total_limit, best_model_checkpoint=None):
    if save_total_limit is None or save_total_limit <= 0: return
    checkpoints_sorted = sorted_checkpoints(output_dir, prefix, best_model_checkpoint)
    if len(checkpoints_sorted) <= save_total_limit: return
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
        shutil.rmtree(checkpoint)


'''[2022-Mar-10] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer.py#L2122'''
def sorted_checkpoints(output_dir, prefix, best_model_checkpoint):
    ignored = [] if best_model_checkpoint is None else [str(Path(best_model_checkpoint))]

    checkpoints = []
    for path in Path(output_dir).glob(f"{prefix}-*"): # the folders have name like 'checkpoint-15'
        if str(path) in ignored: continue
        regex_match = re.match(f".*{prefix}-([0-9]+)", str(path))
        assert regex_match is not None and regex_match.groups() is not None
        checkpoints.append((int(regex_match.groups()[0]), str(path)))

    return [c[1] for c in sorted(checkpoints)]