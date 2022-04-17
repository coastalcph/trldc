import datetime, json, logging, numpy, os, sys, time
from collections import defaultdict
from dainlp.utils.files import make_sure_parent_dir_exists


logger = logging.getLogger(__name__)


'''[20220330] 11111 -> 3:05:11.00'''
def print_seconds(seconds):
    msec = int(abs(seconds - int(seconds)) * 100)
    return f"{datetime.timedelta(seconds=int(seconds))}.{msec:02d}"


'''[20220330]'''
def print_mean_std(values, latex=False, std=True):
    if std:
        if latex:
            return f"{numpy.mean(values):.1f} \\tiny $\\pm$ {numpy.std(values):.1f}"
        else:
            return f"{numpy.mean(values):.1f} ({numpy.std(values):.1f})"
    else:
        return f"{numpy.mean(values):.1f}"


'''[20220330]'''
def print_delta(values, baselines):
    return f"{numpy.mean(values) - numpy.mean(baselines):.1f}"


'''[20220330]'''
def analyse_dev_and_test_results(train_metric_files, test_metric_files, main_metric_name, hp_names):
    all_dev_results, all_test_results = defaultdict(list), defaultdict(list)
    if test_metric_files is not None:
        assert len(train_metric_files) == len(test_metric_files)

    for i, f in enumerate(train_metric_files):
        train_metric = json.load(open(f))
        hp_values = tuple([train_metric["args"][n] for n in hp_names])
        all_dev_results[hp_values].append(train_metric["dev_metrics"])
        if test_metric_files is not None and os.path.exists(test_metric_files[i]):
            all_test_results[hp_values].append(json.load(open(test_metric_files[i])))

    assert len(all_dev_results) > 0
    sorted_dev_results = sorted(all_dev_results.items(),
                                key=lambda kv: numpy.mean([v[main_metric_name] for v in kv[1]]), reverse=True)
    best_hp = sorted_dev_results[0][0]
    best_dev_results = all_dev_results[best_hp]
    test_results = None if test_metric_files is None else all_test_results[best_hp]
    return {"best_dev_results": best_dev_results, "test_results": test_results,
            "best_hp": best_hp, "all_dev_results": all_dev_results, "all_test_results": all_test_results}


# [2021-Apr-20] https://github.com/huggingface/transformers/blob/v4.5.0/src/transformers/trainer_pt_utils.py#L641
def log_metrics(split, metrics):
    logger.info(f"***** {split} metrics *****")
    metrics_formatted = metrics.copy()
    for k, v in metrics_formatted.items():
        if "_memory_" in k:
            metrics_formatted[k] = f"{v >> 20}MB"
        elif k.endswith("_runtime"):
            metrics_formatted[k] = print_seconds(v)
        elif k == "total_flos":
            metrics_formatted[k] = f"{int(v) >> 30}GF"
        elif isinstance(metrics_formatted[k], float):
            metrics_formatted[k] = round(v, 4)
    k_width = max(len(str(x)) for x in metrics_formatted.keys())
    v_width = max(len(str(x)) for x in metrics_formatted.values())
    for k in sorted(metrics_formatted.keys()):
        logger.info(f"  {k: <{k_width}} = {metrics_formatted[k]:>{v_width}}")


# [2021-Jun-09]
def print_large_integer(number, suffix=None):
    if suffix is None:
        if number < 1e3: return f"{number}"
        str_number = str(number)
        if number < 1e6: return f"{str_number[:-3]},{str_number[-3:]}"
        if number < 1e9: return f"{str_number[:-6]},{str_number[-6:-3]},{str_number[-3:]}"
        raise ValueError(f"Maybe not a good idea to display such a large number ({number}) in this way")
    else:
        if suffix == "B": return f"{float(number)/1e9:.1f}B"
        if suffix == "M": return f"{float(number)/1e6:.1f}M"
        if suffix == "K": return f"{float(number)/1e3:.1f}K"

        if number < 1e3: return f"{number}"
        if number < 1e6: return f"{float(number)/1e3:.1f}K"
        if number < 1e9: return f"{float(number)/1e6:.1f}M"
        return f"{float(number)/1e9:.1f}B"


'''[2022-Feb-17]'''
def print_memory_size(size):
    size = size >> 10
    if size < 1024: return f"{size:.1f}K"
    size = size >> 10
    if size < 1024: return f"{size:.1f}M"
    size = size >> 10
    return f"{size:.1f}G"


# [2021-Sep-17]
def set_logging_format(log_filepath=None, debug=False):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_filepath is not None:
        make_sure_parent_dir_exists(log_filepath)
        handlers.append(logging.FileHandler(filename=log_filepath))

    if debug:
        logging.basicConfig(format="%(asctime)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                            level=logging.DEBUG, handlers=handlers)
    else:
        logging.basicConfig(format="%(message)s", level=logging.INFO, handlers=handlers)


# [2021-Mar-31]
def estimate_remaining_time(current_step, total_step, start_time):
    ratio = float(current_step) / total_step
    elapsed = time.time() - start_time
    if current_step == 0: return 0, elapsed, 0

    remaining = elapsed * (1 - ratio) / ratio
    return ratio, elapsed, remaining


# [2021-Mar-31]
def log_remaining_time(current_step, total_step, start_time, prefix="", suffix=""):
    ratio, elapsed, remaining = estimate_remaining_time(current_step, total_step, start_time)
    logger.info(f"{prefix}Progress: {current_step}/{total_step} ({ratio * 100:.1f}%); "
                f"Elapsed: {print_seconds(elapsed)}; Estimated remaining: {print_seconds(remaining)}{suffix}")


'''[2022-Mar-10] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer_utils.py#L257'''
def speed_metrics(split, start_time, num_samples=None, num_steps=None):
    runtime = time.time() - start_time
    result = {f"{split}_runtime": round(runtime, 4)}
    if num_samples is not None:
        result[f"{split}_samples_per_second"] = round(num_samples / runtime, 3)
    if num_steps is not None:
        result[f"{split}_steps_per_second"] = round(num_steps / runtime, 3)
    return result