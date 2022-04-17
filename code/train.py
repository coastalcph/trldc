import dataclasses, logging, os, sys, time
sys.path.insert(0, "../")
import dainlp
from dainlp.utils.args import HfArgumentParser, ArgumentsForHiTransformer as Arguments
from dainlp.data.cls import Dataset
from dainlp.data.cls.hierarchical import Collator
from dainlp.models.cls.hierarchical import Model
from dainlp.metrics.cls import Metric
from dainlp.training import Trainer
from dainlp.training.callback import EarlyStoppingCallback
from dainlp.utils.print import print_seconds
from transformers import AutoTokenizer, AutoConfig


logger = logging.getLogger(__name__)


def parse_args():
    parser = HfArgumentParser([Arguments])
    args = parser.parse_args_into_dataclasses()[0]
    args._setup_devices
    dainlp.utils.print.set_logging_format(os.path.join(args.output_dir, "training.log"))
    dainlp.utils.set_seed(args.seed)
    if args.should_log:
        logger.info(f"DaiNLP {dainlp.__version__}")
        logger.info("**************************************************")
        logger.info("*             Parse the arguments                *")
        logger.info("**************************************************")
        logger.info(args)
    return args


def load_data(args):
    if args.should_log:
        logger.info("**************************************************")
        logger.info("*               Load the datasets                *")
        logger.info("**************************************************")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=True)
    train_dataset = Dataset(args.train_filepath, args, tokenizer, split="train")
    idx2label = {v: k for k, v in train_dataset.label2idx.items()}
    dev_dataset = Dataset(args.dev_filepath, args, tokenizer, split="dev", label2idx=train_dataset.label2idx)
    return tokenizer, train_dataset, dev_dataset, idx2label


def build_trainer(tokenizer, train_dataset, dev_dataset, args, idx2label):
    if args.should_log:
        logger.info("**************************************************")
        logger.info("*               Build the trainer                *")
        logger.info("**************************************************")
    config = AutoConfig.from_pretrained(args.model_dir, id2label=idx2label)
    config.task_name = args.task_name
    config.max_segments = args.max_num_segments
    config.do_use_label_wise_attention = args.do_use_label_wise_attention
    model = Model.from_pretrained(args.model_dir, config=config)
    data_collator = Collator(tokenizer, args.segment_length, args.max_num_segments,
                             args.do_use_stride, args.add_cls_each_segment, args.task_name)
    compute_metrics = Metric(idx2label, args.task_name)

    trainer = Trainer(model=model, args=args, data_collator=data_collator, train_dataset=train_dataset,
                      eval_dataset=dev_dataset, tokenizer=tokenizer, compute_metrics=compute_metrics,
                      callbacks=[EarlyStoppingCallback])
    return trainer


def main():
    args = parse_args()
    tokenizer, train_dataset, dev_dataset, idx2label = load_data(args)

    trainer = build_trainer(tokenizer, train_dataset, dev_dataset, args, idx2label)
    train_metrics = trainer.train()
    dev_metrics = trainer.predict(dev_dataset, metric_key_prefix="dev")["metrics"]

    if args.should_log:
        args.complete_running_time = print_seconds(time.time() - args.init_args_time)
        dainlp.utils.files.write_object_to_json_file(
            {"args": dataclasses.asdict(args), "training_state": dataclasses.asdict(trainer.state),
             "train_metrics": train_metrics, "dev_metrics": dev_metrics}, args.output_metrics_filepath, sort_keys=True)


if __name__ == "__main__":
    main()