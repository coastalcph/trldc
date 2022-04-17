import dataclasses, json, logging, time
from dataclasses import dataclass
from typing import List
from dainlp.utils.print import log_remaining_time
from transformers.file_utils import ExplicitEnum


logger = logging.getLogger(__name__)


'''[TODO] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer_utils.py#L115'''
class IntervalStrategy(ExplicitEnum):
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


'''[2022-Mar-11] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer_callback.py#L36'''
@dataclass
class TrainerState:
    epoch: float = None
    global_step: int = 0
    max_steps: int = 0
    num_train_epochs: int = 0
    total_flos: float = 0
    log_history: List[str] = None
    best_metric: float = None
    best_model_checkpoint: str = None
    is_local_process_zero = True
    is_world_process_zero = True
    is_hyper_param_search = False
    trial_name = None
    trial_params = None
    start_time: float = None

    def __post_init__(self):
        self.start_time = time.time()
        if self.log_history is None:
            self.log_history = []

    def save_to_json(self, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True) + "\n")


'''[2022-Mar-11] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer_callback.py#L111'''
@dataclass
class TrainerControl:
    should_training_stop = False
    should_epoch_stop = False
    should_save = False
    should_evaluate = False
    should_log = False

    def _new_training(self):
        self.should_training_stop = False

    def _new_epoch(self):
        self.should_epoch_stop = False

    def _new_step(self):
        self.should_save = False
        self.should_evaluate = False
        self.should_log = False


'''[2022-Mar-11] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer_callback.py#L160'''
class TrainerCallback:
    def on_init_end(self, args, state, control, **kwargs):
        pass

    def on_train_begin(self, args, state, control, **kwargs):
        pass

    def on_train_end(self, args, state, control, **kwargs):
        pass

    def on_epoch_begin(self, args, state, control, **kwargs):
        pass

    def on_epoch_end(self, args, state, control, **kwargs):
        pass

    def on_step_begin(self, args, state, control, **kwargs):
        pass

    def on_step_end(self, args, state, control, **kwargs):
        pass

    def on_substep_end(self, args, state, control, **kwargs):
        pass

    def on_evaluate(self, args, state, control, **kwargs):
        pass

    def on_save(self, args, state, control, **kwargs):
        pass

    def on_log(self, args, state, control, **kwargs):
        pass

    def on_prediction_step(self, args, state, control, **kwargs):
        pass


'''[2022-Mar-11] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer_callback.py#L285'''
class CallbackHandler(TrainerCallback):
    def __init__(self, callbacks, model, tokenizer, optimizer, lr_scheduler):
        self.callbacks = []
        for cb in callbacks:
            self.add_callback(cb)
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = None
        self.eval_dataloader = None

    def add_callback(self, callback):
        cb = callback() if isinstance(callback, type) else callback
        cb_class = callback if isinstance(callback, type) else callback.__class__
        assert cb_class not in [c.__class__ for c in self.callbacks]
        self.callbacks.append(cb)

    def call_event(self, event, args, state, control, **kwargs):
        for callback in self.callbacks:
            result = getattr(callback, event)(args, state, control, model=self.model, tokenizer=self.tokenizer,
                                              optimizer=self.optimizer, lr_scheduler=self.lr_scheduler,
                                              train_dataloader=self.train_dataloader,
                                              eval_dataloader=self.eval_dataloader, **kwargs)
            if result is not None: control = result
        return control

    def on_init_end(self, args, state, control):
        return self.call_event("on_init_end", args, state, control)

    def on_train_begin(self, args, state, control):
        control.should_training_stop = False
        return self.call_event("on_train_begin", args, state, control)

    def on_train_end(self, args, state, control):
        return self.call_event("on_train_end", args, state, control)

    def on_epoch_begin(self, args, state, control):
        control.should_epoch_stop = False
        return self.call_event("on_epoch_begin", args, state, control)

    def on_epoch_end(self, args, state, control):
        return self.call_event("on_epoch_end", args, state, control)

    def on_step_begin(self, args, state, control):
        control.should_log = False
        control.should_evaluate = False
        control.should_save = False
        return self.call_event("on_step_begin", args, state, control)

    def on_step_end(self, args, state, control):
        return self.call_event("on_step_end", args, state, control)

    def on_substep_end(self, args, state, control):
        return self.call_event("on_subsetep_end", args, state, control)

    def on_evaluate(self, args, state, control, metrics):
        control.should_evaluate = False
        return self.call_event("on_evaluate", args, state, control, metrics=metrics)

    def on_save(self, args, state, control):
        control.should_save = False
        return self.call_event("on_save", args, state, control)

    def on_log(self, args, state, control, logs):
        control.should_log = False
        return self.call_event("on_log", args, state, control, logs=logs)

    def on_prediction_step(self, args, state, control):
        return self.call_event("on_prediction_step", args, state, control)


'''[2022-Mar-11] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer_callback.py#L407'''
class DefaultFlowCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            control.should_log = True
        if args.logging_strategy == IntervalStrategy.STEPS and args.logging_steps > 0 \
                and state.global_step % args.logging_steps == 0:
            control.should_log = True
        if args.evaluation_strategy == IntervalStrategy.STEPS and args.eval_steps > 0 \
                and state.global_step % args.eval_steps == 0:
            control.should_evaluate = True
            if args.load_best_model_at_end: control.should_save = True
        if args.save_strategy == IntervalStrategy.STEPS and args.save_steps > 0 \
                and state.global_step % args.save_steps == 0:
            control.should_save = True
        if state.global_step >= state.max_steps:
            control.should_training_stop = True
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        if args.logging_strategy == IntervalStrategy.EPOCH:
            control.should_log = True
        if args.evaluation_strategy == IntervalStrategy.EPOCH:
            control.should_evaluate = True
        if args.save_strategy == IntervalStrategy.EPOCH:
            control.should_save = True
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and args.should_log:
            suffix = f"; loss: {logs['loss']}" if "loss" in logs else ""
            suffix += f"; lr: {logs['learning_rate']}" if "learning_rate" in logs else ""
            log_remaining_time(state.global_step, state.max_steps, state.start_time, suffix=suffix)


'''[TODO] https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/trainer_callback.py#L505'''
class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, early_stopping_patience=5):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_patience_counter = 0

    def check_metric_value(self, args, state, metric_value):
        if state.best_metric is None:
            self.early_stopping_patience_counter = 0
        else:
            if args.greater_is_better:
                if metric_value > state.best_metric:
                    self.early_stopping_patience_counter = 0
                else:
                    self.early_stopping_patience_counter += 1
            else:
                if metric_value < state.best_metric:
                    self.early_stopping_patience_counter = 0
                else:
                    self.early_stopping_patience_counter += 1

    def on_train_begin(self, args, state, control, **kwargs):
        assert args.load_best_model_at_end and args.metric_for_best_model
        assert args.evaluation_strategy != IntervalStrategy.NO

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric_to_check = args.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics.get(metric_to_check, None)
        assert metric_value is not None
        if metric_value == 0.0:
            self.early_stopping_patience_counter = 0
            logger.info(f"{metric_to_check} is still 0.0")
        else:
            self.check_metric_value(args, state, metric_value)
            if self.early_stopping_patience_counter > 0:
                logger.info(f"No improvement since last {self.early_stopping_patience_counter} evaluations")
            else:
                logger.info(f"The best score: {metric_value}")
            if self.early_stopping_patience_counter >= self.early_stopping_patience:
                logger.info("Early stop !!!")
                control.should_training_stop = True