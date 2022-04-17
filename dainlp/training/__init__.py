import collections, logging, math, os, time, torch
from dainlp.training.optimizer import create_optimizer
from dainlp.training.callback import DefaultFlowCallback, CallbackHandler, TrainerState, TrainerControl
from dainlp.training.utils import get_eval_dataloader, get_train_dataloader, training_step, prediction_step
from dainlp.training.utils import load_state_dict_in_model, save_checkpoint, wrap_model
from dainlp.training.callback import TrainerState
from dainlp.training.scheduler import create_scheduler
from dainlp.utils import set_seed
from dainlp.utils.files import move_best_checkpoint
from dainlp.utils.tensors import nested_gather, nested_truncate, nested_numpify, nested_concat
from dainlp.utils.tensors import distributed_broadcast_scalars, denumpify_detensorize, pad_across_processes
from dainlp.utils.print import print_large_integer, speed_metrics, log_remaining_time
from dainlp.utils.resources import MemoryTracker


logger = logging.getLogger(__name__)


'''[2022-Mar-10] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer.py#L275'''
class Trainer:
    def __init__(self, model, args, data_collator, train_dataset=None, eval_dataset=None, tokenizer=None,
                 compute_metrics=None, callbacks=None):
        self.args = args
        set_seed(args.seed)
        self.is_in_train = False
        self._memory_tracker = MemoryTracker(args.skip_memory_metrics)
        self._memory_tracker.start()
        args._setup_devices
        self.is_model_parallel = False
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        model = model.to(args.device)
        self.model_wrapped = model
        self.model = model
        self.compute_metrics = compute_metrics
        self.optimizer = None
        self.lr_scheduler = None
        callbacks = [DefaultFlowCallback] if callbacks is None else [DefaultFlowCallback] + callbacks
        self.callback_handler = CallbackHandler(callbacks, self.model, self.tokenizer, self.optimizer,
                                                self.lr_scheduler)
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)
        self.use_amp = True
        self.amp_type = torch.float16
        self.do_grad_scaling = True
        self.scaler = torch.cuda.amp.GradScaler()
        self.current_flos = 0
        self.state = TrainerState()
        self.control = TrainerControl()
        self.control = self.callback_handler.on_init_end(self.args, self.state, self.control)
        self._memory_tracker.stop_and_update_metrics()


    '''[2022-Mar-10] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer.py#L1060'''
    def train(self):
        model, start_time, num_train_epochs, num_train_samples, train_dataloader = self.before_training()

        for epoch in range(num_train_epochs):
            if epoch > 0: log_remaining_time(epoch, num_train_epochs, start_time, prefix="Epoch ")
            if isinstance(train_dataloader, torch.utils.data.DataLoader) and isinstance(train_dataloader.sampler,
                                                                                        torch.utils.data.distributed.DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            self.control = self.callback_handler.on_epoch_begin(self.args, self.state, self.control)

            self.train_loop(train_dataloader, model, epoch)

            if self.control.should_epoch_stop or self.control.should_training_stop:
                break

            self.control = self.callback_handler.on_epoch_end(self.args, self.state, self.control)
            self._maybe_log_save_evaluate(self.total_train_loss, model)
            if self.control.should_training_stop:
                break

        if self.args.should_log:
            logger.info("\n\nTraining completed.\n\n")

        metrics = self.after_training(start_time, num_train_samples)
        return metrics


    '''[2022-Mar-16]'''
    def before_training(self):
        self._memory_tracker.start()
        self.is_in_train = True
        assert isinstance(self.train_dataset, collections.abc.Sized)
        train_dataloader = get_train_dataloader(self.train_dataset, self.data_collator, self.args)
        total_train_batch_size = self.args.train_batch_size * self.args.gradient_accumulation_steps * self.args.world_size
        num_update_steps_per_epoch = max(1, len(train_dataloader) // self.args.gradient_accumulation_steps)
        if self.args.max_steps > 0:
            max_steps = self.args.max_steps
            num_train_epochs = self.args.max_steps // num_update_steps_per_epoch
            num_train_epochs += int(self.args.max_steps % num_update_steps_per_epoch > 0)
            num_train_samples = self.args.max_steps * total_train_batch_size
        else:
            max_steps = math.ceil(self.args.num_train_epochs * num_update_steps_per_epoch)
            num_train_epochs = math.ceil(self.args.num_train_epochs)
            num_train_samples = len(self.train_dataset) * self.args.num_train_epochs
        self.optimizer = create_optimizer(self.model, self.args)
        self.lr_scheduler = create_scheduler(self.args.lr_scheduler_type, self.optimizer, max_steps,
                                             self.args.warmup_steps, self.args.warmup_ratio)
        self.state = TrainerState()
        model = wrap_model(self.model_wrapped, self.args, training=True)
        if model is not self.model:
            self.model_wrapped = model

        num_examples = len(train_dataloader.dataset)
        if self.args.should_log:
            logger.info(f"Training on {num_examples} examples for {num_train_epochs} epochs ({max_steps}) steps; "
                        f"Total batch size: {total_train_batch_size} = {self.args.per_device_train_batch_size} "
                        f"x {self.args.n_gpu} x {self.args.world_size} x {self.args.gradient_accumulation_steps}")
            logger.info(f"The model has {print_large_integer(self.model.num_parameters())} parameters, "
                        f"({print_large_integer(self.model.num_parameters(True))} trainable)")
        self.state.epoch = 0
        start_time = time.time()
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = (self.args.local_process_index == 0)
        self.state.is_world_process_zero = (self.args.process_index == 0)
        self.state.log_history.append({"num_parameters": self.model.num_parameters(),
                                       "num_trainable_parameters": self.model.num_parameters(True)})
        self.total_train_loss = torch.tensor(0.0).to(self.args.device)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        self.control = self.callback_handler.on_train_begin(self.args, self.state, self.control)
        return model, start_time, num_train_epochs, num_train_samples, train_dataloader


    '''[2022-Mar-10]'''
    def train_loop(self, epoch_iterator, model, epoch):
        for step, inputs in enumerate(epoch_iterator):
            if step % self.args.gradient_accumulation_steps == 0:
                self.control = self.callback_handler.on_step_begin(self.args, self.state, self.control)

            if (((step + 1) % self.args.gradient_accumulation_steps != 0) and self.args.local_rank != -1):
                with model.no_sync():
                    tr_loss_step = training_step(model, inputs, self.scaler, self.args)
            else:
                tr_loss_step = training_step(model, inputs, self.scaler, self.args)

            if torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step):
                self.total_train_loss += self.total_train_loss / (1 + self.state.global_step - self._globalstep_last_logged)
            else:
                self.total_train_loss += tr_loss_step

            self.current_flos += float(
                self.model.floating_point_ops(inputs) if hasattr(self.model, "floating_point_ops") and "input_ids" in inputs else 0)

            if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    (step + 1 == len(epoch_iterator)) and len(epoch_iterator) <= self.args.gradient_accumulation_steps):
                if self.args.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    # TODO: there is a warning message
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                scale_after = self.scaler.get_scale()
                if scale_before <= scale_after:
                    self.lr_scheduler.step()

                model.zero_grad()
                self.state.global_step += 1
                self.state.epoch = epoch + (step + 1) / len(epoch_iterator)
                self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)
                self._maybe_log_save_evaluate(self.total_train_loss, model)
            else:
                # TODO: on_substep_end
                pass

            if self.control.should_epoch_stop or self.control.should_training_stop:
                break


    '''[2022-Mar-10]'''
    def after_training(self, start_time, num_train_samples):
        if self.state.best_model_checkpoint is not None:
            if self.args.local_rank != -1:
                torch.distributed.barrier()

            if self.args.load_best_model_at_end:
                logger.info(f"Loading best model: {self.state.best_model_checkpoint}, score {self.state.best_metric}).")
                best_model_path = os.path.join(self.state.best_model_checkpoint, "pytorch_model.bin")
                state_dict = torch.load(best_model_path, map_location="cpu")
                load_state_dict_in_model(self.model, state_dict)

        self._total_loss_scalar += self.total_train_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step
        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["loss"] = train_loss
        self.is_in_train = False
        self._memory_tracker.stop_and_update_metrics(metrics)
        self.log(metrics)
        self.control = self.callback_handler.on_train_end(self.args, self.state, self.control)

        if self.args.local_process_index == 0:
            move_best_checkpoint(self.args.output_dir, self.state.best_model_checkpoint)


    '''[2022-Mar-10] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer.py#L2111'''
    def store_flos(self):
        if self.args.local_rank != -1:
            self.state.total_flos += (distributed_broadcast_scalars([self.current_flos], self.args.device).sum().item())
            self.current_flos = 0
        else:
            self.state.total_flos += self.current_flos
            self.current_flos = 0


    '''[2022-Mar-10] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer.py#L2299'''
    def evaluation_loop(self, dataloader, prefix="eval"):
        model = wrap_model(self.model, self.args, training=False)
        batch_size = dataloader.batch_size
        if self.args.should_log:
            logger.info(f"\tEvaluate on {prefix} set: {len(dataloader.dataset)} examples; batch size: {batch_size}")
        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        gpu_losses, gpu_logits, gpu_golds = None, None, None
        for step, inputs in enumerate(dataloader):
            outputs = prediction_step(model, inputs, self.args.device)
            losses = nested_gather(outputs["loss"].repeat(batch_size), self.args.local_rank)
            gpu_losses = losses if gpu_losses is None else torch.cat((gpu_losses, losses), dim=0)

            if "logits" in outputs:
                logits = pad_across_processes(outputs["logits"])
                logits = nested_gather(logits, self.args.local_rank)
                gpu_logits = logits if gpu_logits is None else nested_concat(gpu_logits, logits, padding_index=-100)
                golds = pad_across_processes(outputs["golds"])
                golds = nested_gather(golds, self.args.local_rank)
                gpu_golds = golds if gpu_golds is None else nested_concat(gpu_golds, golds, padding_index=-100)

            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)
            # TODO: the whole predictions are now on GPU (faster but requires more memory)

        num_samples = len(dataloader.dataset)
        cpu_losses = nested_numpify(gpu_losses)[:num_samples]
        cpu_logits = nested_truncate(nested_numpify(gpu_logits), num_samples)
        cpu_golds = nested_truncate(nested_numpify(gpu_golds), num_samples)

        metrics = self.compute_metrics(cpu_logits, cpu_golds)
        metrics = denumpify_detensorize(metrics)

        metrics[f"{prefix}_loss"] = cpu_losses.mean().item()
        for k in list(metrics.keys()):
            if not k.startswith(f"{prefix}_"):
                metrics[f"{prefix}_{k}"] = metrics.pop(k)

        return {"logits": cpu_logits, "metrics": metrics, "golds": cpu_golds}


    '''[2022-Mar-10] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer.py#L2171'''
    def evaluate(self, eval_dataset, prefix="eval"):
        metrics = self.predict(eval_dataset, prefix)["metrics"]
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics


    '''[2022-Mar-10] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer.py#L2240'''
    def predict(self, test_dataset, metric_key_prefix="test"):
        self._memory_tracker.start()
        test_dataloader = get_eval_dataloader(test_dataset, self.data_collator, self.args)
        start_time = time.time()
        outputs = self.evaluation_loop(test_dataloader, prefix=metric_key_prefix)
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        outputs["metrics"].update(speed_metrics(metric_key_prefix, start_time, num_samples=len(test_dataset),
                                                num_steps=math.ceil(len(test_dataset) / total_batch_size)))
        self._memory_tracker.stop_and_update_metrics(outputs["metrics"])
        return outputs


    '''[2022-Mar-10] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer.py#L1852'''
    def log(self, logs):
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)
        self.state.log_history.append({**logs, **{"step": self.state.global_step}})
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)


    '''[2022-Mar-10] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer.py#L1541
                     https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer_pt_utils.py#L797'''
    def _maybe_log_save_evaluate(self, loss, model):
        if self.control.should_log:
            logs = {}
            loss_scalar = nested_gather(loss, self.args.local_rank).mean().item()
            loss -= loss
            logs["loss"] = loss_scalar / (self.state.global_step - self._globalstep_last_logged)
            logs["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
            self._total_loss_scalar += loss_scalar
            self._globalstep_last_logged = self.state.global_step
            # TODO self.store_flos()
            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(self.eval_dataset)
        if self.control.should_save:
            save_checkpoint(model, self.tokenizer, self.args, self.state, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)