import gc, inspect, logging, psutil, threading, torch
from dainlp.utils.print import print_memory_size


logger = logging.getLogger(__name__)


'''[2022-Feb-17] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer_utils.py#L290'''
class MemoryTracker:
    stages = {"__init__": "init", "train": "train", "evaluate": "eval", "predict": "test"}

    def __init__(self, skip_memory_metrics=False):
        self.skip_memory_metrics = skip_memory_metrics
        if skip_memory_metrics: return

        self.gpu = {}
        self.cpu = {}
        self.process = psutil.Process()
        self.cur_stage = None
        self.init_reported = False

    def derive_stage(self):
        caller = inspect.currentframe().f_back.f_back.f_code.co_name
        assert caller in self.stages
        return self.stages[caller]

    def cpu_mem_used(self):
        return self.process.memory_info().rss

    def peak_monitor_function(self):
        self.cpu_memory_used_peak = -1

        while True:
            self.cpu_memory_used_peak = max(self.cpu_mem_used(), self.cpu_memory_used_peak)

            # can't sleep or will not catch the peak right (this comment is here on purpose)
            # time.sleep(0.001) # 1msec

            if not self.peak_monitoring:
                break

    def start(self):
        if self.skip_memory_metrics: return

        stage = self.derive_stage()
        if self.cur_stage is not None and self.cur_stage != stage: return

        self.cur_stage = stage
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        self.gpu_memory_used_at_start = torch.cuda.memory_allocated()
        self.cpu_memory_used_at_start = self.cpu_mem_used()

        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_function)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()

    def stop(self, stage):
        if self.cur_stage is not None and self.cur_stage != stage: return

        self.peak_monitoring = False
        gc.collect()
        torch.cuda.empty_cache()
        self.gpu_memory_used_now = torch.cuda.memory_allocated()
        self.gpu_memory_used_peak = torch.cuda.max_memory_allocated()
        self.gpu[self.cur_stage] = dict(begin=self.gpu_memory_used_at_start, end=self.gpu_memory_used_now,
                                        alloc=(self.gpu_memory_used_now - self.gpu_memory_used_at_start),
                                        peaked=max(0, self.gpu_memory_used_peak - self.gpu_memory_used_now))
        self.cpu_memory_used_now = self.cpu_mem_used()
        self.cpu[self.cur_stage] = dict(begin=self.cpu_memory_used_at_start, end=self.cpu_memory_used_now,
                                        alloc=(self.cpu_memory_used_now - self.cpu_memory_used_at_start),
                                        peaked=max(0, self.cpu_memory_used_peak - self.cpu_memory_used_now))
        self.cur_stage = None

    def update_metrics(self, stage, metrics):
        if self.skip_memory_metrics: return
        if self.cur_stage is not None and self.cur_stage != stage: return

        stages = [stage]
        if not self.init_reported:
            stages.insert(0, "init")
            self.init_reported = True

        for stage in stages:
            for k in ["alloc", "peaked"]:
                if stage in self.cpu and k in self.cpu[stage]:
                    metrics[f"{stage}_memory_cpu_{k}_delta"] = self.cpu[stage][k]
                if stage in self.gpu and k in self.gpu[stage]:
                    metrics[f"{stage}_memory_gpu_{k}_delta"] = self.gpu[stage][k]

        if stages[0] == "init":
            metrics["before_init_memory_cpu"] = self.cpu["init"]["begin"]
            metrics["before_init_memory_gpu"] = self.gpu["init"]["begin"]

    def stop_and_update_metrics(self, metrics=None):
        if self.skip_memory_metrics: return
        stage = self.derive_stage()
        self.stop(stage)
        if metrics is not None:
            self.update_metrics(stage, metrics)