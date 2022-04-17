import logging, numpy
from dainlp.metrics.utils import F_Score, auc_metrics, precision_at_k


logger = logging.getLogger(__name__)


'''[20220330]'''
class Metric:
    def __init__(self, idx2label=None, task_name="singlelabel"):
        self.task_name = task_name
        self.idx2label = idx2label

    def __call__(self, all_logits, all_golds):
        pred_labels = Metric.get_labels_from_logitis(all_logits, self.idx2label, self.task_name)

        if self.task_name == "singlelabel":
            gold_labels = [self.idx2label[g] for g in all_golds.tolist()]
            assert len(gold_labels) == len(pred_labels) and len(gold_labels) > 0

            f_score = F_Score()
            for g, p in zip(gold_labels, pred_labels):
                if g == p:
                    f_score.add_tp(g)
                else:
                    f_score.add_fp(p)
                    f_score.add_fn(g)

            metrics = f_score.get_accuracy()
            metrics.update({f"micro_{k}": v for k, v in f_score.get_micro_avg_scores().items()})
            metrics.update({f"macro_{k}": v for k, v in f_score.get_macro_avg_scores().items()})

            logger.info(metrics)
            return metrics
        elif self.task_name == "multilabel":
            gold_labels = Metric.get_multilabels_from_logitis(all_golds, self.idx2label)

            f_score = F_Score()
            for preds, golds in zip(pred_labels, gold_labels):
                for p in preds:
                    if p in golds:
                        f_score.add_tp(p)
                    else:
                        f_score.add_fp(p)
                for g in golds:
                    if g not in preds:
                        f_score.add_fn(g)

            metrics = {}
            metrics.update({f"macro_{k}": v for k, v in f_score.get_macro_avg_scores().items()})
            metrics.update({f"micro_{k}": v for k, v in f_score.get_micro_avg_scores().items()})
            metrics.update(precision_at_k(all_logits, all_golds, k=5))
            metrics.update(precision_at_k(all_logits, all_golds, k=8))
            metrics.update(precision_at_k(all_logits, all_golds, k=15))
            metrics.update(auc_metrics(all_logits, all_golds))

            logger.info(metrics)
            return metrics

    @staticmethod
    def get_labels_from_logitis(all_logits, idx2label, task_name="singlelabel"):
        if task_name == "multilabel":
            return Metric.get_multilabels_from_logitis(all_logits, idx2label)
        elif task_name == "singlelabel":
            preds = numpy.argmax(all_logits, axis=-1).tolist()
            return [idx2label[i] for i in preds]
        raise NotImplementedError

    @staticmethod
    def get_multilabels_from_logitis(all_logits, idx2label):
        all_labels = []
        for logits in numpy.where(all_logits > 0, 1, 0):
            preds = []
            for i, l in enumerate(logits):
                if l == 1:
                    preds.append(idx2label[i])
            all_labels.append(preds)
        return all_labels