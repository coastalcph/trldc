import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_curve, auc


'''[2022-Mar-30] https://github.com/jamesmullenbach/caml-mimic/blob/master/evaluation.py#L115
Number of gold labels in top k predictions / Number of gold labels'''
def recall_at_k(all_logits, all_golds, k=5):
    topk_logits = np.argsort(all_logits)[:, ::-1][:, :k]
    values = []
    for i, pred in enumerate(topk_logits):
        correct = all_golds[i, pred].sum()
        values.append(correct / float(len(all_golds[i, :].sum())))

    values = np.array(values)
    values[np.isnan(values)] = 0
    return {f"recall@{k}": np.mean(values)}


'''[2022-Mar-30] https://github.com/jamesmullenbach/caml-mimic/blob/master/evaluation.py#L132
Number of gold labels in top k predictions / k'''
def precision_at_k(all_logits, all_golds, k=5):
    topk_logits = np.argsort(all_logits)[:, ::-1][:, :k]
    values = []
    for i, pred in enumerate(topk_logits):
        if len(pred) > 0:
            correct = all_golds[i, pred].sum()
            values.append(correct / float(len(pred)))
    return {f"precision@{k}": np.mean(values)}


'''[2022-Mar-30] https://github.com/jamesmullenbach/caml-mimic/blob/master/evaluation.py#L169'''
def auc_metrics(all_logits, all_golds):
    all_fps, all_tps, all_auc = {}, {}, {}
    labels = []
    # get AUC for each label individually
    for i in range(all_golds.shape[1]):
        # only if there are true positives for this label
        if all_golds[:, i].sum() > 0:
            all_fps[i], all_tps[i], _ = roc_curve(all_golds[:, i], all_logits[:, i])
            if len(all_fps[i]) > 1 and len(all_tps[i]) > 1:
                auc_score = auc(all_fps[i], all_tps[i])
                if not np.isnan(auc_score):
                    all_auc["auc_%d" % i] = auc_score
                    labels.append(i)

    metrics = {}
    fp, tp, _ = roc_curve(all_golds.ravel(), all_logits.ravel())
    metrics["micro_auc"] = auc(fp, tp)
    metrics["macro_auc"] = np.mean([all_auc["auc_%d" % i] for i in labels])
    return metrics


'''[20220330]'''
def calculate_f1(true_positive, false_positive=None, num_predicted=None, false_negative=None, num_gold=None):
    assert num_predicted is not None or false_positive is not None
    num_predicted = num_predicted if num_predicted is not None else true_positive + false_positive
    assert num_gold is not None or false_negative is not None
    num_gold = num_gold if num_gold is not None else true_positive + false_negative

    precision = true_positive / num_predicted if num_predicted > 0 else 0.0
    recall = true_positive / num_gold if num_gold > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "num_predicted": num_predicted, "num_gold": num_gold}


'''[20220330]'''
class F_Score:
    def __init__(self):
        self._tps = defaultdict(int)
        self._fps = defaultdict(int)
        self._tns = defaultdict(int)
        self._fns = defaultdict(int)

    def add_tp(self, class_name):
        self._tps[class_name] += 1

    def add_fp(self, class_name):
        self._fps[class_name] += 1

    def add_tn(self, class_name):
        self._tns[class_name] += 1

    def add_fn(self, class_name):
        self._fns[class_name] += 1

    def get_tp(self, class_name=None):
        return sum(self._tps.values()) if class_name is None else self._tps[class_name]

    def get_fp(self, class_name=None):
        return sum(self._fps.values()) if class_name is None else self._fps[class_name]

    def get_tn(self, class_name=None):
        return sum(self._tns.values()) if class_name is None else self._tns[class_name]

    def get_fn(self, class_name=None):
        return sum(self._fns.values()) if class_name is None else self._fns[class_name]

    def _get_scores(self, class_name=None):
        tp = self.get_tp(class_name)
        fp = self.get_fp(class_name)
        fn = self.get_fn(class_name)
        return calculate_f1(tp, false_positive=fp, false_negative=fn)

    def get_accuracy(self):
        tp = self.get_tp()
        fp = self.get_fp()
        fn = self.get_fn()
        assert fp == fn and tp + fp > 0
        return {"accuracy": tp / (tp + fp)}

    def get_micro_avg_scores(self):
        return self._get_scores()

    def get_macro_avg_scores(self):
        scores = [self._get_scores(c) for c in self.get_classes()]
        precision = sum([s["precision"] for s in scores]) / len(scores)
        recall = sum([s["recall"] for s in scores]) / len(scores)
        f1 = sum([s["f1"] for s in scores]) / len(scores)
        return {"precision": precision, "recall": recall, "f1": f1}

    def get_detailed_scores(self):
        metric = {f"micro_{k}": v for k, v in self.get_micro_avg_scores().items()}
        metric.update({f"macro_{k}": v for k, v in self.get_macro_avg_scores().items()})
        for c in self.get_classes():
            assert c not in ["micro", "macro"]
            metric.update({f"CLASS_[{c}]_{k}": v for k, v in self._get_scores(c).items()})
        return metric

    def get_classes(self):
        all_classes = list(self._tps.keys()) + list(self._fps.keys()) + list(self._tns.keys()) + list(self._fns.keys())
        return sorted([c for c in set(all_classes) if c is not None])