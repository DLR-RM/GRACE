from torchmetrics.retrieval.base import RetrievalMetric
from torchmetrics.utilities.data import get_group_indexes
from torchmetrics.utilities.checks import _check_retrieval_inputs

from typing import Any, Optional, List, Dict
from torch import Tensor, tensor
import torch
import numpy as np
from collections import defaultdict


class SequencePrecision(RetrievalMetric):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        k: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            **kwargs,
        )

        if (k is not None) and not (isinstance(k, int) and k > 0):
            raise ValueError("`k` has to be a positive integer or None")

        self.k = k

    def compute(self) -> Tensor:
        """First concat state ``indexes``, ``preds`` and ``target`` since they were stored as lists.

        After that, compute list of groups that will help in keeping together predictions about the same query. Finally,
        for each group compute the ``_metric`` if the number of positive targets is at least 1, otherwise behave as
        specified by ``self.empty_target_action``.

        !!!
        Diff to base implementation- Added a treatment for cases in which k is larger from target.
        !!!
        """
        indexes = torch.cat(self.indexes, dim=0)
        preds = torch.cat(self.preds, dim=0)
        target = torch.cat(self.target, dim=0)

        res = []
        groups = get_group_indexes(indexes)

        for group in groups:
            mini_preds = preds[group]
            mini_target = target[group]

            if not mini_target.sum():
                if self.empty_target_action == "error":
                    raise ValueError("`compute` method was provided with a query with no positive target.")
                if self.empty_target_action == "pos":
                    res.append(tensor(1.0))
                elif self.empty_target_action == "neg":
                    res.append(tensor(0.0))
            elif (self.k is not None) and self.k > mini_target.sum():
                # can't compute P@k when k > |REL|
                continue
            else:
                # ensure list contains only float tensors
                res.append(self._metric(mini_preds, mini_target))
        
        if res:
            out = torch.stack([x.to(preds) for x in res]).mean()
        else:
            out = tensor(0.0).to(preds) if self.k is None else tensor(-1).to(preds)

        return out
    
    def _metric(self, preds: Tensor, target: Tensor) -> Tensor:
        return _sequence_precision(preds, target, self.k)


class SequenceRecall(RetrievalMetric):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        k: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            **kwargs,
        )

        if (k is not None) and not (isinstance(k, int) and k > 0):
            raise ValueError("`k` has to be a positive integer or None")

        self.k = k

    def _metric(self, preds: Tensor, target: Tensor) -> Tensor:
        return _sequence_recall(preds, target, self.k)


def _sequence_precision(preds: Tensor, target: Tensor, k: Optional[int] = None):

    if not preds.sum():
        return tensor(1.0, device=preds.device)
    elif not target.sum():
        return tensor(0.0, device=preds.device)

    effective_k = preds.count_nonzero() if k is None else min(k, preds.count_nonzero())

    retrieved = target[preds.topk(effective_k, dim=-1)[1]].sum().float()

    return retrieved / effective_k


def _sequence_recall(preds: Tensor, target: Tensor, k: Optional[int] = None):
    if not preds.sum():
        return tensor(0.0, device=preds.device)
    elif not target.sum():
        return tensor(1.0, device=preds.device)
    

    effective_k = preds.count_nonzero() if k is None else min(k, preds.count_nonzero())
    retrieved = target[preds.topk(effective_k, dim=-1)[1]].sum().float()

    relevant = target.sum() if k is None else min(k, target.sum())
    
    return retrieved / relevant


class SequencePrecisionRecallCurve(RetrievalMetric):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    preds_seqs: List[List[float]]

    def __init__(
        self,
        k: Optional[int] = None,
        type: str = "min",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            **kwargs,
        )

        if type not in ["min", "prod"]:
            raise ValueError("`type` has to be one of [`min`, `prod`]")
            
        if (k is not None) and not (isinstance(k, int) and k > 0):
            raise ValueError("`k` has to be a positive integer or None")
        self.type = type
        self.k = k

        self.add_state("preds_seqs", default=[], dist_reduce_fx=None)

    def update(self, preds: Tensor, target: Tensor, indexes: Tensor, stat_dict: Dict) -> None:  # type: ignore
        """Check shape, check and convert dtypes, flatten and add to accumulators."""
        if indexes is None:
            raise ValueError("Argument `indexes` cannot be None")

        if "sequence_prob" not in stat_dict:
            raise ValueError("Argument `stat_dict` doesn't have sequence_prob key")

        indexes, preds, target = _check_retrieval_inputs(
            indexes, preds, target, allow_non_binary_target=self.allow_non_binary_target, ignore_index=self.ignore_index
        )

        if preds.shape[0] != len(stat_dict["sequence_prob"]):
            raise ValueError("`preds` and `preds_seqs` must have the same length")

        self.indexes.append(indexes)
        self.preds.append(preds)
        self.target.append(target)
        self.preds_seqs.extend(stat_dict["sequence_prob"])
    
    def compute(self) -> Tensor:
        """First concat state ``indexes``, ``preds`` and ``target`` since they were stored as lists.

        After that, compute list of groups that will help in keeping together predictions about the same query. Finally,
        for each group compute the ``_metric`` if the number of positive targets is at least 1, otherwise behave as
        specified by ``self.empty_target_action``.
        """
        indexes = torch.cat(self.indexes, dim=0)
        preds = torch.cat(self.preds, dim=0)
        target = torch.cat(self.target, dim=0)
        
        groups = get_group_indexes(indexes)

        precisions, recalls = list(), list()
        
        thresholds = np.linspace(0, 1, num=100, endpoint=False)
        for threshold in thresholds:
            pre, rec = list(), list()

            for group in groups:
                mini_preds = preds[group]
                mini_target = target[group]
                mini_preds_seqs = [self.preds_seqs[i] for i in group.numpy()]

                # zero predictions for which one of the steps in the sequnece is lower than threshold
                for i, seq in enumerate(mini_preds_seqs):
                    if len(seq) == 0:
                        continue
                    
                    val = min(seq) if self.type == "min" else np.prod(seq)
                    if val <= threshold:
                        mini_preds[i] = 0       

                if not mini_target.sum():
                    if self.empty_target_action == "error":
                        raise ValueError("`compute` method was provided with a query with no positive target.")
                    if self.empty_target_action == "pos":
                        pre.append(tensor(1.0))
                        rec.append(tensor(1.0))
                    elif self.empty_target_action == "neg":
                        pre.append(tensor(0.0))
                        rec.append(tensor(0.0))
                else:
                    # ensure list contains only float tensors
                    pre.append(_sequence_precision(mini_preds, mini_target, self.k))
                    rec.append(_sequence_recall(mini_preds, mini_target, self.k))

            precisions.append(torch.stack([x.to(preds) for x in pre]).mean().item() if pre else 0.0)
            recalls.append(torch.stack([x.to(preds) for x in rec]).mean().item() if rec else 0.0)

        return [0.] + precisions + [1.], [1.] + recalls + [0.], thresholds

    def _metric(self, preds: Tensor, target: Tensor) -> Tensor:
        """ 
        Workaround
        """
        pass


class SequenceStats():
    def __init__(self, mode="feasbile") -> None:
        '''
        mode can be either "feasbile" or "infeasbile"
        '''

        self.preds = list()
        self.preds_seqs = list()
        self.indexes = list()
        self.target = list()
        self.vars = list()

        self.mode = mode

    def update(self, preds: Tensor, target: Tensor, indexes: Tensor, stat_dict: Dict) -> None:
        self.indexes.append(indexes)
        self.preds.append(preds)
        self.target.append(target)
        
        self.preds_seqs.extend(stat_dict["sequence_prob"])
        # this holds the variance between repeated forward passes of the same step
        if "inter_var" in stat_dict:
            self.vars.extend(stat_dict["inter_var"])
       
    def compute(self) -> Tensor:
        indexes = torch.cat(self.indexes, dim=0)
        preds = torch.cat(self.preds, dim=0)
        target = torch.cat(self.target, dim=0)
        
        groups = get_group_indexes(indexes)

        min_prob, max_prob, mean_prob, var_prob, prod_prob = list(), list(), list(), list(), list()
        mc_dropout_var = list()
        step_prob = defaultdict(list)

        for group in groups:
            mini_preds = preds[group]
            mini_target = target[group]

            if not mini_preds.sum():
                continue
            
            mini_preds_seqs = [self.preds_seqs[i] for i in group.numpy()]

            top_seq_prod, top_seq_idx = mini_preds.max(dim=-1)
            top_seq_prod, top_seq_idx = top_seq_prod.item(), top_seq_idx.item()
            top_seq = np.array(mini_preds_seqs[top_seq_idx])

            if len(self.vars):
                mini_vars = [self.vars[i] for i in group.numpy()]
                top_seq_vars = np.array(mini_vars[top_seq_idx])
            
            if (mini_target.sum() and mini_target[top_seq_idx].item() and self.mode == "feasbile") or \
                (not mini_target.sum() and self.mode == "infeasbile"):

                prod_prob.append(top_seq_prod)
                min_prob.append(top_seq.min())
                max_prob.append(top_seq.max())
                mean_prob.append(top_seq.mean())
                var_prob.append(top_seq.var())

                if len(self.vars):
                    mc_dropout_var.append(max(top_seq_vars))

                for i, p in enumerate(top_seq):
                    step_prob[i].append(p)
        '''
        stats = {
            "score": min_prob
        }
        '''
        stats = {
            "min": min_prob, "max": max_prob, "mean": mean_prob, "var": var_prob, "prod": prod_prob
        }
        
        '''
        for i, v in step_prob.items():
            stats["step " + str(i)] = v
        '''

        var_stats = dict()
        
        if len(self.vars): 
            var_stats["Inter-Step Max Variance (MC)"] = mc_dropout_var

        return stats, var_stats

