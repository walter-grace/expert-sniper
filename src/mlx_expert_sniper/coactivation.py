"""Co-activation tracker for predictive expert prefetch."""
from collections import defaultdict, Counter


class CoActivationTracker:
    """Track which experts fire together for cross-layer prediction."""

    def __init__(self, num_layers, warmup_tokens=3):
        self.num_layers = num_layers
        self.warmup_tokens = warmup_tokens
        self.token_count = 0
        # cross_layer[(layer_n, eid)] = Counter of experts at layer_n+1
        self.cross_layer = defaultdict(Counter)
        self.prev_layer_experts = {}  # layer_idx -> set
        self.ready = False
        # stats
        self.predictions_made = 0
        self.predictions_correct = 0

    def record_layer(self, layer_idx, active_expert_ids):
        active = set(active_expert_ids)
        if layer_idx > 0 and (layer_idx - 1) in self.prev_layer_experts:
            prev = self.prev_layer_experts[layer_idx - 1]
            for prev_eid in prev:
                for cur_eid in active:
                    self.cross_layer[(layer_idx - 1, prev_eid)][cur_eid] += 1
        self.prev_layer_experts[layer_idx] = active

    def end_token(self):
        self.prev_layer_experts = {}
        self.token_count += 1
        if self.token_count >= self.warmup_tokens and not self.ready:
            self.ready = True

    def predict_next_layer(self, layer_idx, active_ids, top_k=6):
        if not self.ready or layer_idx + 1 >= self.num_layers:
            return []
        candidates = Counter()
        for eid in active_ids:
            candidates.update(self.cross_layer[(layer_idx, eid)])
        # Remove already-active (they'll be in cache from this layer anyway)
        for eid in active_ids:
            candidates.pop(eid, None)
        return [eid for eid, _ in candidates.most_common(top_k)]

    def score_prediction(self, predicted, actual):
        if not predicted:
            return
        pred_set = set(predicted)
        actual_set = set(actual)
        hits = len(pred_set & actual_set)
        self.predictions_made += len(predicted)
        self.predictions_correct += hits

    @property
    def accuracy(self):
        if self.predictions_made == 0:
            return 0.0
        return self.predictions_correct / self.predictions_made
