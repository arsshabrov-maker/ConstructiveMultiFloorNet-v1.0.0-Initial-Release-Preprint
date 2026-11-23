import itertools
import operator
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F


class MicroConstructiveNet(nn.Module):
  """Compact micro-network operating on a selected subset of raw features and prior-floor blocks."""

  def __init__(self, col_indices, hidden_dim=8):
    super().__init__()
    self.register_buffer(
        'cols', torch.tensor(col_indices, dtype=torch.long)
    )
    in_d = len(col_indices)
    self.net = nn.Sequential(
        nn.Linear(in_d, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, 1),
    )

  def forward(self, pool_matrix):
    sub_x = pool_matrix[:, self.cols]
    return self.net(sub_x).squeeze(-1)


class ConstructiveMultiFloorNet(nn.Module):
  """
    Constructive Multi-Floor Neural Boosting (`ConstructiveMultiFloorNet`):
      1. Grows neural depth floor by floor via beam search over feature subsets.
      2. Preserves direct raw-feature passthrough at every new floor.
      3. Reweights hard misclassified examples after each floor (AdaBoost-style
         error focusing) so deeper floors specialize on difficult boundary cases.
      4. Aggregates the Top-K leaders from every floor (1-floor to N-floor)
         into a final L2-regularized multi-depth ensemble.
    """

  def __init__(
      self,
      n_floors=3,
      top_k_per_floor=8,
      beam_candidates=32,
      focus_power=2.5,
      seed=42,
  ):
    super().__init__()
    self.n_floors = n_floors
    self.top_k_per_floor = top_k_per_floor
    self.beam_candidates = beam_candidates
    self.focus_power = focus_power
    self.seed = seed

    self.floor_models = nn.ModuleList()
    self.meta_booster = None

  def _train_candidate(
      self,
      col_indices,
      pool_tr,
      y_tr_t,
      w_tr,
      pool_va,
      y_va_np,
      w_va_np,
      cand_seed,
      epochs=55,
  ):
    torch.manual_seed(cand_seed)
    device = pool_tr.device
    h_dim = min(12, max(6, len(col_indices) * 2))
    net = MicroConstructiveNet(col_indices, hidden_dim=h_dim).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=0.018, weight_decay=1e-3)

    for _ in range(epochs):
      net.train()
      opt.zero_grad()
      logits = net(pool_tr)
      raw_loss = F.binary_cross_entropy_with_logits(
          logits, y_tr_t, reduction='none'
      )
      loss = (raw_loss * w_tr).mean()
      loss.backward()
      opt.step()

    net.eval()
    with torch.no_grad():
      va_logits = net(pool_va)
      va_probs = torch.sigmoid(va_logits).cpu().numpy()
      va_score = roc_auc_score(y_va_np, va_probs, sample_weight=w_va_np)
    return va_score, va_logits, net

  def fit(self, X_tr, y_tr, X_va, y_va):
    np.random.seed(self.seed)
    device = X_tr.device
    n_raw_feats = X_tr.size(1)

    y_tr_t = torch.as_tensor(y_tr, dtype=torch.float32, device=device)
    y_va_t = torch.as_tensor(y_va, dtype=torch.float32, device=device)
    y_va_np = y_va_t.cpu().numpy()

    sample_weights_tr = torch.ones_like(y_tr_t)
    sample_weights_va = torch.ones_like(y_va_t)

    pool_tr = X_tr.clone()
    pool_va = X_va.clone()

    all_leaders_va = []
    prev_floor_indices = []
    self.floor_models = nn.ModuleList()

    for floor in range(1, self.n_floors + 1):
      candidate_combos = []
      raw_indices = list(range(n_raw_feats))

      if floor == 1:
        for i in raw_indices:
          candidate_combos.append((i,))

        all_pairs = list(itertools.combinations(raw_indices, 2))
        np.random.shuffle(all_pairs)
        candidate_combos.extend(all_pairs[: self.beam_candidates])

        all_triples = list(itertools.combinations(raw_indices, 3))
        np.random.shuffle(all_triples)
        candidate_combos.extend(all_triples[: self.beam_candidates // 2])
      else:
        cross_combos = []
        for l_idx in prev_floor_indices:
          for r_idx in raw_indices:
            cross_combos.append((l_idx, r_idx))
          for r1, r2 in itertools.combinations(raw_indices, 2):
            cross_combos.append((l_idx, r1, r2))
        np.random.shuffle(cross_combos)
        candidate_combos.extend(cross_combos[: self.beam_candidates])

        leader_pairs = list(itertools.combinations(prev_floor_indices, 2))
        np.random.shuffle(leader_pairs)
        for lp in leader_pairs[: self.beam_candidates // 2]:
          candidate_combos.append(lp)
          r_rand = int(np.random.choice(raw_indices))
          candidate_combos.append(lp + (r_rand,))

      unique_combos = list(dict.fromkeys(tuple(sorted(c)) for c in candidate_combos))
      w_va_np = sample_weights_va.cpu().numpy()

      evaluated = []
      for idx_c, combo in enumerate(unique_combos):
        sc_v, va_log, net_obj = self._train_candidate(
            combo,
            pool_tr,
            y_tr_t,
            sample_weights_tr,
            pool_va,
            y_va_np,
            w_va_np,
            cand_seed=floor * 1000 + idx_c,
        )
        evaluated.append((sc_v, combo, va_log, net_obj))

      evaluated.sort(key=operator.itemgetter(0), reverse=True)
      winners = evaluated[: self.top_k_per_floor]

      floor_layer = nn.ModuleList()
      new_indices = []
      floor_tr_preds = []
      floor_va_preds = []

      for _, _, va_l, net_w in winners:
        net_w.eval()
        with torch.no_grad():
          tr_l = net_w(pool_tr)

        floor_tr_preds.append(torch.sigmoid(tr_l))
        floor_va_preds.append(torch.sigmoid(va_l))

        tr_norm = torch.tanh(tr_l * 0.5).unsqueeze(1)
        va_norm = torch.tanh(va_l * 0.5).unsqueeze(1)

        new_col_id = pool_tr.size(1)
        pool_tr = torch.cat([pool_tr, tr_norm], dim=1)
        pool_va = torch.cat([pool_va, va_norm], dim=1)

        new_indices.append(new_col_id)
        all_leaders_va.append(va_l)
        floor_layer.append(net_w)

      self.floor_models.append(floor_layer)
      prev_floor_indices = new_indices

      # Amplify sample weights on misclassified examples for the next floor
      if self.focus_power > 0.0:
        err_tr = torch.abs(y_tr_t - torch.stack(floor_tr_preds).mean(dim=0))
        err_va = torch.abs(y_va_t - torch.stack(floor_va_preds).mean(dim=0))
        sample_weights_tr = 1.0 + self.focus_power * (err_tr ** 1.5)
        sample_weights_va = 1.0 + self.focus_power * (err_va ** 1.5)
        sample_weights_tr = sample_weights_tr / sample_weights_tr.mean()
        sample_weights_va = sample_weights_va / sample_weights_va.mean()

    meta_va_X = torch.stack(all_leaders_va, dim=1)
    torch.manual_seed(self.seed)
    self.meta_booster = nn.Linear(meta_va_X.size(1), 1).to(device)
    nn.init.constant_(self.meta_booster.weight, 1.0 / meta_va_X.size(1))
    nn.init.zeros_(self.meta_booster.bias)

    opt_meta = torch.optim.AdamW(
        self.meta_booster.parameters(), lr=0.02, weight_decay=0.5
    )
    for _ in range(250):
      self.meta_booster.train()
      opt_meta.zero_grad()
      loss = F.binary_cross_entropy_with_logits(
          self.meta_booster(meta_va_X).squeeze(-1), y_va_t
      )
      loss.backward()
      opt_meta.step()

    return self

  @torch.no_grad()
  def predict_proba(self, X):
    self.eval()
    pool = X.clone()
    all_logits = []

    for floor_layer in self.floor_models:
      floor_outs = []
      for net_w in floor_layer:
        logits = net_w(pool)
        all_logits.append(logits)
        floor_outs.append(torch.tanh(logits * 0.5).unsqueeze(1))
      pool = torch.cat([pool] + floor_outs, dim=1)

    meta_X = torch.stack(all_logits, dim=1)
    return torch.sigmoid(self.meta_booster(meta_X)).squeeze(-1)
