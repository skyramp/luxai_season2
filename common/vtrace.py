# This file taken from
#     https://github.com/deepmind/scalable_agent/blob/
#         cd66d00914d56c8ba2f0615d9cdeefcb169a8d70/vtrace.py
# and modified.

# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions to compute V-trace off-policy actor critic targets.

For details and theory see:

"IMPALA: Scalable Distributed Deep-RL with
Importance Weighted Actor-Learner Architectures"
by Espeholt, Soyer, Munos et al.

See https://arxiv.org/abs/1802.01561 for the full paper.
"""

import collections

import torch
import torch.nn.functional as F


VTraceFromLogitsReturns = collections.namedtuple(
    "VTraceFromLogitsReturns",
    [
        "vs",
        "pg_advantages",
        "log_rhos",
        "behavior_action_log_probs",
        "target_action_log_probs",
    ],
)

VTraceReturns = collections.namedtuple("VTraceReturns", "vs pg_advantages")

GaeVTraceReturns = collections.namedtuple("VTraceReturns", "vs pg_advantages next_acc next_vs")


@torch.no_grad()
def gae_lambda_vtrace(log_rhos, discounts, rewards,
                      values, normalized_values, mu, sigma,
                      bootstrap_value, bootstrap_vs, bootstrap_acc, clip_rho_threshold=1.0, clip_cs_threshold = 1.0, clip_pg_rho_threshold=1.0, gae_lambda = 0.95):
    with torch.no_grad():
        rhos = torch.exp(log_rhos)
        if clip_rho_threshold is not None:
            clipped_rhos = torch.clamp(rhos, max=clip_rho_threshold)
        else:
            clipped_rhos = rhos

        cs = gae_lambda * torch.clamp(rhos, max=clip_cs_threshold)

        # Append bootstrapped value to get [v1, ..., v_t+1]
        values_t_plus_1 = torch.cat(
            [values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0
        )

        deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)

        acc = bootstrap_acc
        result = []
        for t in range(discounts.shape[0] - 1, -1, -1):
            acc = deltas[t] + discounts[t] * cs[t] * acc
            result.append(acc)
        result.reverse()
        vs_minus_v_xs = torch.stack(result)

        # Add V(x_s) to get v_s.
        vs = torch.add(vs_minus_v_xs, values)

        # Advantage for policy gradient.
        broadcasted_bootstrap_values = bootstrap_vs
        vs_t_plus_1 = torch.cat(
            [vs[1:], broadcasted_bootstrap_values.unsqueeze(0)], dim=0
        )
        if clip_pg_rho_threshold is not None:
            clipped_pg_rhos = torch.clamp(rhos, max=clip_pg_rho_threshold)
        else:
            clipped_pg_rhos = rhos

        # pg_advantages = clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values)
        pg_advantages = clipped_pg_rhos * (((rewards + discounts * vs_t_plus_1) - mu) / sigma - normalized_values)

        # Make sure no gradients backpropagated through the returned values.
        return GaeVTraceReturns(vs=vs.detach().clone(), pg_advantages=pg_advantages.detach().clone(), next_acc=acc.detach().clone(), next_vs=vs[0].detach().clone())

