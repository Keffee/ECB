"""On-policy likelihood-ratio updates with separate gate fitting."""
import math
import random
import torch
from torch import nn
from .features import vector
from .models import ValueBaseline
from .types import TrainingRecord


def reward_to_go(terminal_utility, costs):
    if not math.isfinite(terminal_utility) or any(not math.isfinite(c) or c < 0 for c in costs):
        raise ValueError("utility and costs must be finite")
    running = terminal_utility
    returns = []
    for cost in reversed(costs):
        running -= cost
        returns.append(running)
    return list(reversed(returns))


class Trainer:
    def __init__(self, workflow, reward_model):
        if not reward_model.fitted or any(p.requires_grad for p in reward_model.parameters()):
            raise ValueError("use a fitted, frozen train-only reward model")
        self.workflow, self.reward = workflow, reward_model
        cfg = workflow.config
        self.critic = ValueBaseline(cfg.critic_dim, cfg.hidden_dim).to(next(workflow.actor.parameters()).device)
        self.actor_optimizer = torch.optim.Adam(workflow.actor.parameters(), lr=cfg.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.learning_rate)
        self.gate_optimizer = torch.optim.Adam(workflow.gate.parameters(), lr=cfg.learning_rate)

    def _validate(self, records):
        records = list(records)
        if not records:
            raise ValueError("empty training record batch")
        for record in records:
            if not isinstance(record, TrainingRecord) or record.split != "train":
                raise ValueError("controller fitting accepts training records only")
            vector(record.target_embedding, self.workflow.config.embedding_dim)
        return records

    def actor_batch(self, records, *, force_all=False):
        records = self._validate(records)
        cfg, workflow = self.workflow.config, self.workflow
        for p in workflow.gate.parameters():
            p.requires_grad_(False)
        episode_losses, values, targets, utilities, trajectories = [], [], [], [], []
        for record in records:
            trajectory = workflow.run(record.context, training=True, force_all=force_all)
            trajectories.append(trajectory)
            utility = self.reward.utility(trajectory, record.target_embedding)
            utilities.append(utility - trajectory.cost.units)
            returns = reward_to_go(utility, [step.cost.units for step in trajectory.steps])
            terms = []
            for step, return_ in zip(trajectory.steps, returns):
                if step.log_prob is None:
                    continue
                value = self.critic(step.critic_features).squeeze(0)
                advantage = torch.as_tensor(return_, device=value.device) - value.detach()
                terms.append(-advantage.detach() * step.log_prob - cfg.entropy_weight * step.entropy)
                values.append(value)
                targets.append(return_)
            if terms:
                episode_losses.append(torch.stack(terms).sum())
        # Average sums over ALL sampled trajectories, including those that STOP.
        actor_loss_value, critic_loss_value = 0., 0.
        if episode_losses:
            loss = torch.stack(episode_losses).sum() / len(records)
            self.actor_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(workflow.actor.parameters(), cfg.gradient_clip, error_if_nonfinite=True)
            self.actor_optimizer.step()
            actor_loss_value = float(loss.detach())
            value_tensor = torch.stack(values)
            critic_loss = nn.functional.mse_loss(value_tensor, torch.tensor(targets, device=value_tensor.device))
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.gradient_clip, error_if_nonfinite=True)
            self.critic_optimizer.step()
            critic_loss_value = float(critic_loss.detach())
        return {"episodes": len(records), "mean_net_utility": sum(utilities) / len(records),
                "actor_loss": actor_loss_value, "critic_loss": critic_loss_value,
                "actor_steps": len(values)}, trajectories

    def gate_batch(self, records):
        records = self._validate(records)
        cfg, workflow = self.workflow.config, self.workflow
        features, advantages = [], []
        # Probe an executed first continuation on TRAIN records, then follow the
        # frozen current gate. No counterfactual action reward table is constructed.
        for record in records:
            trajectory = workflow.run(record.context, force_first=True)
            utility = self.reward.utility(trajectory, record.target_embedding)
            returns = reward_to_go(utility, [step.cost.units for step in trajectory.steps])
            for index, (step, continuation) in enumerate(zip(trajectory.steps, returns)):
                stopped = self.reward.utility(trajectory.prefix(index), record.target_embedding)
                features.append(step.pre_features.detach())
                advantages.append(continuation - stopped)
        if not features:
            return {"gate_examples": 0, "gate_loss": 0.}
        x = torch.cat(features)
        y = torch.tensor(advantages, device=x.device)
        for p in workflow.gate.parameters():
            p.requires_grad_(True)
        for _ in range(cfg.gate_passes):
            loss = nn.functional.mse_loss(workflow.gate(x), y)
            self.gate_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(workflow.gate.parameters(), cfg.gradient_clip, error_if_nonfinite=True)
            self.gate_optimizer.step()
        for p in workflow.gate.parameters():
            p.requires_grad_(False)
        return {"gate_examples": len(advantages), "gate_loss": float(loss.detach())}

    def fit(self, records_factory, callback=None):
        cfg = self.workflow.config
        rng = random.Random(cfg.seed)
        history = []
        for epoch in range(cfg.epochs):
            records = self._validate(records_factory())
            rng.shuffle(records)
            batches = [records[start:start + cfg.batch_size] for start in range(0, len(records), cfg.batch_size)]
            logs = []
            for batch in batches:
                log, trajectories = self.actor_batch(batch, force_all=epoch < cfg.warmup_epochs)
                logs.append(log)
                if callback:
                    callback("rollout", epoch, trajectories)
            # Gate is held fixed through actor phase; actor fixed during gate phase.
            gate_log = self.gate_batch(records)
            log = {"epoch": epoch, "episodes": len(records), "actor_steps": sum(x["actor_steps"] for x in logs),
                   "mean_net_utility": sum(x["mean_net_utility"] * x["episodes"] for x in logs) / len(records),
                   **gate_log}
            history.append(log)
            if callback:
                callback("epoch", epoch, log)
        return history
