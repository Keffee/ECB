"""CLI entry points for external integrations; no datasets are bundled."""
import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import random

import torch
from .config import Config
from .evaluation import ranking_metrics
from .models import Gate
from .policy import JointPolicy
from .reward import TrajectoryReward
from .training import Trainer
from .workflow import Workflow


def read_spec(path):
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    unknown = set(spec) - {"integration", "integration_options", "model", "reward_fit",
                          "reward_checkpoint", "output_dir"}
    if unknown:
        raise ValueError(f"unknown configuration fields: {sorted(unknown)}")
    return spec, Config(**spec.get("model", {}))


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def integration(spec, config):
    module_name, attribute = spec["integration"].split(":", 1)
    factory = getattr(importlib.import_module(module_name), attribute)
    adapter = factory(spec.get("integration_options", {}), config)
    # External integrations may wrap non-PyTorch services; module adapters are
    # explicitly frozen here. The API contract also requires frozen internals.
    for component in (adapter.base, adapter.ranker, adapter.retriever):
        if isinstance(component, torch.nn.Module):
            component.eval()
            component.requires_grad_(False)
    return adapter


def new_output(path):
    output = Path(path)
    output.mkdir(parents=True, exist_ok=False)
    return output


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_manifest(output, inputs, config):
    artifacts = {p.name: sha256(p) for p in output.iterdir() if p.is_file() and p.name != "manifest.json"}
    write_json(output / "manifest.json", {
        "format": "ecb-run-v1", "config": config.as_dict(), "seed": config.seed,
        "input_sha256": {name: sha256(path) for name, path in inputs.items()},
        "output_sha256": artifacts, "torch_version": str(torch.__version__),
    })


def run_reward_fit(config_path, output_dir):
    spec, config = read_spec(config_path)
    seed_everything(config.seed)
    adapter = integration(spec, config)
    examples = list(adapter.reward_examples())
    output = new_output(output_dir)
    model = TrajectoryReward(config.embedding_dim, config.hidden_dim)
    loss = model.fit(examples, **spec.get("reward_fit", {}))
    path = output / "reward.pt"
    torch.save(model.checkpoint(), path)
    write_json(output / "fit.json", {"records": len(examples), "loss": loss, "fit_split": "train"})
    write_manifest(output, {"config": config_path}, config)
    return path


def load_reward(path, config):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format") != "ecb-reward-v1" or saved.get("fit_split") != "train":
        raise ValueError("expected train-only ECB reward checkpoint")
    if saved["embedding_dim"] != config.embedding_dim:
        raise ValueError("reward embedding dimension differs from controller")
    reward = TrajectoryReward(saved["embedding_dim"], saved["hidden_dim"])
    reward.load_state_dict(saved["state_dict"])
    reward.fitted = True
    reward.freeze()
    return reward


def run_train(config_path):
    spec, config = read_spec(config_path)
    seed_everything(config.seed)
    reward = load_reward(spec["reward_checkpoint"], config)
    adapter = integration(spec, config)
    actor = JointPolicy(config.state_dim, config.embedding_dim, config.hidden_dim)
    gate = Gate(config.state_dim, config.hidden_dim)
    workflow = Workflow(config, adapter.base, adapter.ranker, adapter.retriever, gate, actor)
    trainer = Trainer(workflow, reward)
    output = new_output(spec["output_dir"])
    with (output / "train_trace.jsonl").open("w", encoding="utf-8") as stream:
        def callback(kind, epoch, value):
            if kind == "rollout":
                for trajectory in value:
                    stream.write(json.dumps({"epoch": epoch, **trajectory.public_trace()}, allow_nan=False) + "\n")
            else:
                print(json.dumps(value, allow_nan=False), flush=True)
        history = trainer.fit(adapter.training_records, callback)
    checkpoint = output / "controller.pt"
    torch.save({
        "format": "ecb-controller-v1", "config": config.as_dict(),
        "actor": actor.state_dict(), "gate": gate.state_dict(),
        "critic": trainer.critic.state_dict(), "reward_sha256": sha256(spec["reward_checkpoint"]),
    }, checkpoint)
    write_json(output / "training.json", history)
    write_manifest(output, {"config": config_path, "reward_checkpoint": spec["reward_checkpoint"]}, config)
    return checkpoint


def run_evaluate(config_path, checkpoint, output_dir, deterministic=False):
    spec, requested_config = read_spec(config_path)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("format") != "ecb-controller-v1":
        raise ValueError("expected ECB controller checkpoint")
    config = Config(**saved["config"])
    if config.as_dict() != requested_config.as_dict():
        raise ValueError("evaluation model config must match the frozen checkpoint")
    seed_everything(config.seed)
    adapter = integration(spec, config)
    actor = JointPolicy(config.state_dim, config.embedding_dim, config.hidden_dim)
    gate = Gate(config.state_dim, config.hidden_dim)
    actor.load_state_dict(saved["actor"])
    gate.load_state_dict(saved["gate"])
    actor.eval().requires_grad_(False)
    gate.eval().requires_grad_(False)
    workflow = Workflow(config, adapter.base, adapter.ranker, adapter.retriever, gate, actor)
    output = new_output(output_dir)
    trajectories, identities = [], set()
    with (output / "trace.jsonl").open("w", encoding="utf-8") as stream:
        for context in adapter.evaluation_contexts():
            if context.request_id in identities:
                raise ValueError("duplicate evaluation request identity")
            identities.add(context.request_id)
            trajectory = workflow.run(context, deterministic=deterministic)
            trajectories.append(trajectory)
            stream.write(json.dumps(trajectory.public_trace(), allow_nan=False) + "\n")
    # Targets are obtained only after all workflow decisions have completed.
    metrics = ranking_metrics([t.item_ids for t in trajectories], list(adapter.evaluation_targets()))
    metrics["mean_cost_units"] = sum(t.cost.units for t in trajectories) / len(trajectories)
    metrics["action_mode"] = "conditional_mean" if deterministic else "sampled"
    metrics["stop_reasons"] = {reason: sum(t.stop_reason == reason for t in trajectories)
                               for reason in sorted({t.stop_reason for t in trajectories})}
    write_json(output / "metrics.json", metrics)
    write_manifest(output, {"config": config_path, "controller": checkpoint}, config)
    return metrics


def train_main():
    parser = argparse.ArgumentParser(description="Train the ECB controller.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run_train(args.config))


def reward_main():
    parser = argparse.ArgumentParser(description="Fit a train-only trajectory reward model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(run_reward_fit(args.config, args.output))


def evaluate_main():
    parser = argparse.ArgumentParser(description="Evaluate a frozen ECB controller.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--deterministic", action="store_true",
                        help="Use conditional means instead of the default sampled actions.")
    args = parser.parse_args()
    print(json.dumps(run_evaluate(args.config, args.checkpoint, args.output, args.deterministic), indent=2))
