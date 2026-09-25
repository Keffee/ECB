import json
from pathlib import Path
import torch
import pytest

from ecb.cli import run_train, run_reward_fit, run_evaluate
from ecb.config import Config
from ecb.models import Gate
from ecb.policy import JointPolicy
from ecb.reward import TrajectoryReward
from ecb.training import Trainer
from ecb.types import TrainingRecord, RewardExample
from ecb.workflow import Workflow
from test_workflow import Base, Ranker, Retriever, context, setup_workflow


def frozen_reward():
    a, _, _, _ = setup_workflow(False)
    b, _, _, _ = setup_workflow(True)
    model = TrajectoryReward(2, 12)
    examples = [
        RewardExample(a.run(context()), (0., 1.), .1),
        RewardExample(b.run(context()), (0., 1.), .9),
    ]
    model.fit(examples, epochs=2)
    return model


def test_actor_updates_every_head_with_frozen_reward_and_gate():
    torch.manual_seed(24)
    workflow, _, _, _ = setup_workflow()
    reward = frozen_reward()
    trainer = Trainer(workflow, reward)
    actor_before = {k: v.clone() for k, v in workflow.actor.state_dict().items()}
    gate_before = {k: v.clone() for k, v in workflow.gate.state_dict().items()}
    reward_before = {k: v.clone() for k, v in reward.state_dict().items()}
    log, _ = trainer.actor_batch([TrainingRecord(context(), (0., 1.))])
    assert log["actor_steps"] == 2
    for prefix in ("lambda_head", "rho_head", "omega_head"):
        assert any(not torch.equal(v, actor_before[k]) for k, v in workflow.actor.state_dict().items() if k.startswith(prefix))
    assert all(torch.equal(v, gate_before[k]) for k, v in workflow.gate.state_dict().items())
    assert all(torch.equal(v, reward_before[k]) for k, v in reward.state_dict().items())


def test_all_stop_batch_skips_actor():
    workflow, _, _, _ = setup_workflow(False)
    trainer = Trainer(workflow, frozen_reward())
    log, _ = trainer.actor_batch([TrainingRecord(context(), (0., 1.))])
    assert log["actor_steps"] == 0
    assert log["actor_loss"] == 0
    with pytest.raises(ValueError):
        TrainingRecord(context(), (0., 1.), "validation")


def test_gate_fit_only_changes_gate():
    workflow, _, _, _ = setup_workflow()
    trainer = Trainer(workflow, frozen_reward())
    actor_before = {k: v.clone() for k, v in workflow.actor.state_dict().items()}
    before = {k: v.clone() for k, v in workflow.gate.state_dict().items()}
    report = trainer.gate_batch([TrainingRecord(context(), (0., 1.))])
    assert report["gate_examples"] == 2
    assert all(torch.equal(v, actor_before[k]) for k, v in workflow.actor.state_dict().items())
    assert any(not torch.equal(v, before[k]) for k, v in workflow.gate.state_dict().items())


class FixtureIntegration:
    def __init__(self, config):
        self.config = config
        self.base, self.ranker, self.retriever = Base(), Ranker(), Retriever()

    def training_records(self):
        return [TrainingRecord(context(), (0., 1.))]

    def reward_examples(self):
        workflow = Workflow(self.config, self.base, self.ranker, self.retriever,
                            Gate(self.config.state_dim, self.config.hidden_dim),
                            JointPolicy(self.config.state_dim, 2, self.config.hidden_dim))
        trajectory = workflow.run(context(), force_all=True)
        return [RewardExample(trajectory, (0., 1.), .8)]

    def evaluation_contexts(self):
        return [context()]

    def evaluation_targets(self):
        return ["a"]


def factory(options, config):
    return FixtureIntegration(config)


def test_training_checkpoint_and_evaluation_cli(tmp_path):
    spec = {
        "integration": "test_training_integration:factory",
        "integration_options": {},
        "model": {"embedding_dim": 2, "hidden_dim": 12, "epochs": 2,
                  "warmup_epochs": 1, "max_rounds": 1, "gate_passes": 2},
        "reward_fit": {"epochs": 2},
        "reward_checkpoint": str(tmp_path / "reward" / "reward.pt"),
        "output_dir": str(tmp_path / "controller"),
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(spec))
    reward_path = run_reward_fit(config_path, tmp_path / "reward")
    assert reward_path.is_file()
    checkpoint = run_train(config_path)
    assert checkpoint.is_file()
    assert (checkpoint.parent / "manifest.json").is_file()
    metrics = run_evaluate(config_path, checkpoint, tmp_path / "evaluation")
    assert metrics["count"] == 1
    assert "hit@5" in metrics
    with pytest.raises(FileExistsError):
        run_train(config_path)
