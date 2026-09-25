import importlib.util
import json
from dataclasses import replace

import pytest
import torch

from ecb.cli import run_evaluate, run_reward_fit, run_train
from ecb.config import Config
from ecb.types import Candidate, Cost, RankedList, ScoredList, State


def module():
    assert importlib.util.find_spec("ecb.integrations") is not None, "default MiniOneRec integration is missing"
    assert importlib.util.find_spec("ecb.integrations.minionerec") is not None, "default MiniOneRec integration is missing"
    from ecb.integrations import minionerec
    return minionerec


def inputs(tmp_path):
    def write(name, rows):
        path = tmp_path / name
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return str(path)
    catalog = write("catalog.jsonl", [
        {"item_id": str(i), "sid": f"<sid_{i}>", "embedding": [1., float(i)], "text": f"toy {i}"}
        for i in range(7)
    ])
    context = {"request_id": "r", "history_ids": ["0"], "embedding": [1., 1.], "text": "toy"}
    train = write("train.jsonl", [dict(context, request_id="train-r")])
    train_targets = write("train_targets.jsonl", [{"request_id": "train-r", "item_id": "4"}])
    evaluation = write("eval.jsonl", [dict(context, request_id="eval-r")])
    targets = write("targets.jsonl", [{"request_id": "eval-r", "item_id": "4"}])
    transitions = tmp_path / "transitions.json"
    transitions.write_text(json.dumps({"fit_split": "train", "scores": {"0": {"4": 3., "5": 2.}}}))
    return {"catalog": catalog, "train_contexts": train, "train_targets": train_targets,
            "eval_contexts": evaluation, "eval_targets": targets,
            "transitions": str(transitions), "minionerec_root": str(tmp_path),
            "base_checkpoint": str(tmp_path), "ranker_checkpoint": str(tmp_path), "device": "cpu"}


class Base:
    def __init__(self, catalog):
        self.catalog = catalog

    def rank(self, state, pool, limit):
        values = pool or tuple(self.catalog.candidates[i] for i in ("1", "2"))
        values = tuple(sorted(values, key=lambda x: x.item_id, reverse=True))[:limit]
        return RankedList(values, tuple(float(len(values) - i) for i in range(len(values))))


class Ranker:
    def estimate_cost(self, state, candidates):
        return .01 * len(candidates)

    def score(self, state, candidates):
        return ScoredList(tuple(c.item_id for c in candidates), tuple(float(c.item_id) for c in candidates), Cost(units=.01 * len(candidates)))


def patch_components(monkeypatch):
    m = module()
    monkeypatch.setattr(m, "load_components", lambda options, catalog, contexts: (Base(catalog), Ranker()))
    return m


def test_default_factory_exists():
    module()


def test_prepared_inputs_run_all_three_commands(tmp_path, monkeypatch):
    patch_components(monkeypatch)
    options = inputs(tmp_path)
    spec = {"integration": "ecb.integrations.minionerec:build", "integration_options": options,
            "model": {"embedding_dim": 2, "shortlist_size": 2, "pool_size": 7, "retrieval_slots": 3,
                      "epochs": 2, "warmup_epochs": 1, "max_rounds": 1, "hidden_dim": 8, "gate_passes": 1},
            "reward_fit": {"epochs": 2}, "reward_checkpoint": str(tmp_path / "reward" / "reward.pt"),
            "output_dir": str(tmp_path / "controller")}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(spec))
    assert run_reward_fit(path, tmp_path / "reward").is_file()
    checkpoint = run_train(path)
    assert checkpoint.is_file()
    metrics = run_evaluate(path, checkpoint, tmp_path / "evaluation")
    assert metrics["count"] == 1
    assert (tmp_path / "evaluation" / "trace.jsonl").is_file()


def test_targets_are_lazy_and_identity_checked(tmp_path, monkeypatch):
    m = patch_components(monkeypatch)
    options = inputs(tmp_path)
    adapter = m.build(options, Config(embedding_dim=2))
    target_path = tmp_path / "targets.jsonl"
    target_path.unlink()
    context = list(adapter.evaluation_contexts())[0]
    assert context.request_id == "eval-r"
    assert not hasattr(context, "target")
    with pytest.raises(FileNotFoundError):
        list(adapter.evaluation_targets())
    target_path.write_text(json.dumps({"request_id": "wrong", "item_id": "4"}) + "\n")
    with pytest.raises(ValueError, match="identit"):
        list(adapter.evaluation_targets())


def test_retrieval_has_three_sources_and_excludes_history(tmp_path, monkeypatch):
    m = patch_components(monkeypatch)
    options = inputs(tmp_path)
    adapter = m.build(options, Config(embedding_dim=2))
    context = list(adapter.evaluation_contexts())[0]
    result = adapter.retriever.retrieve(State(context), (1., 1.), (1, 1, 1), frozenset({"1"}))
    assert result.source_counts == (1, 1, 1)
    ids = [c.item_id for c in result.candidates]
    assert "0" not in ids and "1" not in ids and len(ids) == len(set(ids)) == 3
    assert result.cost.units <= adapter.retriever.estimate_cost((1, 1, 1))


def test_rejects_target_fields_in_contexts(tmp_path, monkeypatch):
    m = patch_components(monkeypatch)
    options = inputs(tmp_path)
    p = tmp_path / "eval.jsonl"
    row = json.loads(p.read_text())
    row["target_sid"] = "<sid_4>"
    p.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="context"):
        m.build(options, Config(embedding_dim=2))


def test_transition_training_provenance_required(tmp_path, monkeypatch):
    m = patch_components(monkeypatch)
    options = inputs(tmp_path)
    p = tmp_path / "transitions.json"
    row = json.loads(p.read_text())
    row["fit_split"] = "test"
    p.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="train"):
        m.build(options, Config(embedding_dim=2))


def test_sequence_scores_cover_suffix_and_eos_only():
    from types import SimpleNamespace
    m = module()
    logits = torch.tensor([[[3., 1., 0.], [0., 3., 1.], [1., 0., 3.], [3., 0., 1.]]])
    class Fixed:
        def __call__(self, input_ids, attention_mask):
            return SimpleNamespace(logits=logits.expand(input_ids.shape[0], -1, -1))
    scores = m.sequence_log_probs(Fixed(), [0, 1], [[1, 2], [0]], 0, "cpu", 2)
    expected = torch.log_softmax(logits, -1)
    assert scores[0] == pytest.approx(float(expected[0, 1, 1] + expected[0, 2, 2]))
    assert scores[1] == pytest.approx(float(expected[0, 1, 0]))


def test_colliding_sids_keep_distinct_item_ids(tmp_path):
    m = module()
    path = tmp_path / "catalog.jsonl"
    path.write_text("".join(json.dumps({"item_id": item_id, "sid": "same", "embedding": [1., 0.]}) + "\n"
                            for item_id in ["a", "b"]))
    catalog = m.Catalog(path, 2)
    assert catalog.sid_items["same"] == ["a", "b"]
