"""Optional CPU check with real tiny Transformers models and upstream decoder."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from test_minionerec import inputs


def test_documented_commands_with_real_models(tmp_path):
    upstream = os.environ.get("MINIONEREC_ROOT")
    if not upstream:
        pytest.skip("set MINIONEREC_ROOT to check the upstream decoder")
    pytest.importorskip("transformers")
    from transformers import Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit

    torch.manual_seed(42)
    vocabulary = ["[UNK]", "[PAD]", "[EOS]", "###", "Response:"] + [f"<sid_{i}>" for i in range(7)] + [str(i) for i in range(5)]
    tokenizer = Tokenizer(WordLevel(dict(zip(vocabulary, range(len(vocabulary)))), unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
    checkpoint = tmp_path / "model"
    config = Qwen2Config(vocab_size=len(vocabulary), hidden_size=16, intermediate_size=32,
                        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                        eos_token_id=2, pad_token_id=1, bos_token_id=2, max_position_embeddings=512)
    Qwen2ForCausalLM(config).save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    options = inputs(tmp_path)
    options.update(minionerec_root=upstream, base_checkpoint=str(checkpoint), ranker_checkpoint=str(checkpoint), reward_rollouts=1)
    spec = {"integration": "ecb.integrations.minionerec:build", "integration_options": options,
            "model": {"embedding_dim": 2, "shortlist_size": 2, "pool_size": 7, "retrieval_slots": 3,
                      "epochs": 2, "warmup_epochs": 1, "max_rounds": 1, "hidden_dim": 8, "gate_passes": 1},
            "reward_fit": {"epochs": 2}, "reward_checkpoint": str(tmp_path / "reward" / "reward.pt"),
            "output_dir": str(tmp_path / "controller")}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(spec))
    scripts = Path(sys.executable).parent
    commands = [
        [str(scripts / "ecb-fit-reward"), "--config", str(path), "--output", str(tmp_path / "reward")],
        [str(scripts / "ecb-train"), "--config", str(path)],
        [str(scripts / "ecb-evaluate"), "--config", str(path), "--checkpoint", str(tmp_path / "controller" / "controller.pt"), "--output", str(tmp_path / "evaluation")],
    ]
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
    metrics = json.loads((tmp_path / "evaluation" / "metrics.json").read_text())
    assert metrics["count"] == 1
    trace = json.loads((tmp_path / "evaluation" / "trace.jsonl").read_text())
    assert len(trace["item_ids"]) == 2
    assert "failure" not in trace["stop_reason"]
    training = [json.loads(line) for line in (tmp_path / "controller" / "train_trace.jsonl").read_text().splitlines()]
    assert training and any(row["steps"] for row in training)
    assert all("failure" not in row["stop_reason"] for row in training)
    assert all(row["steps"][0]["added_ids"] for row in training if row["steps"])
