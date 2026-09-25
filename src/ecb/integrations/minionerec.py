"""MiniOneRec, a frozen local pointwise scorer, and three retrieval sources.

Consumes prepared inputs only. Targets are loaded by the training/evaluation
readers and are never stored in inference contexts or recommendation components.
"""
from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path
import re
import time

import torch
import torch.nn.functional as F

from ..features import validate_context
from ..models import Gate
from ..policy import JointPolicy
from ..ranker import OrganizedEvidenceRanker
from ..types import (Candidate, Context, Cost, RankedList, RetrievalResult,
                     RewardExample, ServiceFailure, State, TrainingRecord)
from ..workflow import Workflow


CONTEXT_FIELDS = {"request_id", "history_ids", "embedding", "text"}
OPTION_FIELDS = {"catalog", "train_contexts", "train_targets", "eval_contexts", "eval_targets",
                 "transitions", "minionerec_root", "base_checkpoint", "ranker_checkpoint",
                 "device", "max_prompt_tokens", "score_batch_size", "signal_items",
                 "ranker_cost", "retrieval_cost", "reward_rollouts"}


def jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"expected an object at {path}:{number}")
                yield row


def finite_vector(value, dimension):
    if not isinstance(value, (list, tuple)) or len(value) != dimension:
        raise ValueError(f"expected a vector of dimension {dimension}")
    result = tuple(float(x) for x in value)
    if not all(math.isfinite(x) for x in result):
        raise ValueError("embeddings must be finite")
    return result


def terms(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class Catalog:
    def __init__(self, path, dimension):
        self.candidates, self.sids, self.sid_items = {}, {}, {}
        for row in jsonl(path):
            if set(row) - {"item_id", "sid", "embedding", "text"}:
                raise ValueError("unexpected catalog field")
            item_id, sid = str(row["item_id"]), str(row["sid"])
            if item_id in self.candidates or not item_id or not sid:
                raise ValueError("catalog requires unique nonempty item IDs and nonempty SIDs")
            self.candidates[item_id] = Candidate(item_id, finite_vector(row["embedding"], dimension), str(row.get("text", "")))
            self.sids[item_id] = sid
            self.sid_items.setdefault(sid, []).append(item_id)
        if not self.candidates:
            raise ValueError("catalog is empty")
        self.ids = tuple(sorted(self.candidates))
        self.matrix = F.normalize(torch.tensor([self.candidates[i].embedding for i in self.ids]), dim=-1)
        self.positions = {item_id: i for i, item_id in enumerate(self.ids)}
        self.tokens = {i: terms(self.candidates[i].text) for i in self.ids}

    def similarities(self, signal):
        return (self.matrix @ F.normalize(torch.tensor(signal, dtype=torch.float32), dim=0)).tolist()

    def nearest(self, signal, limit, excluded=()):
        scores = self.similarities(signal)
        return sorted((i for i in self.ids if i not in excluded),
                      key=lambda i: (-scores[self.positions[i]], i))[:limit]


def read_contexts(path, catalog, config):
    result, identities = [], set()
    for row in jsonl(path):
        if set(row) - CONTEXT_FIELDS:
            raise ValueError("unexpected or target-like field in context")
        request_id = str(row["request_id"])
        if not request_id or request_id in identities:
            raise ValueError("context identities must be nonempty and unique")
        identities.add(request_id)
        history_ids = tuple(str(x) for x in row["history_ids"][-config.history_size:])
        history = tuple(catalog.candidates[i] for i in history_ids)
        context = Context(request_id, tuple(c.embedding for c in history),
                          finite_vector(row["embedding"], config.embedding_dim),
                          str(row.get("text", "")), tuple(c.text for c in history))
        validate_context(context, config.embedding_dim)
        result.append((context, history_ids))
    if not result:
        raise ValueError("context file is empty")
    return result


def load_lm(checkpoint, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("checkpoint tokenizer needs an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, local_files_only=True, torch_dtype=(torch.bfloat16 if device.startswith("cuda") else torch.float32),
        attn_implementation="sdpa").to(device).eval().requires_grad_(False)
    return model, tokenizer


def prompt(history_sids, request, signal_sids=()):
    value = ("Below is an instruction that describes a task, paired with an input that provides further context. "
             "Write a response that appropriately completes the request.\n\n"
             "### Instruction:\nCan you predict the next possible item that the user may expect?\n\n"
             "### User Input:\nCan you predict the next possible item the user may expect, given the following chronological interaction history: "
             + ", ".join(history_sids))
    if request:
        value += "\nRequest: " + request
    if signal_sids:
        value += "\nCurrent retrieval interests: " + ", ".join(signal_sids)
    return value + "\n\n### Response:\n"


def sequence_log_probs(model, prefix, suffixes, pad_id, device, batch_size):
    """Sum next-token log probabilities over each SID and EOS only."""
    result = []
    for start in range(0, len(suffixes), batch_size):
        group = suffixes[start:start + batch_size]
        rows = [prefix + suffix for suffix in group]
        width = max(map(len, rows))
        ids = torch.tensor([row + [pad_id] * (width - len(row)) for row in rows], device=device)
        mask = torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in rows], device=device)
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask).logits
            for index, suffix in enumerate(group):
                scores = logits[index, len(prefix) - 1:len(prefix) + len(suffix) - 1].float()
                targets = torch.tensor(suffix, device=device)
                logp = F.log_softmax(scores, dim=-1).gather(1, targets[:, None]).sum()
                result.append(float(logp))
    return result


class MiniOneRec:
    def __init__(self, options, catalog, contexts):
        self.options, self.catalog, self.contexts = options, catalog, contexts
        self.device = options.get("device", "cuda:0")
        source = Path(options["minionerec_root"]) / "LogitProcessor.py"
        if not source.is_file():
            raise FileNotFoundError("MiniOneRec/LogitProcessor.py is missing; clone MiniOneRec and set minionerec_root")
        spec = importlib.util.spec_from_file_location("ecb_minionerec_logits", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.processor = module.ConstrainedLogitsProcessor
        self.model, self.tokenizer = load_lm(options["base_checkpoint"], self.device)
        self.response_prefix = self.tokenizer.encode("### Response:\n", add_special_tokens=False)
        self.suffixes = {}
        for sid in catalog.sid_items:
            complete = self.tokenizer.encode("### Response:\n" + sid + "\n", add_special_tokens=False)
            if complete[:len(self.response_prefix)] != self.response_prefix:
                raise ValueError("SID tokenization changes the MiniOneRec response prefix")
            self.suffixes[sid] = complete[len(self.response_prefix):] + [self.tokenizer.eos_token_id]
        encoded_sids = [tuple(value) for value in self.suffixes.values()]
        if len(encoded_sids) != len(set(encoded_sids)):
            raise ValueError("different SIDs collapse to identical tokenizer output")

    def _prefix(self, state):
        history_ids = self.contexts[state.context.request_id][1]
        signal_ids = self.catalog.nearest(state.signal, self.options.get("signal_items", 3), history_ids) if state.signal else ()
        value = prompt([self.catalog.sids[i] for i in history_ids], state.context.text,
                       [self.catalog.sids[i] for i in signal_ids])
        ids = self.tokenizer.encode(value, add_special_tokens=False)
        maximum = self.options.get("max_prompt_tokens", 2048)
        ids = ids[-maximum:]
        if ids[-len(self.response_prefix):] != self.response_prefix:
            raise ValueError("prompt must end at the MiniOneRec response prefix")
        return ids

    def rank(self, state, pool, limit):
        prefix = self._prefix(state)
        if pool:
            candidates = tuple(pool)
        else:
            excluded = set(self.contexts[state.context.request_id][1])
            valid_sids = [sid for sid, ids in self.catalog.sid_items.items() if any(i not in excluded for i in ids)]
            if not valid_sids:
                return RankedList((), ())
            beams = min(limit, len(valid_sids))
            allowed = {}
            for sid in valid_sids:
                suffix = self.suffixes[sid]
                for i, token in enumerate(suffix):
                    key = tuple(self.response_prefix) if i == 0 else tuple(suffix[:i])
                    allowed.setdefault(key, set()).add(token)
            processor = self.processor(lambda batch, values: sorted(allowed.get(tuple(values), ())),
                                       beams, base_model="qwen", eos_token_id=self.tokenizer.eos_token_id)
            processor.prefix_index = len(self.response_prefix)
            from transformers import LogitsProcessorList
            ids = torch.tensor([prefix], device=self.device)
            with torch.inference_mode():
                outputs = self.model.generate(
                    input_ids=ids, attention_mask=torch.ones_like(ids), num_beams=beams,
                    num_return_sequences=beams, do_sample=False, length_penalty=0.,
                    max_new_tokens=max(len(self.suffixes[sid]) for sid in valid_sids),
                    pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id,
                    logits_processor=LogitsProcessorList([processor]), use_cache=True)
            lookup = {tuple(self.suffixes[sid][:-1]): sid for sid in valid_sids}
            chosen = {}
            for row in outputs[:, len(prefix):].tolist():
                if self.tokenizer.eos_token_id in row:
                    row = row[:row.index(self.tokenizer.eos_token_id)]
                sid = lookup.get(tuple(row))
                if sid is None:
                    continue
                for item_id in self.catalog.sid_items[sid]:
                    if item_id not in excluded:
                        chosen[item_id] = replace(self.catalog.candidates[item_id], evidence=("minionerec",))
            if not chosen:
                raise ServiceFailure("MiniOneRec generated no decodable catalog SID")
            candidates = tuple(chosen.values())
        sids = list(dict.fromkeys(self.catalog.sids[c.item_id] for c in candidates))
        scores = sequence_log_probs(self.model, prefix, [self.suffixes[sid] for sid in sids],
                                    self.tokenizer.pad_token_id, self.device, self.options.get("score_batch_size", 2))
        mapping = dict(zip(sids, scores))
        # Colliding SIDs remain distinct items with equal model scores. No target
        # is consulted to select a member; item ID breaks ties reproducibly.
        ordered = sorted(candidates, key=lambda c: (-mapping[self.catalog.sids[c.item_id]], c.item_id))[:limit]
        return RankedList(tuple(ordered), tuple(mapping[self.catalog.sids[c.item_id]] for c in ordered))


class LocalEvidenceProvider:
    def __init__(self, options):
        self.device = options.get("device", "cuda:0")
        self.model, self.tokenizer = load_lm(options["ranker_checkpoint"], self.device)
        self.cost = options.get("ranker_cost", .001)
        self.max_tokens = options.get("max_prompt_tokens", 2048)
        labels = [self.tokenizer.encode(str(i), add_special_tokens=False) for i in range(5)]
        if any(len(ids) != 1 for ids in labels):
            raise ValueError("ranker tokenizer must encode each score label 0..4 as one token")
        self.label_ids = [ids[0] for ids in labels]

    def estimate_cost(self, record):
        return self.cost

    def score(self, record):
        text = ("Rate how well this candidate fits the observed user history and request. "
                "Return one score from 0 to 4; 4 is the strongest match.\n"
                "History (oldest to newest):\n" + "\n".join(record.history_text)
                + "\nRequest: " + record.request_text
                + "\nCandidate: " + record.candidate_text
                + "\nCandidate source: " + ", ".join(record.construction_evidence)
                + "\nScore:")
        ids = self.tokenizer.encode(text, add_special_tokens=False)[-self.max_tokens:]
        inputs = torch.tensor([ids], device=self.device)
        started = time.monotonic()
        with torch.inference_mode():
            logits = self.model(input_ids=inputs, attention_mask=torch.ones_like(inputs)).logits[0, -1, self.label_ids].float()
            value = float((logits.softmax(-1) * torch.arange(5, device=self.device)).sum())
        return value, Cost(units=self.cost, physical_requests=1, input_tokens=len(ids),
                           latency_seconds=time.monotonic() - started)


class ThreeSourceRetriever:
    def __init__(self, options, catalog, contexts):
        self.catalog, self.contexts = catalog, contexts
        self.unit_cost = options.get("retrieval_cost", .001)
        saved = json.loads(Path(options["transitions"]).read_text(encoding="utf-8"))
        if saved.get("fit_split") != "train":
            raise ValueError("transition scores must come from the train split")
        self.transitions = saved["scores"]
        for previous, scores in self.transitions.items():
            if previous not in catalog.candidates or not isinstance(scores, dict):
                raise ValueError("invalid transition source")
            for item_id, value in scores.items():
                if item_id not in catalog.candidates or not math.isfinite(value) or value < 0:
                    raise ValueError("invalid transition destination or score")

    def estimate_cost(self, quotas):
        return self.unit_cost * sum(quotas)

    def retrieve(self, state, signal, quotas, excluded):
        started = time.monotonic()
        history_ids = self.contexts[state.context.request_id][1]
        seen = set(excluded) | set(history_ids)
        similarity = dict(zip(self.catalog.ids, self.catalog.similarities(signal)))
        dense = sorted(self.catalog.ids, key=lambda i: (-similarity[i], i))
        query = terms(" ".join(state.context.history_text) + " " + state.context.text)
        for item_id in dense[:3]:
            query.update(self.catalog.tokens[item_id])
        lexical_scores = {i: len(query & self.catalog.tokens[i]) for i in self.catalog.ids}
        lexical = sorted((i for i in self.catalog.ids if lexical_scores[i]),
                         key=lambda i: (-lexical_scores[i], i))
        transitions = self.transitions.get(history_ids[-1], {}) if history_ids else {}
        sequential = sorted(transitions, key=lambda i: (-transitions[i], -similarity[i], i))
        added, counts = [], []
        for name, order, quota in zip(("dense", "lexical", "sequential"), (dense, lexical, sequential), quotas):
            count = 0
            for item_id in order:
                if count >= quota:
                    break
                if item_id in seen:
                    continue
                seen.add(item_id)
                added.append(replace(self.catalog.candidates[item_id], evidence=(name,)))
                count += 1
            counts.append(count)
        return RetrievalResult(tuple(added), Cost(units=self.estimate_cost(quotas),
                               physical_requests=sum(q > 0 for q in quotas),
                               latency_seconds=time.monotonic() - started), tuple(counts))


def load_components(options, catalog, contexts):
    return MiniOneRec(options, catalog, contexts), OrganizedEvidenceRanker(LocalEvidenceProvider(options))


class PreparedIntegration:
    def __init__(self, options, config):
        unknown = set(options) - OPTION_FIELDS
        if unknown:
            raise ValueError(f"unknown integration options: {sorted(unknown)}")
        for name in ("ranker_cost", "retrieval_cost"):
            value = options.get(name, .001)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name, default in (("max_prompt_tokens", 2048), ("score_batch_size", 2), ("signal_items", 3), ("reward_rollouts", 2)):
            value = options.get(name, default)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.options, self.config = options, config
        self.catalog = Catalog(options["catalog"], config.embedding_dim)
        # Only context files are loaded here. Target files are opened by the
        # training reader or after evaluation trajectories have completed.
        self.contexts, self.split_contexts = {}, {}
        for split, key in (("train", "train_contexts"), ("eval", "eval_contexts")):
            if key in options:
                values = read_contexts(options[key], self.catalog, config)
                for context, ids in values:
                    if context.request_id in self.contexts:
                        raise ValueError("request identities must be unique across splits")
                    self.contexts[context.request_id] = (context, ids)
                self.split_contexts[split] = [context for context, _ in values]
        self.retriever = ThreeSourceRetriever(options, self.catalog, self.contexts)
        self.base, self.ranker = load_components(options, self.catalog, self.contexts)

    def _contexts(self, split):
        if split not in self.split_contexts:
            raise ValueError(f"{split} contexts are required for this command")
        return self.split_contexts[split]

    def _targets(self, split):
        rows = list(jsonl(self.options["train_targets" if split == "train" else "eval_targets"]))
        contexts = self._contexts(split)
        if len(rows) != len(contexts) or [str(x["request_id"]) for x in rows] != [c.request_id for c in contexts]:
            raise ValueError("target identities and order must match the context file")
        if any(set(row) != {"request_id", "item_id"} for row in rows):
            raise ValueError("targets require only request_id and item_id")
        result = [str(row["item_id"]) for row in rows]
        if any(item_id not in self.catalog.candidates for item_id in result):
            raise ValueError("target item missing from catalog")
        return result

    def training_records(self):
        for context, item_id in zip(self._contexts("train"), self._targets("train")):
            yield TrainingRecord(context, self.catalog.candidates[item_id].embedding)

    def reward_examples(self):
        cfg = self.config
        workflow = Workflow(cfg, self.base, self.ranker, self.retriever,
                            Gate(cfg.state_dim, cfg.hidden_dim),
                            JointPolicy(cfg.state_dim, cfg.embedding_dim, cfg.hidden_dim))
        for context, item_id in zip(self._contexts("train"), self._targets("train")):
            for _ in range(self.options.get("reward_rollouts", 2)):
                trajectory = workflow.run(context, force_all=True)
                variants = [trajectory.prefix(i) for i in range(len(trajectory.steps))] + [trajectory]
                for value in variants:
                    ranking = value.item_ids[:5]
                    label = 1. / math.log2(ranking.index(item_id) + 2) if item_id in ranking else 0.
                    yield RewardExample(value, self.catalog.candidates[item_id].embedding, label)

    def evaluation_contexts(self):
        return iter(self._contexts("eval"))

    def evaluation_targets(self):
        return iter(self._targets("eval"))


def build(options, config):
    return PreparedIntegration(options, config)
