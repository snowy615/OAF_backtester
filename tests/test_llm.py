"""The Claude stages, exercised against a scripted stand-in for the SDK client."""

from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip("anthropic")

from oaf.data.ingest import ColumnMapping
from oaf.llm.client import LLMError
from oaf.llm.data_mapper import infer_mapping_llm
from oaf.llm.idea_to_signal import build_system_prompt, refine_spec, structure_idea

FIELDS = ["close", "returns", "volume"]


class FakeClient:
    def __init__(self, *outputs, stop_reason="end_turn"):
        self.outputs, self.calls, self.stop_reason = list(outputs), [], stop_reason
        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=self._parse))

    def _parse(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return SimpleNamespace(parsed_output=self.outputs.pop(0), stop_reason=self.stop_reason, content=[{"type": "text", "text": "{}"}])


def test_system_prompt_documents_every_operator():
    from oaf.dsl import OPS

    prompt = build_system_prompt()
    assert all(f"- {op.signature}" in prompt for op in OPS.values())


def test_idea_is_structured_and_request_is_well_formed(momentum_spec):
    client = FakeClient(momentum_spec)
    spec = structure_idea("winners keep winning", FIELDS, ["all", "demo_index"], client=client)
    assert spec.idea == "winners keep winning" and spec.signal == momentum_spec.signal
    call = client.calls[0]
    assert call["model"] == "claude-opus-5" and call["thinking"] == {"type": "adaptive"}
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "demo_index" in call["messages"][0]["content"] and "winners keep winning" in call["messages"][0]["content"]


def test_invalid_draft_is_sent_back_for_repair(momentum_spec):
    broken = momentum_spec.model_copy(update={"signal": "rank(ts_mean(sentiment, lookback) + skip + q)"})
    client = FakeClient(broken, momentum_spec)
    spec = structure_idea("x", FIELDS, client=client)
    assert spec.signal == momentum_spec.signal and len(client.calls) == 2
    repair = client.calls[1]["messages"]
    assert [m["role"] for m in repair] == ["user", "assistant", "user"] and "sentiment" in repair[-1]["content"]


def test_gives_up_after_max_repairs(momentum_spec):
    broken = momentum_spec.model_copy(update={"signal": "rank(sentiment)"})
    with pytest.raises(LLMError, match="still fails validation"):
        structure_idea("x", FIELDS, client=FakeClient(broken, broken), max_repairs=1)


def test_refusal_is_surfaced(momentum_spec):
    with pytest.raises(LLMError, match="declined"):
        structure_idea("x", FIELDS, client=FakeClient(momentum_spec, stop_reason="refusal"))


def test_refine_keeps_the_original_idea(momentum_spec):
    momentum_spec.idea = "original pitch"
    revised = momentum_spec.with_params(rebalance="monthly")
    revised.idea = ""
    client = FakeClient(revised)
    out = refine_spec(momentum_spec, "make it monthly", FIELDS, client=client)
    assert out.rebalance == "monthly" and out.idea == "original pitch"
    assert '"lookback"' in client.calls[0]["messages"][0]["content"]


def test_data_mapper_rejects_hallucinated_columns():
    df = pd.DataFrame({"Dt": ["2024-01-02"], "Px": [1.0]})
    good = ColumnMapping(date_col="Dt", close_col="Px", ticker_value="X")
    assert infer_mapping_llm(df, "x.csv", client=FakeClient(good)) == good
    with pytest.raises(ValueError, match="not a column"):
        infer_mapping_llm(df, "x.csv", client=FakeClient(ColumnMapping(date_col="Date", close_col="Px")))
