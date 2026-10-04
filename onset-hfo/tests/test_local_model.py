"""The local-model path, run against a real `transformers` load.

Every other test of the open-weight path uses a fake backend, because weights
are large and the Hub is not always reachable. That leaves the part most likely
to break untested: whether `hardware.choose` produces kwargs the real loader
accepts, whether the chat template applies with tools attached, and whether a
model that answers badly is *caught* rather than believed.

So this builds a real Qwen2 architecture at toy size -- two layers, 64 hidden,
a 512-word vocabulary, about 107k random parameters -- saves it to disk with a
tool-aware chat template, and drives the shipped code through it. The text it
generates is meaningless. That is deliberate and is the point of the last test
in this file: an incompetent model must fail the guards, not the loader.

Skipped entirely without the ``llm`` extra, which CI does not install.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="needs the llm extra")
pytest.importorskip("transformers", reason="needs the llm extra")

from onset_agent.backends import TransformersBackend, make_backend  # noqa: E402
from onset_agent.hardware import Machine, choose  # noqa: E402

#: Small enough to build in a second, real enough to exercise Qwen2's own
#: attention and the loader's dtype handling.
TINY = dict(vocab_size=512, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=2048, tie_word_embeddings=True)

#: Qwen's template shape, trimmed to what this project uses: system/user/
#: assistant turns plus a tools block. `apply_chat_template(..., tools=...)`
#: silently drops the tools if the template has no `tools` branch, which would
#: make a tool-calling test pass while telling the model nothing.
CHAT_TEMPLATE = (
    "{% if tools %}<|im_start|>system\n# Tools\n"
    "{% for t in tools %}{{ t | tojson }}\n{% endfor %}<|im_end|>\n{% endif %}"
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n"
    "{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory) -> str:
    """A real Qwen2 checkpoint on disk, with random weights."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    out = tmp_path_factory.mktemp("tiny_qwen")
    words = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "system", "user",
             "assistant", "tool", "name", "arguments", "top_channels"]
    vocab = {word: index for index, word in enumerate(words)}
    for index in range(len(words), TINY["vocab_size"]):
        vocab[f"tok{index}"] = index

    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<|endoftext|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.WordPiece()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, chat_template=CHAT_TEMPLATE,
        bos_token="<|endoftext|>", eos_token="<|im_end|>",
        pad_token="<|endoftext|>", unk_token="<|endoftext|>",
    ).save_pretrained(out)
    Qwen2ForCausalLM(Qwen2Config(**TINY)).save_pretrained(
        out, safe_serialization=True)
    return str(out)


TOOLS = [{"type": "function",
          "function": {"name": "top_channels", "description": "rank channels",
                       "parameters": {"type": "object",
                                      "properties": {"k": {"type": "integer"}},
                                      "required": []}}}]

MESSAGES = [{"role": "system", "content": "You rank channels."},
            {"role": "user", "content": "Which channel leads?"}]


def _cpu_box() -> Machine:
    return Machine(accelerator="cpu", ram_gb=16.0, free_disk_gb=50.0, cores=4,
                   torch_available=True, transformers_available=True)


# -- the seam that was never tested -----------------------------------------


def test_the_chosen_kwargs_are_accepted_by_the_real_loader(tiny_model):
    """`hardware.choose` hands `backend_kwargs()` straight to
    `TransformersBackend`. Nothing checked that the two agreed, and a wrong
    keyword there fails only after the weights have downloaded."""
    choice = choose(_cpu_box(), prefer=tiny_model)
    backend = TransformersBackend(**choice.backend_kwargs())
    assert backend.model_id == tiny_model
    assert backend.is_language_model


def test_a_cpu_really_does_load_float32(tiny_model):
    """The 2x sizing bug, pinned against the loader rather than arithmetic.

    `transformers` loads float32 on CPU, so a CPU model costs four bytes per
    parameter and must be budgeted at four. This asserts the dtype that
    actually ends up on the tensors, not the one requested.
    """
    import torch

    choice = choose(_cpu_box(), prefer=tiny_model)
    assert choice.quantization == "fp32"
    assert choice.backend_kwargs()["dtype"] == "float32"
    backend = TransformersBackend(**choice.backend_kwargs())
    assert next(backend.model.parameters()).dtype == torch.float32


def test_the_chat_template_is_applied_with_the_tools_attached(tiny_model):
    """A template without a `tools` branch drops them silently, which would
    leave the model told nothing while every test still passed."""
    backend = TransformersBackend(model_id=tiny_model, dtype="float32",
                                  device="cpu", max_new_tokens=4)
    prompt = backend.tokenizer.apply_chat_template(
        MESSAGES, tools=TOOLS, tokenize=False, add_generation_prompt=True)
    assert "top_channels" in prompt, "the tools never reached the prompt"
    assert "# Tools" in prompt
    assert prompt.rstrip().endswith("assistant")


def test_generation_returns_an_assistant_message(tiny_model):
    """Random weights, so the text is meaningless -- what matters is that the
    round trip produces the shape the agent loop consumes."""
    backend = TransformersBackend(model_id=tiny_model, dtype="float32",
                                  device="cpu", max_new_tokens=4)
    reply = backend.chat(MESSAGES, TOOLS)
    assert isinstance(reply.content, str)
    assert isinstance(reply.tool_calls, list)


def test_make_backend_builds_it_by_name(tiny_model):
    backend = make_backend("transformers", model=tiny_model, dtype="float32",
                           device="cpu", max_new_tokens=4)
    assert isinstance(backend, TransformersBackend)


def test_quantization_is_refused_rather_than_attempted_off_cuda(tiny_model):
    """bitsandbytes has no CPU path. `choose` never asks for this, but a person
    passing --model and a quantization by hand should get a refusal and not a
    half-loaded model."""
    with pytest.raises(Exception):   # noqa: B017 - bitsandbytes' own error
        TransformersBackend(model_id=tiny_model, quantization="4bit",
                            device="cpu")


# -- the point of the exercise ----------------------------------------------


def test_a_model_that_answers_badly_is_caught_not_believed(tiny_model, store):
    """The designed failure mode, demonstrated against a real model.

    A 107k-parameter model with random weights is the worst possible agent. It
    must not produce a confident rate: the citation and number checks have to
    reject it, and the CLI has to say so. "Declines to answer" is usable; "a
    confidently wrong ripple rate" is not, and this is the test that tells
    those apart on a real generation path rather than a scripted stub.
    """
    from onset_agent.agent import OnsetAgent

    backend = TransformersBackend(model_id=tiny_model, dtype="float32",
                                  device="cpu", max_new_tokens=8)
    answer = OnsetAgent(store, backend, max_steps=2).ask(
        "Which channel had the highest ripple rate?")

    assert answer.refused, "a random-weight model produced an unrefused answer"
    text = f"{answer.text} {getattr(answer, 'reason', '')}".lower()
    assert "verification" in text or "stand behind" in text
