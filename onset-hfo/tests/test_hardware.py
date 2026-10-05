"""Choosing a model for a machine nobody here owns.

The point of keeping `choose` a pure function of a :class:`Machine` is this
file: a 4 GB laptop GPU, an Apple unified-memory box, a two-card workstation
and a Raspberry Pi are all describable, so the sizing policy can be tested
without any of them and without downloading a byte.

Two properties matter more than any individual choice:

* **Nothing off CUDA is ever quantised.** `bitsandbytes` needs a CUDA device,
  so 4-bit on a MacBook is not a slow configuration, it is an ImportError
  after the download has already finished.
* **Whatever is chosen fits.** The arithmetic is the whole feature; a choice
  that overruns the budget is the out-of-memory crash this module exists to
  prevent.
"""

from __future__ import annotations

import pytest

from onset_agent.hardware import (
    CATALOGUE,
    QUANTIZATIONS,
    WORKABLE_PARAMS_B,
    Choice,
    Gpu,
    Machine,
    ModelSpec,
    NotEnoughRoom,
    bytes_per_parameter,
    choose,
    describe,
    ensure_model,
    footprint_gb,
    probe,
)


def machine(**kwargs) -> Machine:
    """A described machine with torch present, so GPU fields mean something."""
    kwargs.setdefault("torch_available", True)
    kwargs.setdefault("transformers_available", True)
    kwargs.setdefault("free_disk_gb", 500.0)
    kwargs.setdefault("cores", 8)
    return Machine(**kwargs)


def cuda(vram: float, **kwargs) -> Machine:
    kwargs.setdefault("ram_gb", 32.0)
    return machine(accelerator="cuda", gpus=(Gpu(f"{vram:g} GB card", vram),), **kwargs)


def cpu(ram: float, **kwargs) -> Machine:
    return machine(accelerator="cpu", ram_gb=ram, **kwargs)


def mps(ram: float, **kwargs) -> Machine:
    return machine(accelerator="mps", ram_gb=ram, **kwargs)


# -- the two invariants ------------------------------------------------------


ALL_MACHINES = [
    cuda(4.0), cuda(6.0), cuda(8.0), cuda(12.0), cuda(15.8), cuda(24.0),
    cuda(48.0), cuda(80.0),
    cpu(4.0), cpu(8.0), cpu(16.0), cpu(32.0), cpu(64.0),
    mps(8.0), mps(16.0), mps(36.0), mps(64.0),
]


@pytest.mark.parametrize("box", ALL_MACHINES)
def test_quantization_is_never_chosen_without_cuda(box):
    """The invariant worth more than any single choice: `bitsandbytes` has no
    CPU or Metal path, so 4-bit there fails at load, after the download.

    fp32 on CPU and fp16 on Metal are *dtypes*, not bitsandbytes modes -- both
    are the plain unquantised path.
    """
    for route in (None, "transformers", "ollama"):
        picked = choose(box, route=route)
        assert picked.quantization in QUANTIZATIONS
        if box.accelerator != "cuda":
            # q4 is a GGUF served by llama.cpp and runs anywhere; 8-bit and
            # 4-bit are bitsandbytes and do not.
            assert picked.quantization not in ("8bit", "4bit"), \
                f"{picked.quantization} chosen on {box.accelerator} via {route}"
        if picked.route == "transformers":
            if box.accelerator == "cpu":
                assert picked.quantization == "fp32"
            elif box.accelerator == "mps":
                assert picked.quantization == "fp16"
        else:
            assert picked.quantization == "q4"


@pytest.mark.parametrize("box", ALL_MACHINES)
def test_a_choice_either_fits_or_says_it_does_not(box):
    """The invariant is honesty, not universal fit. A 4 GB CPU machine cannot
    hold even 0.6B once fp32 and the OS reserve are accounted for, and saying
    so is the right answer -- what must never happen is a choice that overruns
    the budget without a word."""
    picked = choose(box)
    if picked.fits:
        assert picked.needs_gb <= picked.budget_gb
        assert picked.download_gb <= box.free_disk_gb
    else:
        assert any("does not fit" in warning for warning in picked.warnings), \
            describe(picked, box)


@pytest.mark.parametrize("box", [cuda(8.0), cuda(24.0), cuda(80.0),
                                 cpu(16.0), cpu(32.0), mps(16.0), mps(64.0)])
def test_an_ordinary_machine_gets_something_that_fits(box):
    """The machines a person is actually likely to have."""
    picked = choose(box)
    assert picked.fits, describe(picked, box)
    assert picked.needs_gb <= picked.budget_gb


@pytest.mark.parametrize("box", ALL_MACHINES)
def test_every_choice_explains_itself_and_carries_the_unmeasured_caveat(box):
    picked = choose(box)
    assert picked.reasons
    # The caveat that keeps this honest must be on every single choice.
    assert any("UNMEASURED" in warning for warning in picked.warnings)
    assert "benchmark" in " ".join(picked.warnings)


# -- the policy --------------------------------------------------------------


def test_a_big_card_gets_the_reference_size_unquantised():
    picked = choose(cuda(24.0))
    assert picked.model.params_b >= WORKABLE_PARAMS_B
    assert picked.quantization == "fp16"


def test_a_sixteen_gig_card_takes_the_reference_size_over_a_bigger_quantised_one():
    """The policy decision worth a test of its own.

    On a T4 both 7B-at-8-bit and 14B-at-4-bit fit. 7B is the size this project
    has actually watched call tools correctly; 14B-at-4-bit trades accuracy and
    parameter count at once, with nothing measured on either axis. Preferring
    the bigger number would be buying capability this repository cannot
    account for.
    """
    picked = choose(cuda(15.8))
    assert picked.model.model_id == "Qwen/Qwen2.5-7B-Instruct"
    assert picked.quantization == "8bit"


def test_going_above_the_reference_size_requires_no_quantization():
    """An 80 GB card may have 14B, because it fits at fp16 -- not because it is
    bigger."""
    picked = choose(cuda(80.0))
    assert picked.model.params_b > WORKABLE_PARAMS_B
    assert picked.quantization == "fp16"


def test_fp16_is_preferred_while_there_is_room_for_it():
    """Quantization is a concession, not a default."""
    assert choose(cuda(24.0)).quantization == "fp16"
    assert choose(cuda(12.0)).quantization != "fp16"   # forced


def test_a_small_machine_still_gets_an_answer_with_a_blunt_warning():
    """Refusing to run on a modest laptop would be worse than running badly on
    it -- but the warning has to say the failure mode is tool calling, not
    quality, or the user reads refusals as a broken program."""
    for route in ("ollama", "transformers"):
        picked = choose(cpu(8.0), route=route)
        assert picked.fits
        assert picked.model.params_b < WORKABLE_PARAMS_B
        joined = " ".join(picked.warnings)
        assert "tool calling" in joined
        assert "reference size" in joined


def test_cpu_is_warned_about_speed():
    in_process = choose(cpu(16.0), route="transformers").warnings
    assert any("minutes" in w for w in in_process)
    assert any("--route ollama" in w for w in in_process)   # and told the way out


def test_shared_memory_reserves_room_for_the_operating_system():
    """Unified memory and plain RAM are both shared with the OS and whatever
    else is open; spending the last gigabyte is how a load becomes ten minutes
    of swapping."""
    assert mps(16.0).budget_gb == pytest.approx(12.0)      # 16 - 4
    assert cpu(8.0).budget_gb == pytest.approx(4.0)
    assert cpu(2.0).budget_gb == pytest.approx(0.0)        # never negative
    assert choose(mps(16.0), route="transformers").quantization == "fp16"


def test_two_cards_are_not_added_together():
    """A model too big for one card needs sharding, which this module does not
    set up -- so summing VRAM would promise a configuration nobody built."""
    one = cuda(24.0)
    two = machine(accelerator="cuda", ram_gb=128.0,
                  gpus=(Gpu("a", 24.0), Gpu("b", 24.0)))
    assert two.vram_gb == 24.0
    assert choose(two).model_id == choose(one).model_id


# -- the constraints that are not memory -------------------------------------


def test_a_small_disk_holds_the_choice_down_and_says_so():
    """An 80 GB card running a 1.5B model looks like a bug unless the reason is
    printed next to it."""
    picked = choose(cuda(80.0, free_disk_gb=5.0))
    assert picked.download_gb <= 5.0
    assert any("download does not fit" in reason for reason in picked.reasons)


def test_a_machine_too_small_for_anything_says_so_rather_than_lying():
    tiny = cpu(1.0, free_disk_gb=8.0)
    picked = choose(tiny)
    assert not picked.fits
    assert any("does not fit" in warning for warning in picked.warnings)
    # And it names the smallest model, not the 32B nobody asked for.
    assert picked.model.params_b == min(spec.params_b for spec in CATALOGUE)


def test_without_torch_the_advice_says_it_could_not_check():
    """"No GPU" and "could not look" are different answers, and only one of
    them should make somebody buy a smaller model."""
    blind = Machine(accelerator="cpu", ram_gb=32.0, free_disk_gb=100.0,
                    cores=8, torch_available=False)
    # In-process is the route that needs torch; the served route is covered
    # by test_the_gguf_route_needs_no_torch and must *not* warn about it.
    picked = choose(blind, route="transformers")
    assert any("torch is not installed" in warning for warning in picked.warnings)


# -- asking for a particular model -------------------------------------------


def test_a_requested_model_is_not_silently_downgraded():
    """Substituting a smaller model would have the caller draw conclusions
    about weights they never ran."""
    picked = choose(cuda(6.0), prefer="Qwen/Qwen3-30B-A3B")
    assert picked.model_id == "Qwen/Qwen3-30B-A3B"
    assert not picked.fits
    assert any("does not fit" in warning for warning in picked.warnings)


def test_a_requested_model_can_be_named_by_its_short_name():
    assert choose(cuda(24.0), prefer="Qwen3-4B").model.params_b == 4.0


def test_a_model_outside_the_catalogue_is_allowed_but_flagged():
    picked = choose(cuda(24.0), prefer="someone/their-own-finetune")
    assert picked.model_id == "someone/their-own-finetune"
    assert "size is unknown" in picked.model.note


# -- the arithmetic ----------------------------------------------------------


def test_four_bit_is_not_half_of_eight_bit():
    """Quantised layers carry fp16 scales, and the embeddings usually stay
    unquantised. Treating 4-bit as exactly 0.5 bytes under-counts, which is
    the direction that ends in an out-of-memory crash."""
    assert bytes_per_parameter("4bit") > bytes_per_parameter("8bit") / 2


def test_the_footprint_is_more_than_the_weights():
    spec = ModelSpec("x/y", params_b=7.0, download_gb=14.0)
    assert footprint_gb(spec, "fp16") > 7.0 * 2.0


def test_an_unknown_quantization_is_refused():
    with pytest.raises(ValueError):
        bytes_per_parameter("2bit")


def test_the_download_size_does_not_shrink_with_quantization():
    """Asking for 4-bit saves memory at run time, not bandwidth: the Hub
    stores fp16 weights and bitsandbytes quantises them on load. A download
    estimate that pretended otherwise would under-promise by 3x."""
    spec = next(s for s in CATALOGUE if s.params_b == 7.6)
    on_a_small_card = choose(cuda(8.0), prefer=spec.model_id)
    assert on_a_small_card.quantization == "4bit"
    assert on_a_small_card.download_gb == spec.download_gb


# -- downloading -------------------------------------------------------------


def test_a_download_that_cannot_fit_is_refused_before_it_starts():
    """One subtraction, versus a download that dies at 90% and leaves a
    partial cache."""
    spec = next(s for s in CATALOGUE if s.params_b == 7.6)
    choice = Choice(model=spec, quantization="fp16", device="cuda",
                    needs_gb=18.0, budget_gb=80.0, download_gb=spec.download_gb)
    with pytest.raises(NotEnoughRoom, match="free"):
        ensure_model(choice, machine=cuda(80.0, free_disk_gb=2.0))


def test_a_model_that_does_not_fit_in_memory_is_refused_too():
    spec = max(CATALOGUE, key=lambda s: s.params_b)
    choice = Choice(model=spec, quantization="fp16", device="cuda",
                    needs_gb=76.0, budget_gb=8.0, download_gb=spec.download_gb)
    with pytest.raises(NotEnoughRoom):
        ensure_model(choice, machine=cuda(8.0))


# -- probing this machine ----------------------------------------------------


def test_probing_works_with_nothing_optional_installed():
    """`torch` is in the llm extra, so the probe has to survive without it --
    this test runs on exactly such a machine in CI."""
    found = probe()
    assert found.cores >= 1
    assert found.ram_gb > 0
    assert found.accelerator in ("cuda", "mps", "cpu")
    assert isinstance(found.notes, tuple)


def test_a_passed_machine_is_used_as_is():
    """So every test above describes a machine instead of mocking a probe."""
    described = cuda(24.0)
    assert probe(described) is described


def test_describe_is_readable_and_names_the_revision():
    """A revision the reader can see is the difference between a reproducible
    run and 'some Qwen, some weights, some time last year'."""
    text = describe(choose(cuda(24.0)), cuda(24.0))
    assert "Qwen" in text and "revision" in text
    assert "fits" in text


def test_a_missing_extra_is_its_own_error_naming_the_remedy():
    """The bare `ModuleNotFoundError: No module named 'transformers'` this
    replaces reads like a bug in this project rather than an optional extra
    nobody installed. CI runs without the extra, so this is the real path."""
    from onset_agent.hardware import MissingDependency, auto_backend

    box = cuda(80.0, transformers_available=False)
    with pytest.raises(MissingDependency) as raised:
        auto_backend(machine=box, download=False)
    assert raised.value.module == "transformers"
    assert "onset-hfo[llm]" in str(raised.value)
    # And it offers the route that needs no local weights at all.
    assert "openai_compat" in str(raised.value)


def test_a_missing_hub_is_not_reported_as_a_disk_problem(monkeypatch):
    """`NotEnoughRoom` and `MissingDependency` have completely different
    remedies -- 'free some disk' versus 'pip install'.

    The absence is simulated rather than assumed: this test used to pass only
    because the llm extra was not installed, and silently became a test of the
    network path the moment it was.
    """
    import builtins

    from onset_agent.hardware import MissingDependency

    real_import = builtins.__import__

    def without_hub(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise ImportError("pretend it is absent")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_hub)
    box = cuda(80.0)
    with pytest.raises(MissingDependency) as raised:
        ensure_model(choose(box), machine=box)
    assert raised.value.module == "huggingface_hub"


@pytest.mark.skipif(
    __import__("importlib.util", fromlist=["util"]).find_spec("huggingface_hub") is None,
    reason="needs the llm extra")
def test_a_blocked_hub_says_so_instead_of_leaking_a_library_error():
    """The usual symptom of a proxy or an allowlist is a connection error from
    deep inside `huggingface_hub` that reads like a bug in this project.

    Verified against a real refusal: the container this was written in cannot
    reach huggingface.co, so the 403 path is exercised rather than imagined.
    """
    box = cuda(80.0, free_disk_gb=500.0)
    try:
        ensure_model(choose(box), machine=box)
    except ConnectionError as error:
        assert "huggingface.co" in str(error)
        assert "--backend openai_compat" in str(error)
        assert "local directory" in str(error)
    except Exception as error:                 # a reachable Hub, or no auth
        pytest.skip(f"the Hub behaved differently here: {type(error).__name__}")
    else:
        pytest.skip("the Hub is reachable from this machine")


def test_backend_kwargs_are_what_the_loader_accepts():
    """A float16 dtype is forced on CUDA because most Qwen configs declare
    bfloat16 and a T4 cannot run it -- 'auto' would honour the declaration."""
    from onset_agent.backends import TransformersBackend

    kwargs = choose(cuda(24.0)).backend_kwargs()
    assert kwargs["quantization"] in TransformersBackend.QUANTIZATIONS
    assert kwargs["dtype"] == "float16"
    assert set(kwargs) <= {"model_id", "quantization", "revision", "dtype", "device"}
    # A CPU loads fp32: fp16 matmuls there are slower and often unstable.
    on_cpu = choose(cpu(16.0), route="transformers").backend_kwargs()
    assert on_cpu["dtype"] == "float32"
    assert on_cpu["device"] == "cpu"
    assert on_cpu["quantization"] == "fp16"   # the loader's plain path
    # Metal takes fp16.
    assert choose(mps(32.0), route="transformers").backend_kwargs()["dtype"] == "float16"


def test_ram_is_measurable_without_psutil():
    """psutil is not a declared dependency of this project -- it is only here
    transitively. The naive version returned 0.0 without it, which makes the
    budget 0.0 and reports that nothing will run: "could not measure" and
    "you have none" must not give the same answer."""
    import builtins

    from onset_agent import hardware

    real_import = builtins.__import__

    def no_psutil(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("pretend it is absent")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = no_psutil
    try:
        measured = hardware.total_ram_gb()
    finally:
        builtins.__import__ = real_import
    # This test host is Linux, so /proc/meminfo answers.
    assert measured > 0.5, "fell through every platform route"
    assert measured == pytest.approx(hardware.total_ram_gb(), rel=0.05)


def test_unmeasurable_memory_is_a_note_not_a_verdict_that_nothing_fits():
    blind = Machine(accelerator="cpu", ram_gb=0.0, free_disk_gb=100.0, cores=4)
    assert blind.budget_gb == 0.0
    picked = choose(blind)
    assert not picked.fits          # honest: it cannot promise anything
    assert any("does not fit" in w for w in picked.warnings)


def test_a_cpu_is_sized_for_fp32_not_fp16():
    """The bug `berdakh/ROBT613` caught in this module.

    `transformers` loads fp32 on CPU -- fp16 matmuls on a consumer CPU are
    slower than fp32 and often numerically unstable, so its loader picks
    float32 there, as that workshop's `_resolve_dtype` does. Sizing a CPU
    machine as though it would load fp16 understates the requirement by
    exactly 2x, which is the out-of-memory crash this module exists to
    prevent. Measured before the fix: an 8 GB machine was promised 1.5B at
    "4.0 GB" when the real figure was 7.5 GB.
    """
    spec = ModelSpec("x/y", params_b=1.5, download_gb=3.0)
    assert footprint_gb(spec, "fp32") == pytest.approx(
        2.0 * footprint_gb(spec, "fp16") - 0.6, rel=1e-6)

    picked = choose(cpu(8.0), route="transformers")
    assert picked.quantization == "fp32"
    # Whatever is chosen is costed at 4 bytes per parameter, not 2.
    assert picked.needs_gb == pytest.approx(
        footprint_gb(picked.model, "fp32"), rel=1e-6)
    assert picked.needs_gb > picked.model.params_b * 2.0


def test_bfloat16_only_goes_to_a_card_that_can_run_it():
    """Most Qwen configs declare bfloat16, which needs Ampere (capability 8).
    On a T4 `dtype="auto"` would honour a declaration the card cannot execute,
    so the dtype is resolved here rather than left to the loader."""
    ampere = machine(accelerator="cuda", ram_gb=64.0, cuda_capability=8,
                     gpus=(Gpu("A100", 80.0),))
    turing = machine(accelerator="cuda", ram_gb=12.0, cuda_capability=7,
                     gpus=(Gpu("Tesla T4", 15.8),))
    assert choose(ampere).backend_kwargs()["dtype"] == "bfloat16"
    assert choose(turing).backend_kwargs()["dtype"] == "float16"


def test_the_catalogue_keeps_the_one_size_this_project_has_watched_work():
    """Qwen3 is the current line, but the single qualitative data point this
    repository owns is about Qwen2.5-7B-Instruct. Dropping it would discard the
    only evidence behind the reference size."""
    ids = [spec.model_id for spec in CATALOGUE]
    assert "Qwen/Qwen2.5-7B-Instruct" in ids
    assert any("Qwen3" in model_id for model_id in ids)
    anchor = next(s for s in CATALOGUE if s.model_id == "Qwen/Qwen2.5-7B-Instruct")
    assert anchor.params_b >= WORKABLE_PARAMS_B


# -- the Ollama / GGUF route -------------------------------------------------


def test_off_a_cuda_card_the_default_route_is_a_served_gguf():
    """transformers on a CPU loads fp32 -- twice the memory and a fraction of
    the speed of the same model as a Q4 GGUF -- so it is the wrong tool there,
    and the chooser should say so by default rather than on request."""
    assert choose(cpu(16.0)).route == "ollama"
    assert choose(mps(16.0)).route == "ollama"
    assert choose(cuda(24.0)).route == "transformers"


def test_the_gguf_route_runs_a_bigger_model_on_the_same_laptop():
    """The whole point, as a number: the same 16 GB CPU box."""
    served = choose(cpu(16.0))
    in_process = choose(cpu(16.0), route="transformers")
    assert served.model.params_b > in_process.model.params_b
    assert served.model.params_b >= WORKABLE_PARAMS_B      # the reference size
    assert in_process.model.params_b < WORKABLE_PARAMS_B
    assert served.fits and in_process.fits


def test_the_laptop_default_is_the_projects_long_standing_ollama_default():
    """`backends.DEFAULT_OLLAMA_MODEL` has been qwen2.5:7b-instruct since the
    agent was written. The chooser should land on it, not invent a rival."""
    from onset_agent.backends import DEFAULT_OLLAMA_MODEL

    picked = choose(cpu(16.0))
    assert picked.ollama_tag == DEFAULT_OLLAMA_MODEL
    assert picked.backend_kwargs() == {"model": DEFAULT_OLLAMA_MODEL}


def test_the_gguf_route_needs_no_torch():
    """Ollama owns the runtime, so a machine without the llm extra still gets
    a real recommendation -- and no warning about an extra it does not need."""
    blind = Machine(accelerator="cpu", ram_gb=16.0, free_disk_gb=100.0,
                    cores=8, torch_available=False, transformers_available=False)
    picked = choose(blind)
    assert picked.route == "ollama" and picked.fits
    assert not any("torch is not installed" in w for w in picked.warnings)


def test_every_catalogue_entry_has_an_ollama_tag():
    """A guessed tag that 404s is a worse experience than a short table."""
    for spec in CATALOGUE:
        assert spec.ollama_tag, spec.model_id
        assert ":" in spec.ollama_tag


def test_a_cpu_is_capped_at_the_reference_size_for_speed_not_memory():
    """A 64 GB desktop could hold a 14B at Q4. The agent makes several model
    calls per question and a 14B on CPU answers at a couple of tokens a
    second, so the cap is patience, and it lifts when a model is named."""
    from onset_agent.hardware import CPU_CEILING_PARAMS_B

    big_cpu = choose(cpu(64.0))
    assert big_cpu.model.params_b <= CPU_CEILING_PARAMS_B
    assert any("capped at the reference size" in r for r in big_cpu.reasons)
    named = choose(cpu(64.0), prefer="Qwen3-14B")
    assert named.model.params_b == 14.8 and named.fits
    # Apple silicon is exempt from the speed cap -- Metal is fast enough for
    # the budget to rule -- but the reference-size anchor still applies on
    # every route, so the default is the same 7B and the cap is never cited.
    on_mac = choose(mps(64.0))
    assert on_mac.model.params_b >= WORKABLE_PARAMS_B
    assert not any("capped" in r for r in on_mac.reasons)
    assert choose(mps(64.0), prefer="Qwen3-30B-A3B").fits


def test_the_gguf_download_is_the_quantised_blob_not_the_fp16_weights():
    """Ollama's registry stores the Q4 blob; the Hub stores fp16. Quoting the
    fp16 size for an `ollama pull` would over-promise the wait by 3x."""
    served = choose(cpu(16.0))
    assert served.download_gb == served.model.q4_gb
    assert served.download_gb < served.model.download_gb


def test_a_local_folder_cannot_be_served_by_tag_so_it_goes_in_process():
    """Ollama serves tags, not directories."""
    picked = choose(cpu(16.0), prefer="/some/local/checkpoint")
    assert picked.route == "transformers"
    assert "size is unknown" in picked.model.note


def test_a_model_may_be_named_by_its_ollama_tag():
    picked = choose(cpu(32.0), prefer="qwen3:8b")
    assert picked.model_id == "Qwen/Qwen3-8B"
    assert picked.route == "ollama"


def test_an_unknown_route_is_refused():
    with pytest.raises(ValueError):
        choose(cpu(16.0), route="vllm")


def test_describe_on_the_gguf_route_gives_the_two_commands():
    text = describe(choose(cpu(16.0)), cpu(16.0))
    assert "ollama pull qwen2.5:7b-instruct" in text
    assert "--backend ollama --model qwen2.5:7b-instruct" in text


# -- auto_backend on the Ollama route, with the daemon simulated -------------


def test_auto_without_an_ollama_server_names_the_three_commands(monkeypatch):
    """auto cannot install a daemon, so it must say exactly what to run, and
    offer the in-process route as the slower alternative."""
    from onset_agent import hardware
    from onset_agent.hardware import OllamaNotRunning, auto_backend

    monkeypatch.setattr(hardware, "ollama_status", lambda *a, **k: (False, []))
    with pytest.raises(OllamaNotRunning) as raised:
        auto_backend(machine=cpu(16.0))
    text = str(raised.value)
    assert "ollama serve" in text
    assert "ollama pull qwen2.5:7b-instruct" in text
    assert "--route transformers" in text


def test_auto_with_ollama_up_but_no_model_pulls_it_or_says_so(monkeypatch):
    from onset_agent import hardware
    from onset_agent.hardware import OllamaModelMissing, auto_backend

    monkeypatch.setattr(hardware, "ollama_status", lambda *a, **k: (True, ["llama3:8b"]))
    pulled = []
    monkeypatch.setattr(hardware, "ollama_pull", lambda tag, *a, **k: pulled.append(tag))

    with pytest.raises(OllamaModelMissing) as raised:
        auto_backend(machine=cpu(16.0), download=False)
    assert "ollama pull qwen2.5:7b-instruct" in str(raised.value)
    assert pulled == []

    backend, choice = auto_backend(machine=cpu(16.0), download=True)
    assert pulled == ["qwen2.5:7b-instruct"]
    assert backend.model == "qwen2.5:7b-instruct"


def test_auto_with_the_model_already_pulled_builds_the_backend(monkeypatch):
    from onset_agent import hardware
    from onset_agent.backends import OllamaBackend
    from onset_agent.hardware import auto_backend

    monkeypatch.setattr(hardware, "ollama_status",
                        lambda *a, **k: (True, ["qwen2.5:7b-instruct", "qwen3:8b"]))
    monkeypatch.setattr(hardware, "ollama_pull",
                        lambda *a, **k: pytest.fail("nothing should be pulled"))
    backend, choice = auto_backend(machine=cpu(16.0))
    assert isinstance(backend, OllamaBackend)
    assert backend.model == "qwen2.5:7b-instruct"
    assert backend.base_url.endswith("/v1")
    assert choice.route == "ollama"


def test_the_live_probe_reports_down_cleanly_when_nothing_listens():
    """Against a real closed port: no exception, no hang."""
    from onset_agent.hardware import ollama_status

    up, names = ollama_status("http://127.0.0.1:1", timeout=0.5)
    assert up is False and names == []


def test_the_cli_turns_a_missing_daemon_into_a_message_not_a_traceback(
        monkeypatch, capsys, result, tmp_path):
    """The error text names the three commands; a traceback on top would bury
    them. Exit code 2, the argparse convention for 'cannot proceed'."""
    from onset_agent import cli, hardware

    saved = result.save(tmp_path / "analysis")
    monkeypatch.setattr(hardware, "ollama_status", lambda *a, **k: (False, []))
    code = cli.main(["--results", str(saved), "--backend", "auto",
                     "--question", "Which channel leads?"])
    out = capsys.readouterr().out
    assert code == 2
    assert "cannot start a local model" in out
    assert "ollama serve" in out and "ollama pull" in out
    assert "Traceback" not in out


@pytest.mark.parametrize("value, expected", [
    ("", "http://127.0.0.1:11434"),
    ("127.0.0.1:11434", "http://127.0.0.1:11434"),
    ("0.0.0.0:11434", "http://127.0.0.1:11434"),
    ("http://box:11434", "http://box:11434"),
    ("http://box:11434/", "http://box:11434"),
    (":8080", "http://127.0.0.1:8080"),
    ("box", "http://box:11434"),
])
def test_ollama_host_is_honoured_in_every_spelling_ollama_accepts(value, expected):
    """Ollama's own variable, read at call time: a person who moved their
    server with it expects this project to follow, and a test can point the
    probe at a fake on a free port."""
    from onset_agent.hardware import ollama_url

    assert ollama_url({"OLLAMA_HOST": value}) == expected
