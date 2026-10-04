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
    CPU or Metal path, so 4-bit there fails at load, after the download."""
    picked = choose(box)
    if box.accelerator != "cuda":
        assert picked.quantization == "fp16", \
            f"{picked.quantization} chosen on {box.accelerator}"
    assert picked.quantization in QUANTIZATIONS


@pytest.mark.parametrize("box", ALL_MACHINES)
def test_whatever_is_chosen_actually_fits(box):
    picked = choose(box)
    assert picked.fits, describe(picked, box)
    assert picked.needs_gb <= picked.budget_gb
    assert picked.download_gb <= box.free_disk_gb


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
    picked = choose(cpu(8.0))
    assert picked.fits
    assert picked.model.params_b < WORKABLE_PARAMS_B
    joined = " ".join(picked.warnings)
    assert "tool calling" in joined
    assert "reference size" in joined


def test_cpu_is_warned_about_speed():
    assert any("CPU" in w or "minutes" in w for w in choose(cpu(16.0)).warnings)


def test_apple_silicon_gets_a_share_of_ram_not_all_of_it():
    """Unified memory is shared with the OS and everything else; spending all
    of it is how a Mac starts swapping."""
    box = mps(16.0)
    assert box.budget_gb == pytest.approx(9.6)
    assert choose(box).quantization == "fp16"


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
    picked = choose(blind)
    assert any("torch is not installed" in warning for warning in picked.warnings)


# -- asking for a particular model -------------------------------------------


def test_a_requested_model_is_not_silently_downgraded():
    """Substituting a smaller model would have the caller draw conclusions
    about weights they never ran."""
    picked = choose(cuda(6.0), prefer="Qwen/Qwen2.5-32B-Instruct")
    assert picked.model_id == "Qwen/Qwen2.5-32B-Instruct"
    assert not picked.fits
    assert any("does not fit" in warning for warning in picked.warnings)


def test_a_requested_model_can_be_named_by_its_short_name():
    assert choose(cuda(24.0), prefer="Qwen2.5-3B-Instruct").model.params_b == 3.0


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
    spec = next(s for s in CATALOGUE if s.params_b == 32.8)
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


def test_a_missing_hub_is_not_reported_as_a_disk_problem():
    """`NotEnoughRoom` and `MissingDependency` have completely different
    remedies -- 'free some disk' versus 'pip install'."""
    from onset_agent.hardware import MissingDependency

    box = cuda(80.0)
    with pytest.raises(MissingDependency) as raised:
        ensure_model(choose(box), machine=box)
    assert raised.value.module == "huggingface_hub"


def test_backend_kwargs_are_what_the_loader_accepts():
    """A float16 dtype is forced on CUDA because most Qwen configs declare
    bfloat16 and a T4 cannot run it -- 'auto' would honour the declaration."""
    from onset_agent.backends import TransformersBackend

    kwargs = choose(cuda(24.0)).backend_kwargs()
    assert kwargs["quantization"] in TransformersBackend.QUANTIZATIONS
    assert kwargs["dtype"] == "float16"
    assert set(kwargs) <= {"model_id", "quantization", "revision", "dtype", "device"}
    # Off CUDA the dtype is left to the loader, and the device is explicit.
    on_cpu = choose(cpu(16.0)).backend_kwargs()
    assert on_cpu["dtype"] == "auto"
    assert on_cpu["device"] == "cpu"


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
