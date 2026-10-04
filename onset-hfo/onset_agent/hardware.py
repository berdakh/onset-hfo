"""Look at the machine, pick an open-weight model that will actually run on it.

The agent has always been able to drive a local Qwen; what it could not do was
tell somebody *which* Qwen their laptop can hold. The answer is arithmetic over
four numbers -- VRAM, RAM, free disk, and whether CUDA is present -- and getting
it wrong has two unpleasant shapes: a model that dies with a CUDA
out-of-memory page after a 15 GB download, or a model small enough to load and
too small to call a tool, which looks like the software is broken rather than
the model being out of its depth.

So this module separates three questions that are easy to run together and
want different answers:

* **What is this machine?** :func:`probe` -- measured, with every optional
  dependency allowed to be absent.
* **What fits?** :func:`choose` -- a pure function of a :class:`Machine`, so it
  is tested against a 6 GB card and an Apple laptop and a tiny VM without
  owning any of them.
* **Is it any good at this job?** *Not answered here, and not answerable yet.*
  See the warning :func:`choose` attaches to every small model, and
  :mod:`onset_agent.benchmark`, which exists to measure it and has never been
  run under a real model.

That third point is the one worth being blunt about. Fitting in memory and
being competent at tool-constrained evidence work are different properties,
and this project has measured neither for any Qwen size. What follows is sizing
arithmetic plus the one qualitative claim the existing code already documents
(7B calls tools reliably, 1.5B is noticeably worse at it), carried forward as
an expectation and labelled as one.

Nothing here downloads anything. :func:`ensure_model` does, and it refuses
before spending a byte if the disk cannot hold the result.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field, replace

__all__ = [
    "CATALOGUE",
    "Choice",
    "Gpu",
    "Machine",
    "MissingDependency",
    "ModelSpec",
    "NotEnoughRoom",
    "QUANTIZATIONS",
    "bytes_per_parameter",
    "choose",
    "describe",
    "ensure_model",
    "footprint_gb",
    "probe",
    "total_ram_gb",
]

#: What the loader accepts. 8-bit and 4-bit go through ``bitsandbytes``, which
#: needs a CUDA device -- on CPU or Apple silicon they are not slower, they do
#: not work, so `choose` never selects them off CUDA.
#:
#: ``fp32`` is here because it is what a CPU actually loads. fp16 matmuls on a
#: consumer CPU are slower than fp32 and often numerically unstable, so
#: ``transformers`` is run at fp32 there -- which costs *twice* the memory of
#: the fp16 figure. Sizing a CPU machine as though it would load fp16 is the
#: one mistake in this module that produces the exact out-of-memory crash it
#: exists to prevent, so CPU gets its own entry rather than borrowing fp16's.
QUANTIZATIONS = ("fp32", "fp16", "8bit", "4bit")

#: Bytes per parameter once loaded. 4-bit is not 0.5: the quantised layers
#: carry fp16 scales and zero-points, and the embedding and output layers are
#: usually left unquantised, which is where the extra tenth comes from.
_BYTES = {"fp32": 4.0, "fp16": 2.0, "8bit": 1.0, "4bit": 0.6}

#: Reserved for the operating system and whatever else the person is running.
#: Spending a laptop's last gigabyte is how a model load turns into ten
#: minutes of swapping.
_OS_RESERVE_GB = 4.0

#: Multiplier for everything that is not weights: KV cache, activations,
#: CUDA context, the allocator's fragmentation. A short evidence-assistant
#: exchange is not a long-context job, so this is deliberately modest.
_RUNTIME_OVERHEAD = 1.15

#: Fixed cost on top, in GB -- framework, context, tokenizer.
_RUNTIME_FLOOR = 0.6

#: Keep this much of the budget free rather than filling it exactly. A machine
#: driven to its last 200 MB swaps, and on a laptop that means the fan and a
#: ten-minute answer.
_HEADROOM = 0.9


def bytes_per_parameter(quantization: str) -> float:
    """Bytes one parameter occupies when loaded at this quantization."""
    if quantization not in _BYTES:
        raise ValueError(f"quantization must be one of {QUANTIZATIONS}, "
                         f"not {quantization!r}")
    return _BYTES[quantization]


@dataclass(frozen=True)
class Gpu:
    name: str
    vram_gb: float


@dataclass(frozen=True)
class ModelSpec:
    """One candidate, with the two sizes that matter kept apart.

    ``download_gb`` is what crosses the network -- always the fp16 weights,
    because that is what the Hub stores and what ``bitsandbytes`` quantises on
    load. Asking for 4-bit saves memory at run time, not bandwidth, and a
    download estimate that pretended otherwise would under-promise by 3x on
    the model most likely to be chosen for a small card.
    """

    model_id: str
    params_b: float
    download_gb: float
    #: Pinned so that a re-run months later loads the same weights. ``main``
    #: is a moving target, and an unrecorded difference in the model is
    #: exactly the kind this project exists not to have.
    revision: str = "main"
    note: str = ""

    @property
    def family(self) -> str:
        return self.model_id.split("/")[-1]


#: Qwen3 Instruct, smallest first, with the Qwen2.5 7B kept because it is the
#: only size whose tool calling this project has actually watched work.
#:
#: Sizes and the sizing rule of thumb follow `berdakh/ROBT613`'s
#: ``qwen_workshop.config``, which is the workshop this catalogue should agree
#: with: two bytes per parameter at bf16/fp16, about 0.6 at int4 once the
#: unquantised embeddings and the fp16 scales are counted, plus headroom for
#: the KV cache and the framework.
#:
#: Revisions are left at ``main`` rather than invented: this container cannot
#: reach huggingface.co (the gateway refuses CONNECT), so a pinned commit could
#: not be verified, and a made-up SHA is worse than an honest moving
#: reference. :func:`pinned` fills them in from the Hub on a machine that can
#: see it.
CATALOGUE: tuple[ModelSpec, ...] = (
    ModelSpec("Qwen/Qwen3-0.6B", 0.6, 1.4,
              note="runs on a CPU-only laptop; weak at facts, and expect it to "
                   "fail this project's tool contract"),
    ModelSpec("Qwen/Qwen3-1.7B", 1.7, 3.4,
              note="better instruction following than 0.6B, still laptop-sized"),
    ModelSpec("Qwen/Qwen3-4B", 4.0, 8.0,
              note="the sweet spot for a free Colab T4 or an 8 GB card at int4"),
    ModelSpec("Qwen/Qwen2.5-7B-Instruct", 7.6, 15.3,
              note="the reference size for this project: the one whose tool "
                   "calling has been observed to work here"),
    ModelSpec("Qwen/Qwen3-8B", 8.2, 16.4,
              note="good tool calling and RAG quality; needs int4 to fit 8 GB"),
    ModelSpec("Qwen/Qwen3-14B", 14.8, 29.6,
              note="lab-machine territory: a 24 GB card at int8"),
    ModelSpec("Qwen/Qwen3-30B-A3B", 30.5, 61.0,
              note="mixture-of-experts: the memory of a 30B at the speed of a "
                   "3B, so it is worth reaching for when the memory is there"),
)

#: The smallest model this project is willing to describe as a reasonable
#: default. Below it, `choose` still answers -- refusing to run on a small
#: machine would be worse -- but the warning is unambiguous.
WORKABLE_PARAMS_B = 7.0


class NotEnoughRoom(RuntimeError):
    """The chosen model cannot be stored or loaded on this machine."""


class MissingDependency(RuntimeError):
    """The optional ``llm`` extra is needed and is not installed.

    Separate from :class:`NotEnoughRoom` because the remedy is completely
    different -- one is "free some disk", the other "pip install" -- and
    because the bare ``ModuleNotFoundError: No module named 'transformers'``
    this replaces reads like a bug in this project rather than a missing
    optional extra.
    """

    def __init__(self, module: str):
        super().__init__(
            f"{module} is not installed, which the local-model path needs: "
            "pip install 'onset-hfo[llm]'. Alternatively serve the model "
            "elsewhere and use --backend openai_compat, or --backend ollama.")
        self.module = module


@dataclass(frozen=True)
class Machine:
    """What was measured. Every accelerator field is allowed to be empty."""

    cores: int = 1
    ram_gb: float = 0.0
    free_disk_gb: float = 0.0
    gpus: tuple[Gpu, ...] = ()
    #: ``"cuda"``, ``"mps"`` or ``"cpu"``.
    accelerator: str = "cpu"
    #: CUDA compute capability major version. bfloat16 needs Ampere (8.0); an
    #: older card such as a T4 must be given float16 explicitly, because most
    #: Qwen configs declare bfloat16 and ``dtype="auto"`` would honour it.
    cuda_capability: int = 0
    torch_available: bool = False
    transformers_available: bool = False
    hub_available: bool = False
    notes: tuple[str, ...] = ()

    @property
    def vram_gb(self) -> float:
        """The largest single card, not the sum.

        A model that does not fit on one card needs sharding to use two, and
        this module does not set that up -- so adding VRAM across cards would
        promise a configuration nobody built.
        """
        return max((gpu.vram_gb for gpu in self.gpus), default=0.0)

    @property
    def budget_gb(self) -> float:
        """How much memory a model may occupy on this machine.

        On CUDA that is the card. On Apple silicon the GPU shares system
        memory, so it is a fraction of RAM rather than all of it -- the OS and
        everything else still need to live there. On CPU, likewise.
        """
        if self.accelerator == "cuda" and self.vram_gb > 0:
            return self.vram_gb
        # Shared with the OS and everything else the person has open, so the
        # reserve comes off the top before any fraction is taken.
        return max(self.ram_gb - _OS_RESERVE_GB, 0.0)

    @property
    def can_quantize(self) -> bool:
        """4-bit and 8-bit go through bitsandbytes, which requires CUDA."""
        return self.accelerator == "cuda" and self.vram_gb > 0


def footprint_gb(spec: ModelSpec, quantization: str) -> float:
    """Memory the model needs once loaded, weights plus running room."""
    weights = spec.params_b * bytes_per_parameter(quantization)
    return weights * _RUNTIME_OVERHEAD + _RUNTIME_FLOOR


@dataclass(frozen=True)
class Choice:
    """A model, how it will be loaded, and why -- including what it costs."""

    model: ModelSpec
    quantization: str
    device: str
    needs_gb: float
    budget_gb: float
    download_gb: float
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def model_id(self) -> str:
        return self.model.model_id

    @property
    def fits(self) -> bool:
        return self.needs_gb <= self.budget_gb * _HEADROOM

    #: CUDA compute capability of the card this was chosen for, carried so
    #: that `backend_kwargs` can tell an Ampere card from a T4.
    cuda_capability: int = 0

    def backend_kwargs(self) -> dict:
        """Exactly what :class:`onset_agent.backends.TransformersBackend` wants.

        The dtype is resolved here rather than left to ``"auto"``, because
        "auto" honours whatever the model config declares -- and most Qwen
        configs declare bfloat16, which a T4 cannot run and a CPU runs slowly
        and unstably. Same rule as `berdakh/ROBT613`'s
        ``qwen_workshop.loading._resolve_dtype``.
        """
        if self.device == "cuda":
            dtype = "bfloat16" if self.cuda_capability >= 8 else "float16"
        elif self.device == "mps":
            dtype = "float16"
        else:
            dtype = "float32"
        # fp32 is a dtype, not a bitsandbytes mode: the loader takes it as the
        # plain (unquantised) path, which is what "fp16" means to it too.
        quantization = "fp16" if self.quantization == "fp32" else self.quantization
        return {"model_id": self.model.model_id,
                "quantization": quantization,
                "revision": self.model.revision,
                "dtype": dtype,
                "device": None if self.device == "cuda" else self.device}


def probe(machine: Machine | None = None) -> Machine:
    """Measure this machine. Nothing optional is required to be installed.

    ``psutil`` is a hard dependency of this project, so RAM and core count are
    always real. ``torch`` is in the ``llm`` extra: without it there is no way
    to see a GPU from Python, and rather than claim the machine has none this
    says so in ``notes`` -- "no CUDA" and "cannot tell" are different answers
    and only one of them should make somebody buy a smaller model.
    """
    if machine is not None:
        return machine

    cores = os.cpu_count() or 1
    try:
        import psutil
        cores = psutil.cpu_count(logical=False) or cores
    except Exception:                                     # pragma: no cover
        pass
    ram_gb = total_ram_gb()

    free_disk_gb = 0.0
    try:
        free_disk_gb = shutil.disk_usage(_cache_root()).free / 2 ** 30
    except Exception:                                     # pragma: no cover
        pass

    gpus: tuple[Gpu, ...] = ()
    accelerator, notes, capability = "cpu", [], 0
    torch_available = transformers_available = hub_available = False
    try:
        import torch
        torch_available = True
        if torch.cuda.is_available():
            accelerator = "cuda"
            gpus = tuple(
                Gpu(torch.cuda.get_device_name(i),
                    torch.cuda.get_device_properties(i).total_memory / 2 ** 30)
                for i in range(torch.cuda.device_count()))
            capability = torch.cuda.get_device_capability(0)[0]
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            accelerator = "mps"
            notes.append("Apple GPU shares system memory, so the budget is a "
                         "share of RAM rather than dedicated VRAM")
    except ImportError:
        notes.append("torch is not installed (pip install 'onset-hfo[llm]'), so "
                     "no GPU could be detected -- this is 'cannot tell', not "
                     "'no GPU'")
    except Exception as error:                            # pragma: no cover
        notes.append(f"torch is installed but could not be queried: {error}")

    if ram_gb <= 0:
        notes.append("system memory could not be measured on this platform, so "
                     "no sizing advice is possible -- pass a size explicitly "
                     "with --model, or install psutil")

    for module, flag in (("transformers", "transformers"), ("huggingface_hub", "hub")):
        try:
            __import__(module)
            if flag == "transformers":
                transformers_available = True
            else:
                hub_available = True
        except ImportError:
            notes.append(f"{module} is not installed")

    return Machine(cores=cores, ram_gb=ram_gb, free_disk_gb=free_disk_gb,
                   gpus=gpus, accelerator=accelerator,
                   cuda_capability=capability,
                   torch_available=torch_available,
                   transformers_available=transformers_available,
                   hub_available=hub_available, notes=tuple(notes))


def total_ram_gb() -> float:
    """System memory in GB, without requiring anything that is not installed.

    ``psutil`` is not a declared dependency of this project -- it is present
    here only transitively -- and on a machine without it the naive version of
    this returned 0.0, which makes the budget 0.0 and reports that nothing
    will run. "I could not measure your RAM" and "you have no RAM" must not
    produce the same answer, so each platform's own interface is tried before
    giving up: ``/proc/meminfo`` on Linux, ``sysctl`` on macOS and
    ``GlobalMemoryStatusEx`` on Windows.

    Returns 0.0 only when every route failed, and `probe` turns that into a
    note rather than a verdict.
    """
    try:
        import psutil
        return psutil.virtual_memory().total / 2 ** 30
    except Exception:
        pass

    try:                                    # any POSIX: Linux, macOS, BSD
        return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
                / 2 ** 30)
    except (ValueError, OSError, AttributeError):
        pass

    try:                                    # Windows
        import ctypes

        class _Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = _Status()
        status.dwLength = ctypes.sizeof(_Status)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return float(status.ullTotalPhys) / 2 ** 30
    except Exception:
        pass

    return 0.0


def _cache_root() -> str:
    """Where the Hub will put the weights, which is the disk that must hold them."""
    for variable in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        value = os.environ.get(variable)
        if value:
            return value
    return os.path.expanduser("~")


def _quantizations_for(machine: Machine) -> tuple[str, ...]:
    """Largest-memory option first, so fp16 wins when there is room for it.

    fp16 before 4-bit on purpose: quantization costs accuracy, so it is a
    concession to a small card rather than a default.
    """
    if machine.can_quantize:
        return ("fp16", "8bit", "4bit")
    if machine.accelerator == "mps":
        # Metal runs fp16 happily; bitsandbytes has no Metal path.
        return ("fp16",)
    # CPU: fp32 is what transformers loads, so fp32 is what must be budgeted.
    return ("fp32",)


def choose(machine: Machine | None = None, *,
           catalogue: tuple[ModelSpec, ...] = CATALOGUE,
           prefer: str | None = None) -> Choice:
    """The best model from the catalogue this machine can actually run.

    "Best" is not "biggest", and the difference is the whole policy:

    1. **Prefer unquantised room over raw size.** Among models that fit at
       fp16, take the largest. Quantization trades accuracy for room, so it is
       a concession to a small card rather than a way to claim a bigger number.
    2. **Anchor on the reference size.** If nothing at or above
       :data:`WORKABLE_PARAMS_B` fits at fp16, but the reference model fits
       once quantised, take *that* rather than a larger model at a more
       aggressive quantization. On a 16 GB card the alternatives are 7B at
       8-bit and 14B at 4-bit; 7B is the size whose tool-calling this project
       has actually watched work, and 14B-at-4-bit is an unmeasured trade on
       both axes at once. Reaching for it would be buying a bigger parameter
       count with accuracy this project cannot account for.
    3. **Only then go smaller**, taking the largest model that fits at all.

    ``prefer`` pins a ``model_id`` and asks only *how* to load it; if it does
    not fit at any quantization the returned :class:`Choice` says so --
    ``fits`` is False and the warnings explain -- rather than quietly
    substituting something smaller. A caller that asked for 14B and silently
    got 1.5B would draw conclusions about the wrong model.

    Never returns 8-bit or 4-bit off CUDA, because ``bitsandbytes`` cannot run
    there: that is not a slow configuration, it is a failed import at load
    time, and discovering it after a 15 GB download is a bad afternoon.
    """
    machine = probe(machine)
    options = _quantizations_for(machine)

    candidates = list(catalogue)
    if prefer:
        wanted = [spec for spec in catalogue if spec.model_id == prefer
                  or spec.family.lower() == prefer.lower()]
        if not wanted:
            wanted = [ModelSpec(prefer, params_b=0.0, download_gb=0.0,
                                note="not in the catalogue, so its size is unknown")]
        candidates = wanted

    biggest_first = sorted(candidates, key=lambda s: -s.params_b)

    def usable(spec: ModelSpec, quantization: str) -> Choice | None:
        choice = _build(machine, spec, quantization, options)
        return choice if choice is not None and choice.fits and \
            spec.download_gb <= machine.free_disk_gb else None

    # 1. the largest model that needs no quantization at all
    if "fp16" in options:
        for spec in biggest_first:
            found = usable(spec, "fp16")
            if found is not None and spec.params_b >= WORKABLE_PARAMS_B:
                return found

    # 2. the reference size, quantised if that is what it takes. Smallest
    #    model at or above the reference, not largest: if 7B will not fit
    #    quantised then nothing above it will either.
    at_or_above = sorted((spec for spec in candidates
                          if spec.params_b >= WORKABLE_PARAMS_B),
                         key=lambda s: s.params_b)
    for spec in at_or_above[:1]:
        for quantization in options:
            found = usable(spec, quantization)
            if found is not None:
                return found

    # 3. whatever does fit
    for spec in biggest_first:
        for quantization in options:
            found = usable(spec, quantization)
            if found is not None:
                return found

    # Nothing fits. Report the smallest candidate so the message names a real
    # shortfall rather than the 32B nobody asked for.
    smallest = biggest_first[-1]
    return _build(machine, smallest, options[-1], options)


def _build(machine: Machine, spec: ModelSpec, quantization: str,
           options: tuple[str, ...]) -> Choice:
    """One fully-explained candidate at one quantization."""
    return _assemble(machine, spec, quantization,
                     footprint_gb(spec, quantization), machine.budget_gb,
                     machine.free_disk_gb, machine.accelerator, options)


def _assemble(machine: Machine, spec: ModelSpec, quantization: str,
              needs: float, budget: float, disk: float, device: str,
              options: tuple[str, ...] = QUANTIZATIONS) -> Choice:
    reasons, warnings = [], []

    if device == "cuda":
        reasons.append(f"CUDA device with {machine.vram_gb:.1f} GB on the largest card")
    elif device == "mps":
        reasons.append(f"Apple GPU sharing {machine.ram_gb:.1f} GB of system memory")
    else:
        reasons.append(f"no GPU detected; {machine.ram_gb:.1f} GB of RAM on "
                       f"{machine.cores} core(s)")
    reasons.append(f"{spec.family} at {quantization} needs about {needs:.1f} GB "
                   f"of a {budget:.1f} GB budget")
    if quantization in ("8bit", "4bit"):
        reasons.append(f"{quantization} chosen because fp16 would not fit")
    elif quantization == "fp32":
        # Not a fallback from fp16 -- it costs twice as much. It is simply the
        # only dtype a CPU should be given, so saying "fp16 would not fit"
        # here would be backwards.
        reasons.append("fp32 because that is what a CPU loads, not as a "
                       "fallback: it costs twice fp16, and is budgeted so")
    if machine.accelerator == "cpu":
        reasons.append("no CUDA device, so bitsandbytes cannot quantise here; "
                       "a CPU loads fp32, which is twice the memory of the "
                       "fp16 figure usually quoted")
    elif machine.accelerator == "mps":
        reasons.append("Metal runs fp16, but bitsandbytes has no Metal path, "
                       "so quantization is not available here")

    # Say when the disk, not the memory, is what held the choice down --
    # otherwise an 80 GB card running a 1.5B model looks like a bug.
    bigger = [other for other in CATALOGUE
              if other.params_b > spec.params_b
              and footprint_gb(other, quantization) <= budget * _HEADROOM]
    if bigger and any(other.download_gb > disk for other in bigger):
        blocked = min(bigger, key=lambda s: s.params_b)
        reasons.append(
            f"memory would allow {blocked.family}, but its {blocked.download_gb:.1f} GB "
            f"download does not fit the {disk:.1f} GB free on the cache disk")

    if spec.note:
        reasons.append(spec.note)

    if needs > budget * _HEADROOM:
        warnings.append(
            f"{spec.family} does not fit: it needs about {needs:.1f} GB and the "
            f"budget is {budget:.1f} GB. Nothing here will run it; use a smaller "
            "model, or a hosted server with --backend openai_compat.")
    if spec.download_gb > disk:
        warnings.append(
            f"the download is about {spec.download_gb:.1f} GB and only "
            f"{disk:.1f} GB is free on the model cache disk")
    if spec.params_b and spec.params_b < WORKABLE_PARAMS_B:
        warnings.append(
            f"{spec.family} is below the {WORKABLE_PARAMS_B:g}B this project "
            "treats as the reference size. Small models fail at tool calling "
            "rather than answering badly -- expect refusals and malformed "
            "calls, which the guards will catch and report as failures.")
    if device == "cpu":
        warnings.append(
            "on CPU expect tens of seconds to minutes per answer; the agent "
            "makes several model calls per question.")
    if not machine.torch_available:
        warnings.append(
            "torch is not installed, so this is sizing advice rather than a "
            "plan that has been checked against a real device: "
            "pip install 'onset-hfo[llm]'")
    warnings.append(
        "WHETHER THIS SIZE IS GOOD ENOUGH AT THIS TASK IS UNMEASURED. This "
        "chooses a model that fits and loads. onset_agent.benchmark exists to "
        "measure tool-use competence by model and quantization and has not yet "
        "been run under a real model.")

    return Choice(model=spec, quantization=quantization, device=device,
                  needs_gb=round(needs, 2), budget_gb=round(budget, 2),
                  download_gb=spec.download_gb, reasons=tuple(reasons),
                  warnings=tuple(warnings),
                  cuda_capability=machine.cuda_capability)


def describe(choice: Choice, machine: Machine | None = None) -> str:
    """A paragraph a person can read before committing to a download."""
    lines = [f"model        {choice.model_id}",
             f"revision     {choice.model.revision}",
             f"load as      {choice.quantization} on {choice.device}",
             f"needs        {choice.needs_gb:.1f} GB of a {choice.budget_gb:.1f} GB budget",
             f"download     {choice.download_gb:.1f} GB",
             f"fits         {'yes' if choice.fits else 'NO'}"]
    lines.append("")
    lines += [f"  - {reason}" for reason in choice.reasons]
    if choice.warnings:
        lines.append("")
        lines += [f"  ! {warning}" for warning in choice.warnings]
    if machine is not None and machine.notes:
        lines.append("")
        lines += [f"  . {note}" for note in machine.notes]
    return "\n".join(lines)


def ensure_model(choice: Choice, *, machine: Machine | None = None,
                 progress: bool = True) -> str:
    """Download the weights if they are not cached, and return the local path.

    Refuses before spending a byte when the disk cannot hold the result: a
    download that dies at 90% has cost the user their bandwidth and left a
    partial cache, and the check is one subtraction.

    A blocked or absent network is reported as itself. This matters more than
    it looks: the usual symptom is a connection error from deep inside
    ``huggingface_hub`` that reads like a bug in this project.
    """
    machine = probe(machine)
    if not choice.fits:
        raise NotEnoughRoom(
            f"{choice.model_id} needs about {choice.needs_gb:.1f} GB and the "
            f"budget on this machine is {choice.budget_gb:.1f} GB. "
            + " ".join(choice.warnings[:1]))
    if choice.download_gb > machine.free_disk_gb:
        raise NotEnoughRoom(
            f"{choice.model_id} is about {choice.download_gb:.1f} GB and only "
            f"{machine.free_disk_gb:.1f} GB is free on {_cache_root()}. Free "
            "some space or set HF_HOME to a larger disk.")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise MissingDependency("huggingface_hub") from error

    try:
        return snapshot_download(
            choice.model_id, revision=choice.model.revision,
            # The loader reads safetensors; the .bin duplicates double the
            # download for nothing.
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
            tqdm_class=None if progress else _Silent)
    except Exception as error:
        raise ConnectionError(
            f"could not fetch {choice.model_id} from huggingface.co: {error}. "
            "If this machine is behind a proxy or an allowlist, the Hub may be "
            "blocked -- download the model elsewhere and point --model at the "
            "local directory, or serve it with --backend openai_compat."
        ) from error


class _Silent:                                            # pragma: no cover
    """A tqdm stand-in for callers that do not want a progress bar."""

    def __init__(self, *args, **kwargs):
        self.n = 0

    def update(self, _n=1):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def auto_backend(machine: Machine | None = None, *, prefer: str | None = None,
                 download: bool = True):
    """Probe, choose, fetch and build a ready backend. The one-call path.

    Kept here rather than in :mod:`onset_agent.backends` so that importing the
    backends does not drag in a hardware probe, and so that `choose` stays a
    pure function somebody can call just to ask what *would* happen.
    """
    machine = probe(machine)
    if not machine.transformers_available:
        # Checked here rather than left to the import below, so the caller gets
        # the remedy instead of "No module named 'transformers'".
        raise MissingDependency("transformers")
    from onset_agent.backends import TransformersBackend

    choice = choose(machine, prefer=prefer)
    if download:
        ensure_model(choice, machine=machine)
    return TransformersBackend(**choice.backend_kwargs()), choice


def pinned(catalogue: tuple[ModelSpec, ...] = CATALOGUE) -> tuple[ModelSpec, ...]:
    """Resolve every ``main`` in the catalogue to the Hub's current commit.

    Run on a machine that can reach huggingface.co; paste the result back into
    `CATALOGUE`. Pinning matters for the same reason the rest of this project
    records its versions: a benchmark that cannot say which weights it ran is
    not a benchmark.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    out = []
    for spec in catalogue:
        info = api.model_info(spec.model_id)
        out.append(replace(spec, revision=info.sha))
    return tuple(out)
