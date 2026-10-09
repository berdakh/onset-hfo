#!/usr/bin/env bash
#
# Install Onset Review on Ubuntu.
#
# What this does, in order:
#   1. installs the system libraries Qt needs (apt, needs sudo once),
#   2. builds a virtual environment under ~/.local/share/onset-review,
#   3. installs this checkout into it with the `review` extra,
#   4. puts an `onset-review` command on PATH and an entry in the
#      applications menu.
#
# What it deliberately does not do: touch the system Python, install anything
# outside the user's home beyond the apt packages, or fetch any recording. A
# reviewer gets a working window and no data; `--with-sample` adds one minute
# of one public patient so the first launch has something to open.
#
# With --with-assistant it also sets up the local language model: installs
# Ollama if it is missing, gives it a context window the agent fits in, picks
# the Qwen that suits this machine (the same choice `onset-agent --hardware`
# prints), pulls it (a few GB, once), and records the choice so the reviewer's
# assistant panel opens on it. No model is fetched without that flag.
#
# Usage:
#   ./packaging/install-ubuntu.sh [--with-sample] [--with-assistant]
#                                 [--assistant-model TAG] [--no-pull]
#                                 [--prefix DIR] [--no-apt] [--dry-run]
#   ./packaging/install-ubuntu.sh --with-assistant --skip-install   # add it later
#   ./packaging/install-ubuntu.sh --uninstall
#
set -euo pipefail

PREFIX="${HOME}/.local/share/onset-review"
BIN_DIR="${HOME}/.local/bin"
DESKTOP_DIR="${HOME}/.local/share/applications"
ICON_DIR="${HOME}/.local/share/icons/hicolor/scalable/apps"
WITH_SAMPLE=0
WITH_ASSISTANT=0
ASSISTANT_MODEL=""
PULL=1
SKIP_INSTALL=0
DRY_RUN=0
RUN_APT=1
UNINSTALL=0
OLLAMA_CTX="${OLLAMA_CONTEXT_LENGTH:-8192}"
OLLAMA_API="http://127.0.0.1:11434"

# In the checkout this script lives in packaging/, so the project is one
# level up. In an unpacked release bundle it sits beside the wheel, and the
# bundle directory itself is the project.
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SELF_DIR}/pyproject.toml" ]] || ls "${SELF_DIR}"/onset_hfo-*.whl >/dev/null 2>&1; then
  HERE="${SELF_DIR}"
else
  HERE="$(cd "${SELF_DIR}/.." && pwd)"
fi

# Qt 6 on a server or minimal desktop install is missing most of these. The
# list is what PySide6's QtWidgets, its xcb platform plugin and its offscreen
# plugin actually dlopen; a missing one surfaces as "could not load the Qt
# platform plugin xcb", which says nothing about which library is absent.
APT_PACKAGES=(
  python3-venv python3-pip
  libegl1 libgl1 libglib2.0-0
  libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 libxcb-keysyms1
  libxcb-shape0 libxcb-xinerama0 libxcb-randr0 libxcb-render-util0
  libdbus-1-3 libfontconfig1 libfreetype6
)

say()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m==>\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m==>\033[0m %s\n' "$*" >&2; exit 1; }
# Every command that changes the machine goes through run(), so --dry-run can
# print the plan instead of executing it.
run()  { if [[ "${DRY_RUN}" == "1" ]]; then printf '    $ %s\n' "$*"; else "$@"; fi; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-sample) WITH_SAMPLE=1; shift ;;
    --with-assistant)  WITH_ASSISTANT=1; shift ;;
    --assistant-model) ASSISTANT_MODEL="$2"; WITH_ASSISTANT=1; shift 2 ;;
    --no-pull)         PULL=0; shift ;;
    --skip-install)    SKIP_INSTALL=1; shift ;;
    --dry-run)         DRY_RUN=1; shift ;;
    --no-apt)      RUN_APT=0; shift ;;
    --prefix)      PREFIX="$2"; shift 2 ;;
    --uninstall)   UNINSTALL=1; shift ;;
    -h|--help)     sed -n '2,30p' "$0"; exit 0 ;;
    *)             die "unknown option: $1 (try --help)" ;;
  esac
done

if [[ "${UNINSTALL}" == "1" ]]; then
  say "Removing Onset Review"
  rm -rf "${PREFIX}"
  rm -f "${BIN_DIR}/onset-review" \
        "${DESKTOP_DIR}/onset-review.desktop" \
        "${ICON_DIR}/onset-review.svg"
  command -v update-desktop-database >/dev/null \
    && update-desktop-database "${DESKTOP_DIR}" 2>/dev/null || true
  rm -f "${ONSET_REVIEW_CONFIG_DIR:-${XDG_CONFIG_HOME:-${HOME}/.config}/onset-review}/assistant.json"
  if command -v ollama >/dev/null; then
    say "Ollama and its models are left in place: other software may use them.
    To remove them: sudo systemctl disable --now ollama; sudo rm -rf /usr/local/bin/ollama /usr/share/ollama"
  fi
  say "Removed. The apt packages and any cached recordings were left alone."
  exit 0
fi

# -- 0. sanity ---------------------------------------------------------------
# Either a source checkout (pyproject.toml) or an unpacked release bundle
# (a wheel next to this script, from packaging/make-release.sh).
BUNDLE_WHEEL="$(ls "${HERE}"/onset_hfo-*.whl 2>/dev/null | head -n1 || true)"
[[ -f "${HERE}/pyproject.toml" || -n "${BUNDLE_WHEEL}" ]] \
  || die "run this from the onset-hfo checkout or an unpacked release bundle (nothing to install at ${HERE})"
[[ "${DRY_RUN}" == "0" ]] || say "Dry run: printing what would be done, changing nothing"

PYTHON="$(command -v python3 || true)"
[[ -n "${PYTHON}" ]] || die "python3 not found"
PY_VERSION="$("${PYTHON}" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
"${PYTHON}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' \
  || die "Python 3.10 or newer is required; this is ${PY_VERSION}"
say "Python ${PY_VERSION} at ${PYTHON}"

# -- 1. system libraries -----------------------------------------------------
if [[ "${SKIP_INSTALL}" == "1" ]]; then
  say "Skipping the package install (--skip-install)"
  RUN_APT=0
fi
if [[ "${RUN_APT}" == "1" ]]; then
  say "Installing the system libraries Qt needs (sudo)"
  run sudo apt-get update -qq
  run sudo apt-get install -y --no-install-recommends "${APT_PACKAGES[@]}"
else
  say "Skipping apt (--no-apt)"
fi

# -- 2. virtual environment --------------------------------------------------
VENV_PY="${PREFIX}/venv/bin/python"
if [[ "${SKIP_INSTALL}" == "0" ]]; then
  say "Building the environment in ${PREFIX}"
  run mkdir -p "${PREFIX}"
  [[ -x "${VENV_PY}" ]] || run "${PYTHON}" -m venv "${PREFIX}/venv"
  run "${VENV_PY}" -m pip install --quiet --upgrade pip wheel

  # -- 3. the package --------------------------------------------------------
  if [[ -n "${BUNDLE_WHEEL}" && ! -f "${HERE}/pyproject.toml" ]]; then
    say "Installing the release wheel with the desktop reviewer (a few minutes)"
    run "${VENV_PY}" -m pip install --quiet "${BUNDLE_WHEEL}[review]"
    if [[ -d "${HERE}/site" ]]; then
      # The study pages' loaders and tables, which the wheel does not carry.
      say "Installing the study pages' tables in ${PREFIX}/site"
      run rm -rf "${PREFIX}/site"
      run cp -R "${HERE}/site" "${PREFIX}/site"
    fi
    if [[ -d "${HERE}/docs" ]]; then
      say "Installing the documents the assistant answers background questions from in ${PREFIX}/docs"
      run rm -rf "${PREFIX}/docs"
      run cp -R "${HERE}/docs" "${PREFIX}/docs"
    fi
  else
    say "Installing onset-hfo with the desktop reviewer (a few minutes)"
    run "${VENV_PY}" -m pip install --quiet -e "${HERE}[review]"
  fi

  if [[ "${DRY_RUN}" == "0" ]]; then
    say "Checking that Qt can start"
    if ! QT_QPA_PLATFORM=offscreen "${VENV_PY}" -c \
          'from qtpy.QtWidgets import QApplication; QApplication([]); print("Qt ok")'; then
      die "Qt could not start. Re-run without --no-apt, or install the libraries listed
in APT_PACKAGES at the top of this script."
    fi
  fi
else
  [[ -x "${VENV_PY}" || "${DRY_RUN}" == "1" ]] \
    || die "--skip-install, but there is no environment at ${PREFIX}; run without it first"
fi

# -- 4. command, menu entry, icon -------------------------------------------
say "Installing the launcher"
run mkdir -p "${BIN_DIR}" "${DESKTOP_DIR}" "${ICON_DIR}"
if [[ "${DRY_RUN}" == "0" ]]; then

# A wrapper rather than a symlink into the venv, so the working directory is
# the checkout: that is where `artifacts/data` lives, and a reviewer launching
# from the applications menu starts in their home directory otherwise and sees
# an empty recording list.
cat > "${BIN_DIR}/onset-review" <<WRAPPER
#!/usr/bin/env bash
# Generated by packaging/install-ubuntu.sh. Re-run that script to update.
cd "${HERE}" || exit 1
exec "${PREFIX}/venv/bin/onset-review" "\$@"
WRAPPER
chmod +x "${BIN_DIR}/onset-review"

# In a bundle the desktop file and icon sit next to this script.
ASSETS="${HERE}/packaging"; [[ -d "${ASSETS}" ]] || ASSETS="${HERE}"
install -m 644 "${ASSETS}/onset-review.svg" "${ICON_DIR}/onset-review.svg"
sed "s|^Exec=onset-review|Exec=${BIN_DIR}/onset-review|" \
    "${ASSETS}/onset-review.desktop" > "${DESKTOP_DIR}/onset-review.desktop"
chmod 644 "${DESKTOP_DIR}/onset-review.desktop"
command -v update-desktop-database >/dev/null \
  && update-desktop-database "${DESKTOP_DIR}" 2>/dev/null || true
else
  printf '    $ write %s/onset-review (wrapper), the menu entry and the icon\n' "${BIN_DIR}"
fi

# -- 5. a recording to open --------------------------------------------------
if [[ "${WITH_SAMPLE}" == "1" ]]; then
  say "Fetching one minute of sub-01 from OpenNeuro ds003498"
  if ! (cd "${HERE}" && "${VENV_PY}" -m onset_hfo.cli fetch \
          --subject sub-01 --t-start 0 --t-stop 60); then
    SAMPLE_FAILED=1
    warn "the fetch failed -- its own error is above this line. The software is
installed and works; there is just nothing cached to open yet."
  fi
fi

# -- 6. the assistant: a local model the panel opens on ----------------------
#
# Ollama rather than in-process transformers, deliberately. On a machine
# without an NVIDIA card transformers loads fp32 -- twice the memory and a
# fraction of the speed of the same model as a Q4 GGUF -- and needs two
# gigabytes of torch wheels to do it. Ollama runs the quantised model on CPU or
# GPU alike, is what the assistant panel has always documented, and is what
# `onset-agent --hardware` recommends off CUDA. The in-process route stays
# available to anyone who wants it: pip install 'onset-hfo[llm]'.
ASSISTANT_TAG=""
ASSISTANT_OK=0
if [[ "${WITH_ASSISTANT}" == "1" ]]; then
  say "Setting up the assistant (a local Qwen under Ollama)"

  # 6a. Ollama itself. The official script installs /usr/local/bin/ollama and,
  #     where systemd exists, an `ollama` service running as its own user.
  if ! command -v ollama >/dev/null; then
    say "Installing Ollama (official script from ollama.com; sudo once)"
    run bash -c 'curl -fsSL https://ollama.com/install.sh | sh'
  else
    say "Ollama is already installed: $(command -v ollama)"
  fi

  # 6b. A context window the agent fits in. Its opening turn is the system
  #     prompt plus the tool schemas -- a few thousand tokens -- and Ollama's
  #     default of 2048 rejects it before the first answer. Set it where the
  #     server will actually read it: the service's environment.
  HAVE_UNIT=0
  if command -v systemctl >/dev/null \
     && systemctl list-unit-files --type=service 2>/dev/null | grep -q '^ollama\.service'; then
    HAVE_UNIT=1
  fi
  if [[ "${HAVE_UNIT}" == "1" ]]; then
    say "Giving the Ollama service a ${OLLAMA_CTX}-token context (systemd drop-in; sudo)"
    run sudo mkdir -p /etc/systemd/system/ollama.service.d
    run bash -c "printf '[Service]\nEnvironment=\"OLLAMA_CONTEXT_LENGTH=${OLLAMA_CTX}\"\n' \
      | sudo tee /etc/systemd/system/ollama.service.d/onset-review.conf >/dev/null"
    run sudo systemctl daemon-reload
    run sudo systemctl enable --now ollama
    run sudo systemctl restart ollama
  else
    warn "No systemd service for Ollama here; starting it in the background with
OLLAMA_CONTEXT_LENGTH=${OLLAMA_CTX}. It will not survive a reboot -- start it again with:
    OLLAMA_CONTEXT_LENGTH=${OLLAMA_CTX} ollama serve"
    run bash -c "OLLAMA_CONTEXT_LENGTH=${OLLAMA_CTX} nohup ollama serve >/dev/null 2>&1 &"
  fi

  # 6c. Wait for the API, but not forever.
  if [[ "${DRY_RUN}" == "0" ]]; then
    for _ in $(seq 1 40); do
      curl -fsS --max-time 1 "${OLLAMA_API}/api/tags" >/dev/null 2>&1 && break
      sleep 0.5
    done
  fi

  # 6d. Which model. The same decision `onset-agent --hardware` prints, made
  #     by the installed code so the two cannot disagree.
  if [[ -n "${ASSISTANT_MODEL}" ]]; then
    ASSISTANT_TAG="${ASSISTANT_MODEL}"
    say "Model: ${ASSISTANT_TAG} (as asked)"
  elif [[ -x "${VENV_PY}" ]]; then
    ASSISTANT_TAG="$(cd "${HERE}" && "${VENV_PY}" -c \
      'from onset_agent.hardware import choose; print(choose(route="ollama").ollama_tag)' 2>/dev/null || true)"
    [[ -n "${ASSISTANT_TAG}" ]] || ASSISTANT_TAG="qwen2.5:7b-instruct"
    say "Model: ${ASSISTANT_TAG} (chosen for this machine; see 'onset-agent --hardware')"
  else
    ASSISTANT_TAG="qwen2.5:7b-instruct"
    say "Model: ${ASSISTANT_TAG} (default; the real choice is made at install time)"
  fi

  # 6e. Pull it. The Ollama registry stores the quantised file, so this is the
  #     Q4 size -- about 4.6 GB for the 7B -- not the fp16 weights.
  if [[ "${PULL}" == "1" ]]; then
    say "Pulling ${ASSISTANT_TAG} (a few GB, once; Ctrl-C and re-run with --no-pull to skip)"
    run ollama pull "${ASSISTANT_TAG}"
  else
    say "Not pulling (--no-pull); run 'ollama pull ${ASSISTANT_TAG}' when ready"
  fi

  # 6f. Record the choice where the reviewer's assistant panel reads it.
  if [[ -x "${VENV_PY}" || "${DRY_RUN}" == "1" ]]; then
    run bash -c "cd '${HERE}' && '${VENV_PY}' -c \"from onset_review.assistant_config import write_defaults; print(write_defaults('ollama', '${ASSISTANT_TAG}', '${OLLAMA_API}/v1'))\""
  fi

  # 6g. Verify, and say exactly what is and is not in place.
  if [[ "${DRY_RUN}" == "0" ]]; then
    if curl -fsS --max-time 2 "${OLLAMA_API}/api/tags" 2>/dev/null | grep -q "\"${ASSISTANT_TAG}\""; then
      ASSISTANT_OK=1
    fi
  fi
fi

CACHED=0
if [[ "${DRY_RUN}" == "0" && -x "${VENV_PY}" ]]; then
  CACHED="$(cd "${HERE}" && "${VENV_PY}" -c \
    'from onset_review.session import cached_windows; print(len(cached_windows()))' \
    2>/dev/null || echo 0)"
fi

say "Done."
echo

# The PATH note comes first. Printing "Launch: onset-review" and only then
# that the directory is not on PATH means the first thing the reader tries is
# the thing that cannot work -- which is exactly what happens when the install
# line ends in `&& onset-review`.
LAUNCH="onset-review"
if ! printf '%s' ":${PATH}:" | grep -q ":${BIN_DIR}:"; then
  LAUNCH="${BIN_DIR}/onset-review"
  warn "${BIN_DIR} is not on your PATH, so plain 'onset-review' will not be
found in this shell. Either use the full path below, or add this to ~/.bashrc
and open a new terminal:"
  echo "    export PATH=\"${BIN_DIR}:\${PATH}\""
  echo
fi

echo "  Launch:      ${LAUNCH}          (or 'Onset Review' in the menu)"
if [[ "${WITH_ASSISTANT}" == "1" ]]; then
  if [[ "${ASSISTANT_OK}" == "1" ]]; then
    echo "  Assistant:   ${ASSISTANT_TAG} via Ollama, ${OLLAMA_CTX}-token context -- the panel opens on it"
  elif [[ "${DRY_RUN}" == "1" ]]; then
    echo "  Assistant:   would be ${ASSISTANT_TAG} via Ollama (dry run)"
  else
    warn "Assistant: ${ASSISTANT_TAG} is not reported by Ollama at ${OLLAMA_API} yet.
    Check: curl ${OLLAMA_API}/api/tags      then: ollama pull ${ASSISTANT_TAG}
    The reviewer still works; its assistant panel will say when the model is missing."
  fi
fi
echo "  Cached:      ${CACHED} window(s) ready to open offline"
echo "  List them:   ${LAUNCH} --list"
echo "  Straight in: ${LAUNCH} --subject sub-01 --window 0 60 --expert"
echo
if [[ "${CACHED}" == "0" ]]; then
  # Never advise re-running the flag that has just failed.
  if [[ "${SAMPLE_FAILED:-0}" == "1" ]]; then
    warn "Nothing is cached, because the fetch above did not succeed. Once the
network is available, run:"
  else
    warn "Nothing is cached yet. Re-run with --with-sample, or:"
  fi
  echo "    cd ${HERE} && ${VENV_PY} -m onset_hfo.cli fetch --subject sub-01 --t-start 0 --t-stop 60"
  echo
  echo "  You can also open a recording of your own without any of this:"
  echo "    ${LAUNCH} --open /path/to/recording.edf"
fi
echo "Research tool — not a medical device. See docs/LIMITATIONS.md."
