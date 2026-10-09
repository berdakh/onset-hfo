#!/usr/bin/env bash
#
# Build the Linux release bundle: the wheel, the source distribution, and a
# tarball a person can unpack anywhere and run ./install.sh from, with no git
# checkout. The installer detects the wheel next to it and installs that.
#
#   ./packaging/make-release.sh            -> dist/onset-hfo-<version>-linux.tar.gz
#
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"
cd "${HERE}"

VERSION="$("${PY}" -c 'import tomllib; print(tomllib.load(open("pyproject.toml","rb"))["project"]["version"])' 2>/dev/null \
  || "${PY}" -c 'import re; print(re.search(r"^version = \"([^\"]+)\"", open("pyproject.toml").read(), re.M).group(1))')"
NAME="onset-hfo-${VERSION}-linux"
STAGE="dist/${NAME}"

echo "==> building wheel and sdist for ${VERSION}"
"${PY}" -m build --outdir dist . >/dev/null
WHEEL="$(ls dist/onset_hfo-${VERSION}-*.whl | head -n1)"

echo "==> assembling ${STAGE}"
rm -rf "${STAGE}"; mkdir -p "${STAGE}"
cp "${WHEEL}" "${STAGE}/"
cp packaging/install-ubuntu.sh "${STAGE}/install.sh"
cp packaging/onset-review.desktop packaging/onset-review.svg "${STAGE}/"
cp README.md LICENSE "${STAGE}/"
# The guides, and the documents the assistant answers background questions
# from and the study pages draw their glossary tooltips from: install.sh
# copies docs/ next to the environment, and the reviewer looks there.
mkdir -p "${STAGE}/docs"
cp docs/INSTALL.md docs/CLINICAL_GUIDE.md docs/AGENT.md docs/LIMITATIONS.md \
   docs/GLOSSARY.md docs/METHODS.md docs/EVALUATION.md docs/OUTCOME.md docs/ICTAL.md \
   docs/TEMPLATE_MAP.md docs/IMAGING.md "${STAGE}/docs/"
# The study pages: the site's loaders, the committed tables they read, and
# the two figures the Outcome page shows. install.sh copies site/ next to the
# environment and the reviewer finds it there (onset_review.studies.site_root).
mkdir -p "${STAGE}/site/app" "${STAGE}/site/data" "${STAGE}/site/docs/img"
cp app/panels.py "${STAGE}/site/app/"
for table in benchmark outcome stability cohort ictal template hup imaging; do
  cp -R "data/${table}" "${STAGE}/site/data/"
done
cp docs/img/window_stability.png docs/img/run_stability.png "${STAGE}/site/docs/img/"
chmod +x "${STAGE}/install.sh"
tar -C dist -czf "dist/${NAME}.tar.gz" "${NAME}"
rm -rf "${STAGE}"

echo "==> dist/${NAME}.tar.gz"
tar -tzf "dist/${NAME}.tar.gz" | sed 's/^/    /'
echo
echo "Unpack and run:  tar xzf ${NAME}.tar.gz && cd ${NAME} && ./install.sh --with-assistant"
