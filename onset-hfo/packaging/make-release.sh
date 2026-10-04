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
mkdir -p "${STAGE}/docs"
cp docs/INSTALL.md docs/CLINICAL_GUIDE.md docs/AGENT.md docs/LIMITATIONS.md "${STAGE}/docs/"
chmod +x "${STAGE}/install.sh"
tar -C dist -czf "dist/${NAME}.tar.gz" "${NAME}"
rm -rf "${STAGE}"

echo "==> dist/${NAME}.tar.gz"
tar -tzf "dist/${NAME}.tar.gz" | sed 's/^/    /'
echo
echo "Unpack and run:  tar xzf ${NAME}.tar.gz && cd ${NAME} && ./install.sh --with-assistant"
