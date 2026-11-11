#!/usr/bin/env bash
# Upload notebooks/kaggle_train_150.ipynb to Kaggle and start a background
# "Save & Run All (Commit)" run on a T4 with Internet on.
#
# Needs: the Kaggle CLI (`pip install kaggle`) and an API token from
# https://www.kaggle.com/settings -> API -> Create New Token, saved as
# ~/.kaggle/kaggle.json (chmod 600) or exported as KAGGLE_USERNAME/KAGGLE_KEY.
# The account must be phone-verified, or Kaggle silently runs without Internet
# and the git clone fails.
#
# Usage:  scripts/kaggle_push.sh [kernel-slug]      (default: swinunetr-spleen-150-epochs)
# Status: kaggle kernels status <user>/<slug>
# Output: kaggle kernels output <user>/<slug> -p <dir>
set -euo pipefail

SLUG="${1:-swinunetr-spleen-150-epochs}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

USER_NAME="${KAGGLE_USERNAME:-}"
if [[ -z "$USER_NAME" && -f "$HOME/.kaggle/kaggle.json" ]]; then
  USER_NAME="$(python3 -c 'import json,os;print(json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))["username"])')"
fi
[[ -n "$USER_NAME" ]] || { echo "No Kaggle credentials: see the header of $0" >&2; exit 1; }

BRANCH="$(python3 - "$ROOT/notebooks/kaggle_train_150.ipynb" <<'PY'
import json, re, sys
src = ''.join(''.join(c['source']) for c in json.load(open(sys.argv[1]))['cells'])
print(re.search(r"BRANCH = '([^']+)'", src).group(1))
PY
)"
if ! git -C "$ROOT" ls-remote --exit-code --heads origin "$BRANCH" >/dev/null; then
  echo "Branch '$BRANCH' (BRANCH in the notebook) is not on origin; push it first." >&2
  exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cp "$ROOT/notebooks/kaggle_train_150.ipynb" "$TMP/"
cat > "$TMP/kernel-metadata.json" <<JSON
{
  "id": "$USER_NAME/$SLUG",
  "title": "$SLUG",
  "code_file": "kaggle_train_150.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": []
}
JSON

kaggle kernels push -p "$TMP" --accelerator NvidiaTeslaT4
echo "Started. Check with:  kaggle kernels status $USER_NAME/$SLUG"
echo "Web page:             https://www.kaggle.com/code/$USER_NAME/$SLUG"
