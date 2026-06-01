#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

if [[ -f "${repo_root}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${repo_root}/.env"
  set +a
fi

matlab_bin="${MATLAB_BIN:-matlab}"

if [[ -z "${MATLAB_USER_ID:-}" || -z "${MATLAB_PASSWORD:-}" ]]; then
  echo "MATLAB_USER_ID and MATLAB_PASSWORD must be set in .env for online login." >&2
  exit 2
fi

MATLAB_BIN="${matlab_bin}" python3 - <<'PY'
import os
import pexpect
import sys

user = os.environ["MATLAB_USER_ID"]
password = os.environ["MATLAB_PASSWORD"]
matlab_bin = os.environ.get("MATLAB_BIN", "matlab")

cmd = f"{matlab_bin} -licmode onlinelicensing -batch \"disp('ONLINE_AUTH_OK')\""
child = pexpect.spawn("/bin/bash", ["-lc", cmd], encoding="utf-8", timeout=60)
try:
    child.setecho(False)
except Exception:
    pass

status = "UNKNOWN"
sent_user = False
sent_password = False

try:
    while True:
        idx = child.expect([
            r"Please enter your MathWorks Account email address.*:",
            r"(?i)password.*:",
            r"(?i)would you like to retry.*\[n\]",
            r"ONLINE_AUTH_OK",
            r"MathWorks Licensing Error",
            r"(?i)(verification|two[- ]?step|multi[- ]?factor|browser|single sign[- ]?on|sso|code)",
            pexpect.EOF,
            pexpect.TIMEOUT,
        ])
        if idx == 0:
            sent_user = True
            child.sendline(user)
        elif idx == 1:
            sent_password = True
            child.sendline(password)
        elif idx == 2:
            child.sendline("n")
            status = "LOGIN_RETRY_PROMPT"
            break
        elif idx == 3:
            status = "ONLINE_AUTH_OK"
            try:
                child.expect(pexpect.EOF, timeout=20)
            except pexpect.TIMEOUT:
                pass
            break
        elif idx == 4:
            status = "LICENSING_ERROR"
            break
        elif idx == 5:
            status = "INTERACTIVE_SSO_OR_MFA_REQUIRED"
            break
        elif idx == 6:
            status = "EOF"
            break
        elif idx == 7:
            status = "TIMEOUT"
            break
finally:
    if child.isalive():
        child.terminate(force=True)

if status == "ONLINE_AUTH_OK":
    print("MATLAB online licensing authentication OK")
    sys.exit(0)

print(f"MATLAB online licensing authentication failed: {status}", file=sys.stderr)
print(f"sent_user={sent_user}", file=sys.stderr)
print(f"sent_password={sent_password}", file=sys.stderr)
sys.exit(1)
PY
