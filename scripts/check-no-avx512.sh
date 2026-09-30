#!/usr/bin/env bash
# Fail if a binary contains AVX-512 instructions (#150, #152).
#
# ggml builds with -march=native unless GGML_NATIVE=OFF, so an unpinned
# release inherits the runner's ISA. v0.1.26 shipped AVX-512 this way and
# SIGILLed on every CPU without it. AVX-512 shows up as %zmm registers,
# %k0-7 mask registers, the EVEX-only %xmm16-31/%ymm16-31, embedded
# broadcasts ({1toN}), or EVEX-only mnemonics on low ymm/xmm registers.
#
# Usage: scripts/check-no-avx512.sh [binary...]
#        (default: target/release/whisrsd target/release/whisrs)
set -euo pipefail

pattern='%zmm|%k[0-7]\b|%[xy]mm(1[6-9]|2[0-9]|3[01])\b|\{1to[0-9]+\}|\bvpternlog|\bvpermt2|\bvpro[lr][dq]\b'

[ "$#" -gt 0 ] || set -- target/release/whisrsd target/release/whisrs

dis=$(mktemp)
trap 'rm -f "$dis"' EXIT

failed=0
for bin in "$@"; do
  [ -f "$bin" ] || { echo "::error::$bin not found"; exit 1; }
  # Separate step so an objdump failure fails the check instead of reading as 0.
  objdump -d "$bin" > "$dis"
  # grep exits 1 on zero matches; anything above that is a real error.
  rc=0
  count=$(grep -cE "$pattern" "$dis") || rc=$?
  [ "$rc" -le 1 ] || { echo "::error::grep failed on $bin"; exit 1; }
  echo "AVX-512 instructions in $bin: $count"
  if [ "$count" -ne 0 ]; then
    echo "::error::$bin requires AVX-512 and would SIGILL on most x86_64 CPUs (#150)"
    failed=1
  fi
done
exit "$failed"
