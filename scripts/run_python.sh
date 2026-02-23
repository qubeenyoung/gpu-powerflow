#!/usr/bin/env bash
# --------------------------------------------
# Run PYPOWER my_runpf() over .mat case files
# --------------------------------------------
set -euo pipefail

# ====== [설정: 필요시 수정] ======
OUTPUT_DIR="runs/pf_logs"           # 로그 저장 경로
PYTHON_BIN="python3"                 # 파이썬 실행기
PY_SCRIPT_IMPORT="python.my_pypower.my_runpf"  # 모듈 경로 (패키지 기준)
DATASET_DIR="datasets/pf_dataset"    # .mat 케이스 디렉터리
N_RANDOM_DEFAULT=5                   # 무작위 선택 기본 개수

LOG_PF=1
LOG_NEWTONPF=1
# =================================

usage() {
  cat <<'USAGE'
사용법:
  ./run_pf.sh                # DATASET_DIR에서 무작위 케이스 N개 실행
  ./run_pf.sh -n 10          # 무작위 10개 실행
  ./run_pf.sh case9.mat ...  # 인자로 준 케이스들만 실행
  ./run_pf.sh -l list.txt    # list.txt에 있는 케이스 목록 실행 (한 줄에 하나)

옵션:
  -n N       무작위 선택 개수 (기본값: 5)
  -l FILE    실행할 케이스 목록 파일(.mat 파일명 또는 경로)
  -h         도움말
설명:
  • 출력은 케이스별로 ${OUTPUT_DIR}/<케이스이름>.log 로 저장됩니다.
  • 케이스 경로를 절대경로로 주지 않으면 ${DATASET_DIR} 아래에서 찾습니다.
  • 파이썬 파일을 수정하지 않고, my_runpf(casedata) 를 직접 호출합니다.
USAGE
}

# 스크립트 위치 기준으로 PYTHONPATH 보정 (repo 루트에서 실행하지 않아도 동작)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PYTHONPATH:-.}"

N_RANDOM="$N_RANDOM_DEFAULT"
LIST_FILE=""
ARGS=()

# 옵션 파싱
while (( "$#" )); do
  case "$1" in
    -n)
      shift
      [[ $# -ge 1 ]] || { echo "에러: -n 옵션에는 정수가 필요합니다."; exit 1; }
      N_RANDOM="$1"
      shift
      ;;
    -l)
      shift
      [[ $# -ge 1 ]] || { echo "에러: -l 옵션에는 파일 경로가 필요합니다."; exit 1; }
      LIST_FILE="$1"
      shift
      ;;
    -h|--help)
      usage; exit 0;;
    -*)
      echo "알 수 없는 옵션: $1"
      usage; exit 1;;
    *)
      ARGS+=("$1"); shift;;
  esac
done

# 케이스 목록 만들기
declare -a CASE_FILES=()

to_abs_path() {
  local x="$1"
  if [[ -f "$x" ]]; then
    # 이미 경로로 존재
    readlink -f "$x"
  elif [[ -f "${DATASET_DIR}/$x" ]]; then
    readlink -f "${DATASET_DIR}/$x"
  else
    echo "경고: 케이스를 찾을 수 없음: $x" >&2
    echo ""
  fi
}

if [[ ${#ARGS[@]} -gt 0 ]]; then
  # 인자로 받은 항목 사용
  for a in "${ARGS[@]}"; do
    # 확장자 보정 (.mat 빠뜨린 경우)
    if [[ "$a" != *.mat ]]; then a="${a}.mat"; fi
    p="$(to_abs_path "$a")"
    [[ -n "$p" ]] && CASE_FILES+=("$p")
  done
elif [[ -n "$LIST_FILE" ]]; then
  # 목록 파일에서 읽기
  [[ -f "$LIST_FILE" ]] || { echo "에러: 목록 파일이 없습니다: $LIST_FILE"; exit 1; }
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    # 공백 줄/주석 무시
    [[ "$line" =~ ^# ]] && continue
    x="$line"
    if [[ "$x" != *.mat ]]; then x="${x}.mat"; fi
    p="$(to_abs_path "$x")"
    [[ -n "$p" ]] && CASE_FILES+=("$p")
  done < "$LIST_FILE"
else
  # 무작위 선택
  mapfile -t ALL < <(ls -1 "${DATASET_DIR}"/*.mat 2>/dev/null | sort || true)
  if [[ ${#ALL[@]} -eq 0 ]]; then
    echo "에러: ${DATASET_DIR} 아래에 .mat 케이스가 없습니다."; exit 1
  fi
  # shuf로 N개 추출
  mapfile -t CASE_FILES < <(printf "%s\n" "${ALL[@]}" | shuf -n "$N_RANDOM")
fi

# 실행 대상 확인
if [[ ${#CASE_FILES[@]} -eq 0 ]]; then
  echo "에러: 실행할 케이스가 없습니다."; exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "총 ${#CASE_FILES[@]}개 케이스 실행:"
for f in "${CASE_FILES[@]}"; do
  echo " - $f"
done
echo

# 각 케이스 실행
RET_CODE=0
for CASE in "${CASE_FILES[@]}"; do
  BASE="$(basename "$CASE" .mat)"
  LOG_PATH="${OUTPUT_DIR}/${BASE}.log"

  echo "[실행] ${BASE} -> ${LOG_PATH}"

  # 파이썬 파일을 수정하지 않고 직접 my_runpf(casedata) 호출
  # 표준출력·표준에러 모두 로그로 저장
  "${PYTHON_BIN}" - <<PYCODE > "${LOG_PATH}" 2>&1
from ${PY_SCRIPT_IMPORT} import my_runpf
results, success = my_runpf(r"${CASE}", log_pf=${LOG_PF}, log_newtonpf=${LOG_NEWTONPF})
# 종료코드를 표시하려면 아래 프린트가 아니라, 파이썬 종료코드로 내보내야 함.
# 여기서는 my_runpf가 내부적으로 printpf를 호출하므로 로그에 결과가 저장됨.
PYCODE

  # 실행 상태 표시 (파이썬 쪽에서 예외 시 bash가 set -e로 중단됨)
  echo "[완료] ${BASE}"
done

echo
echo "모든 실행이 완료되었습니다. 로그: ${OUTPUT_DIR}"
