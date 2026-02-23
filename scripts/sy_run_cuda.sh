#!/usr/bin/env bash
# 소영CUDA: Run CUDA NewtonPF over .npy/.npz case files
# --------------------------------------------
set -euo pipefail

# ====== [설정: 필요시 수정] ======
OUTPUT_DIR="runs/cuda_pf_logs"        # 로그 저장 경로
CUDA_BIN="build/core/sy_test_cuda_newtonpf"   # CUDA 실행 파일 경로
DATASET_DIR="datasets/nr_dataset"     # 소영CUDA: nr_dataset 디렉터리 (.npy/.npz 형식)
N_RANDOM_DEFAULT=5                    # 무작위 선택 기본 개수
# =================================

usage() {
  cat <<'USAGE'
사용법:
  ./sy_run_cuda.sh                # DATASET_DIR에서 무작위 케이스 N개 실행
  ./sy_run_cuda.sh -n 10          # 무작위 10개 실행
  ./sy_run_cuda.sh case9 ...      # 인자로 준 케이스들만 실행
  ./sy_run_cuda.sh -l list.txt    # list.txt에 있는 케이스 목록 실행

옵션:
  -n N       무작위 선택 개수 (기본값: 5)
  -l FILE    실행할 케이스 목록 파일
  -h         도움말

설명:
  • CUDA NewtonPF 실행 파일을 사용하여 power flow 계산을 수행합니다.
  • 출력은 케이스별로 ${OUTPUT_DIR}/<케이스이름>/ 디렉터리에 저장됩니다.
  • 먼저 './scripts/sy_build.sh'로 BUILD_CUDA=ON 설정하여 빌드가 필요합니다.
  • 케이스 이름은 디렉터리명입니다 (예: pglib_opf_case14_ieee)
USAGE
}

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

# CUDA 실행 파일 존재 확인
if [[ ! -f "$CUDA_BIN" ]]; then
  echo "에러: CUDA 실행 파일을 찾을 수 없습니다: $CUDA_BIN"
  echo "먼저 './scripts/sy_build.sh'로 BUILD_CUDA=ON 설정하여 빌드하세요."
  exit 1
fi

# 케이스 목록 만들기 (소영CUDA: 디렉터리 기반)
declare -a CASE_FILES=()

if [[ ${#ARGS[@]} -gt 0 ]]; then
  # 인자로 받은 항목 사용
  for a in "${ARGS[@]}"; do
    # 디렉터리 존재 확인
    if [[ -d "${DATASET_DIR}/$a" ]]; then
      CASE_FILES+=("$a")
    else
      echo "경고: 케이스를 찾을 수 없음: ${DATASET_DIR}/$a" >&2
    fi
  done
elif [[ -n "$LIST_FILE" ]]; then
  # 목록 파일에서 읽기
  [[ -f "$LIST_FILE" ]] || { echo "에러: 목록 파일이 없습니다: $LIST_FILE"; exit 1; }
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    [[ "$line" =~ ^# ]] && continue
    if [[ -d "${DATASET_DIR}/$line" ]]; then
      CASE_FILES+=("$line")
    else
      echo "경고: 케이스를 찾을 수 없음: ${DATASET_DIR}/$line" >&2
    fi
  done < "$LIST_FILE"
else
  # 무작위 선택 (디렉터리 목록에서)
  mapfile -t ALL < <(ls -1d "${DATASET_DIR}"/*/ 2>/dev/null | xargs -n1 basename | sort || true)
  if [[ ${#ALL[@]} -eq 0 ]]; then
    echo "에러: ${DATASET_DIR} 아래에 케이스 디렉터리가 없습니다."; exit 1
  fi
  mapfile -t CASE_FILES < <(printf "%s\n" "${ALL[@]}" | shuf -n "$N_RANDOM")
fi

# 실행 대상 확인
if [[ ${#CASE_FILES[@]} -eq 0 ]]; then
  echo "에러: 실행할 케이스가 없습니다."; exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "총 ${#CASE_FILES[@]}개 케이스 실행 (CUDA):"
for case_name in "${CASE_FILES[@]}"; do
  echo " - $case_name"
done
echo

# 각 케이스 실행
RET_CODE=0
for CASE_NAME in "${CASE_FILES[@]}"; do
  echo "[CUDA 실행] ${CASE_NAME}"

  # CUDA 실행: --case <case_name> --out <output_dir>
  if "$CUDA_BIN" --case "$CASE_NAME" --out "$OUTPUT_DIR"; then
    echo "[완료] ${CASE_NAME}"
  else
    echo "[실패] ${CASE_NAME}" >&2
    RET_CODE=1
  fi
done

echo
if [[ $RET_CODE -eq 0 ]]; then
  echo "모든 CUDA 실행이 완료되었습니다. 결과: ${OUTPUT_DIR}"
else
  echo "일부 케이스 실행에 실패했습니다."
fi

exit $RET_CODE
