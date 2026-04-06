from matpowercaseframes import CaseFrames
import scipy.io  
import numpy as np
import os

DATASET_ROOT = "/datasets/pglib-opf"
SAVE_DIR = "/workspace/datasets/pf_dataset"

"""
PGLib-OPF 데이터셋과 같이 MATPOWER 케이스 파일(.m) 형식으로 저장된
전력 계통 데이터들을 일괄적으로 PYPOWER/SciPy에서 불러올 수 있는 
MATLAB .mat 파일 형식으로 변환하는 스크립트입니다.

주요 동작:
1. DATASET_ROOT 디렉터리에 있는 모든 .m 파일을 순회합니다.
2. 'matpowercaseframes' 라이브러리를 사용해 .m 파일을 파싱합니다.
3. .to_mpc() 메서드를 호출하여 PYPOWER 케이스(ppc) 딕셔너리 형식으로 변환합니다.
4. ppc 딕셔너리 내부의 핵심 데이터(bus, gen, branch 등)를
   scipy.io.savemat 호환성을 위해 NumPy 배열로 명시적 변환합니다.
5. 변환된 딕셔너리를 'mpc'라는 키 값으로 래핑하여( {'mpc': ppc} ),
   SAVE_DIR에 원본 파일명과 동일한 .mat 파일로 저장합니다.
6. 전체 변환 과정의 성공 및 실패 횟수를 집계하여 출력합니다.
"""

os.makedirs(SAVE_DIR, exist_ok=True)

files = os.listdir(DATASET_ROOT)
failure_cnt = 0
success_cnt = 0

print(f"Starting conversion for files in: {DATASET_ROOT}")
print(f"Output directory: {SAVE_DIR}")
print("-" * 30)

for file in files:
    if not file.endswith(".m"):
        continue

    file_path = os.path.join(DATASET_ROOT, file)
    
    try:
        cf = CaseFrames(file_path)
        ppc = cf.to_mpc()

        ppc['bus'] = np.array(ppc['bus'])
        ppc['gen'] = np.array(ppc['gen'])
        ppc['branch'] = np.array(ppc['branch'])
        ppc['gencost'] = np.array(ppc['gencost'])

        base_filename = os.path.splitext(file)[0]
        save_path = os.path.join(SAVE_DIR, f"{base_filename}.mat")
        
        scipy.io.savemat(save_path, {'mpc': ppc})

        print(f"Successfully converted and saved: {file} -> {base_filename}.mat")

    except Exception as e:
        print(f"error (conversion or numpy failed): {file}")
        print(f"  > {e}")
        failure_cnt += 1

print("-" * 30)
print(f"Total processed: {success_cnt + failure_cnt}, Success: {success_cnt}, Fail: {failure_cnt}")