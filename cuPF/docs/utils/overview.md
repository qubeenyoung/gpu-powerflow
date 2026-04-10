# 유틸리티

**위치**: `inc/utils/`

네 가지 헤더-only (또는 헤더 중심) 유틸리티. 모두 `namespace newton_solver::utils` 아래.

---

## logger.hpp

spdlog 기반 로거 래퍼. 전역 싱글턴 상태(`LoggerState`)를 통해 레벨과 활성화 여부를 제어한다.

```cpp
initLogger(LogLevel::Debug);       // 레벨 + 활성화 한 번에
setLogLevel(LogLevel::Warn);
setLogEnabled(false);

LOG_DEBUG("msg");
LOG_INFO("msg");
LOG_WARN("msg");
LOG_ERROR("msg");
```

**레벨**: `Debug < Info < Warn < Error < Off`

매크로(`LOG_*`)가 주 사용 방식. `log<T>()` 템플릿 함수는 스트림 변환(`ostringstream`)을 통해 임의 타입을 지원한다.

---

## timer.hpp

RAII 스코프 타이머. 소멸 시 자동으로 경과 시간을 로그에 출력한다.

```cpp
{
    ScopedTimer t("analyze", LogLevel::Debug);
    // ... 측정할 코드 ...
}  // → "[debug] analyze took 0.123456 sec"
```

```cpp
ScopedTimer t("solve");
// 중간에 수동 종료
t.stop();
double sec = t.elapsedSeconds();
// stop() 이후 소멸자에서는 중복 출력 안 함
```

내부 클록: `std::chrono::steady_clock` (마이크로초 분해능).

---

## dump.hpp

NR 루프 중간 상태를 파일로 덤프하는 진단용 유틸리티. `setDumpEnabled(true)` 해야 동작한다.

```cpp
setDumpDirectory("debug_output");
setDumpEnabled(true);

dumpVector("F", iter, f_vec);            // → debug_output/F_iter3.txt
dumpCSRMatrix("J", iter, j_csr);         // CSR 행렬 덤프
dumpCSCMatrix("Ybus", iter, y_csc);      // CSC 행렬 덤프
```

`dumpMatrix()`는 CSR/CSC/COO 타입을 오버로드로 자동 구분한다.  
비활성화 상태(`isDumpEnabled() == false`)에서는 즉시 `false` 반환, 오버헤드 없음.

출력 파일 형식:
```
type csr_matrix
rows 5
cols 5
nnz 12
row_ptr 0 2 4 7 10 12
col_idx 0 2 1 3 ...
values  1.0 -0.5 ...
```

---

## cuda_utils.hpp

CUDA/cuSPARSE/cuDSS API 호출 결과를 검사하는 매크로. CUDA 빌드에서만 include된다.

```cpp
CUDA_CHECK(cudaMalloc(&ptr, size));
CUSPARSE_CHECK(cusparseCreate(&handle));
CUDSS_CHECK(cudssCreate(&handle));
```

실패 시 `std::runtime_error` 발생. 메시지에 파일명과 줄 번호 포함:
```
CUDA error at initialize.cpp:140 - out of memory
```
