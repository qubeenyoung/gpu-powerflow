# Schur Complement Notes

이 문서는 `cuPF`의 Newton-Raphson Jacobian을 Schur complement 관점에서 볼 때
중요한 개념적 이슈를 정리한다. 구현 상세나 코드 변경 계획이 아니라, 왜
이 접근이 가능한지와 무엇을 조심해서 해석해야 하는지를 설명하는 문서다.

---

## 1. Jacobian은 자연스럽게 4개 블록이다

NR power flow에서 상태 변수는 보통 다음 순서로 놓인다.

```text
x = [ Δtheta(pv,pq), Δ|V|(pq) ]
```

따라서 Jacobian은 아래와 같은 2×2 block matrix가 된다.

```text
J = [ J11  J12 ] = [ dP/dtheta  dP/d|V| ]
    [ J21  J22 ]   [ dQ/dtheta  dQ/d|V| ]
```

각 블록의 의미는 다음과 같다.

- `J11`: 활성전력 mismatch의 angle 민감도
- `J12`: 활성전력 mismatch의 voltage magnitude 민감도
- `J21`: 무효전력 mismatch의 angle 민감도
- `J22`: 무효전력 mismatch의 voltage magnitude 민감도

크기는 보통 아래와 같이 읽을 수 있다.

- `J11`: `(n_pv + n_pq) × (n_pv + n_pq)`
- `J12`: `(n_pv + n_pq) × n_pq`
- `J21`: `n_pq × (n_pv + n_pq)`
- `J22`: `n_pq × n_pq`

즉, Jacobian은 처음부터 Schur complement로 분할하기 쉬운 구조를 갖고 있다.

---

## 2. 블록 구조 덕분에 Schur 인덱스를 정하기 쉽다

cuDSS sample은 Schur complement를 만들 부분공간을 "행/열별 binary index vector"
로 받는다. 개념적으로는:

- `0`: Schur 바깥
- `1`: Schur 안

우리 Jacobian은 변수 ordering이 이미 block contiguous하기 때문에, 특정 block을
Schur 대상로 잡는 인덱스를 비교적 단순하게 정의할 수 있다.

예를 들면:

- `J22` 쪽을 Schur 대상으로 잡고 싶으면 마지막 `n_pq`개 인덱스를 `1`로 둔다.
- `J11` 쪽을 Schur 대상으로 잡고 싶으면 앞의 `n_pv + n_pq`개 인덱스를 `1`로 둔다.
- 더 일반적으로는, angle 변수 일부와 magnitude 변수 일부를 섞은 custom subset도 가능하다.

핵심은 다음이다.

- Jacobian이 4블록이라는 사실 자체가 Schur 분할 후보를 자연스럽게 제공한다.
- 특히 `J11` 또는 `J22` 중심 분할은 수학적으로 가장 해석하기 쉽다.

---

## 3. Schur complement는 "solve를 없애는 것"이 아니다

행렬을

```text
J = [ A  B ]
    [ C  D ]
```

로 나누면, full system

```text
J x = b
```

를 더 작은 reduced system으로 바꿀 수 있다.

예를 들어 `A`를 먼저 없애면 Schur complement는

```text
S = D - C A^{-1} B
```

가 된다.

중요한 점은:

- full system이 reduced system으로 바뀔 뿐이다.
- 따라서 `S z = rhs`는 여전히 풀어야 한다.
- 즉, Jacobian이 4블록이라고 해서 Schur complement만 만들면 곧바로 답이 나오는 것은 아니다.

이 점 때문에 Schur 접근에서는 항상 "reduced Schur system을 누가 풀 것인가"가 같이 따라온다.

---

## 4. cuDSS Schur mode는 외부 Schur solve를 전제로 한다

참고 코드 [sample_schur_complement.cpp](/workspace/sample_schur_complement.cpp)는
Schur complement를 만들 수는 있지만, 그것만으로 full system을 한 번에 끝내지 않는다는
점을 분명히 보여준다.

sample의 흐름은 개념적으로 아래와 같다.

1. Schur mode를 켠다.
2. 어떤 행/열이 Schur 대상인지 index vector를 준다.
3. 분석과 factorization을 수행한다.
4. cuDSS에서 Schur matrix를 꺼낸다.
5. full solve를 바로 끝내는 대신:
   - 앞부분 partial solve
   - Schur system external solve
   - 뒷부분 partial solve
   로 나눠서 진행한다.

즉 sample은 다음 사실을 보여준다.

- cuDSS는 Schur matrix를 계산하고 노출할 수 있다.
- 하지만 Schur mode에서는 reduced system을 caller가 별도로 풀어야 한다.
- sample은 그 external solver로 `cuSOLVER` dense LU를 사용한다.

이건 sample의 핵심 메시지이지, 부수적인 구현 디테일이 아니다.

---

## 5. sample은 어떻게 하고 있나

sample은 대략 아래 순서로 움직인다.

### Schur 모드 활성화

- solver config에서 Schur mode를 켠다.
- user-provided Schur index vector를 data object에 넣는다.

### Analysis 이후 Schur shape 확인

- 분석이 끝난 뒤 Schur matrix의 크기를 조회한다.
- dense로 받을지 sparse로 받을지에 따라 필요한 저장 공간을 준비한다.

### Factorization 후 Schur matrix 추출

- factorization을 수행한 뒤 Schur matrix를 얻는다.
- sample은 이를 dense matrix로 받아서 출력까지 해본다.

### Full solve는 partial solve + external Schur solve로 분리

- forward solve를 Schur 경계까지 수행한다.
- sample의 경우 대칭 행렬 예제라 diagonal solve도 포함된다.
- 그 다음 Schur matrix를 외부 dense solver로 푼다.
- 마지막으로 backward solve를 수행해 원래 변수 전체 해를 복원한다.

여기서 특히 중요한 해석은 다음 두 가지다.

- sample은 "Schur matrix를 꺼내는 것"과 "전체 선형계 해를 완성하는 것"을 분리한다.
- sample의 diagonal solve 단계는 대칭 예제의 LDL 계열 분해 흐름에 묶여 있는 부분이므로,
  모든 행렬에서 똑같이 읽으면 안 된다.

---

## 6. Schur 관점에서 봐야 할 이슈

### 6-1. 어떤 block을 Schur 대상으로 잡을 것인가

가장 자연스러운 후보는 다음 둘이다.

- angle block 기준: 앞쪽 변수 집합
- magnitude block 기준: 뒤쪽 `pq` 변수 집합

이 선택은 reduced system 크기와 성질에 직접 영향을 준다.

### 6-2. Schur matrix는 더 작아져도 더 조밀해질 수 있다

Schur complement는 차원을 줄일 수 있지만, 원래 희소 구조가 그대로 유지된다는 보장은 없다.
오히려 reduced matrix가 dense에 가까워질 수 있다.

즉, "작아졌으니 무조건 더 쉽다"는 식으로 보면 안 된다.

### 6-3. Jacobian 블록 구조와 cuDSS sample의 행렬 가정은 구분해서 봐야 한다

sample은 작은 대칭 예제를 사용한다. 반면 power flow Jacobian은 일반적으로
"block structure는 분명하지만, sample과 완전히 같은 행렬 성질"이라고 단순화하면 안 된다.

따라서 sample은 Schur workflow를 이해하는 데는 매우 유용하지만,
sample의 solve phase를 그대로 1:1 대응물처럼 읽는 것은 조심할 필요가 있다.

---

## 7. 요약

- NR Jacobian은 본질적으로 4블록 구조라서 Schur 분할 후보가 명확하다.
- 변수 ordering이 연속적이므로 Schur index vector를 정의하는 일 자체는 어렵지 않다.
- 하지만 Schur complement를 만든다고 full solve가 자동으로 끝나는 것은 아니다.
- cuDSS sample도 reduced Schur system은 외부 solver가 필요하다는 전제로 작성되어 있다.
- 따라서 Schur 이슈의 핵심은 "블록을 어떻게 자를 수 있는가"와 함께
  "잘라낸 뒤 reduced system을 무엇으로 풀 것인가"를 같이 보는 데 있다.
