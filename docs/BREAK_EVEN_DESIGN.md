# 복원 프레임워크 손익분기 예측 실험 설계

2026년 10월 4일 작성. 이 문서는 재실행 R, 수작업 네이티브 복원 N, 공통
복원 프레임워크 C1의 비용을 수식으로 설명하고, 관측하지 않은 조건의 실행시간과
손익분기 여부를 예측하는 후속 연구의 설계다. **설계만 완료하며 구현과 새 실험은
아직 수행하지 않는다.** 실행 예산은 제안이며 이전 코호트의 1,200초 승인을 승계하지 않는다.

독자는 구현자와 논문 저자다. 기존 [연구 결과](RESEARCH_RESULTS.md)는 동기와
한계의 근거이며 새 실험의 계수 추정, 표본 또는 확인 자료에 합치지 않는다.
설계 설정은 [구조화된 프로토콜](break-even-protocol.json), 작업과 추적성은
[구현 계획](BREAK_EVEN_IMPLEMENTATION_PLAN.md)에 연결한다. JSON은 향후 구현의
입력 명세이며 현재 실행기가 읽을 수 있는 실행 설정이 아니다.

## 연구 질문과 주장 범위

- RQ1: 의미 보존을 유지하면서 앞부분 계산을 재사용하는 이득은 일반화 비용을 언제 상쇄하는가?
- RQ2: 계산량과 보유 상태 크기를 구분한 비용모형이 새 조건의 평균 시간 차이를 예측하는가?
- RQ3: N 대비 C1의 추가 비용은 얼마이며, 입력 구조가 바뀌면 예측은 얼마나 달라지는가?

목표는 항상 빠른 방법이나 새 checkpoint 원리를 주장하는 것이 아니다. 조건부 비용식,
명시적인 복원·개입 계약, 독립 예측 검증을 결합한다. 학습 수렴, 정책 품질, 사람의
개발시간 절감, gorti federation, 일반적인 병렬 우위는 평가하지 않는다.

기존 Q/M 실험은 324개 시간 측정과 6개 계수 측정을 완료했고, 18개 조건 모두에서
C1이 R과 N보다 느렸다. 앞부분 생략은 확인했지만 L을 늘리면 계산량과 누적 이력이
함께 늘었다. 또 Q의 분기는 동일했고 M은 두 행동 계획을 반복했다. 새 설계는 이
교란을 분리한다. 기존 결과를 불리하다는 이유로 삭제하거나 새 결과로 대체하지 않는다.

## 분석 모형

### 실제 실행 절차의 비용 항등식

현재 절차와 동일하게 원본을 한 번 실행·저장하고 닫은 뒤 첫 분기를 포함한 B개
분기를 모두 복원한다. m은 N 또는 C1이다. fresh는 모델 생성과 초기화, close는
해당 runtime 종료를 뜻한다. 모든 시간은 같은 endpoint에서 구간이 겹치지 않게 잰다.

```text
T_R = U_R + sum_b(f_R,b + P_R,b + S_R,b + c_R,b) + H_R
T_m = U_m + f_m + P_m + K_m + c_m,source
      + sum_b(r_m,b + S_m,b + c_m,b) + H_m
```

U는 backend 구성, f는 fresh/reset, P는 앞부분 실행, S는 개입 후 실행, K는 저장,
r은 복원, c는 runtime 종료다. H에는 endpoint 안에 남는 루프·정리·계측 비용이
들어간다. R은 fresh와 prefix가 B회, N/C1은 1회다. N/C1의 restore는 B회,
runtime close는 B+1회다. snapshot 파일 폐기 비용도 누락하지 않는다.

### 분기 수의 손익분기 조건

입력 분포와 분기 목록의 비용 평균이 안정적이라는 가정 아래 위 식을 기대시간에 대해
축약한다. 한 번 측정한 분기 평균이 모든 B에서 같다는 뜻은 아니다.

```text
E[T_R] = A_R + B(P_R + S_R)
E[T_m] = A_m + P_m + K_m + B(r_m + S_m)
F_m = A_m + P_m + K_m - A_R
D_m = P_R + S_R - r_m - S_m
G_m = E[T_R - T_m] = B D_m - F_m
```

여기서 축약식의 P에는 각 방식의 fresh 비용을, S에는 분기 종료 비용을 포함한다.
A에는 source 종료 등 나머지 일회성 비용을 포함한다. 실제 계수 추정은 아래의
전체시간 회귀로 수행하고 구간별 비용은 설명과 누락 확인에 사용한다.

| 경우 | m이 더 빠른 정수 분기 수 B |
|---|---|
| D > 0 | B >= max(1, floor(F/D)+1) |
| D = 0 | F < 0이면 모든 B, 그 외 엄밀한 우위 없음 |
| D < 0 | 1 <= B <= ceil(F/D)-1, 이 범위가 비어 있으면 없음 |

F>=0이고 D<=0이면 분기를 늘려도 이 모형에서 이득이 없다. D<0인 유한 근은
일반적인 이득 시작점이 아니라 기존 이득이 사라지는 지점일 수 있다. F=D=0이면
모든 B에서 동률이다. 첫 분기를 복원 없이 실행하는 B-1 방식은 별도 알고리즘이며
이번 설계와 혼합하지 않는다. 관측 B는 4, 8, 16이므로 그 밖의 근은 외삽이다.

### 계산량과 상태 크기

계산량 K와 상태 수준 s별로 방법 m의 평균 전체시간을 다음의 단순 모형으로 적합한다.
K는 아래의 risk scenario 수이며 저장비용 K_m과는 다른 변수다.

```text
E[T_m | K,B,s] = a_m,s + b_m,s K + c_m,s B + d_m,s K B
Delta_m,s(K,B) = E[T_m - T_R]
              = (alpha_m,s + beta_m,s K) - B(gamma_m,s + eta_m,s K)
F_m,s(K) = alpha_m,s + beta_m,s K
D_m,s(K) = gamma_m,s + eta_m,s K
```

동일 설계행렬로 절대시간과 대응 시간 차이의 비가중 최소제곱을 각각 적합한다.
부호를 강제하지 않고 family를 동등 가중한다. 기하평균 시간비를 가감하여 계수를
만들지 않는다. 상태 수준별로 따로 적합하며 두 수준 사이의 연속 상태 크기 법칙은
주장하지 않는다. K와 B의 열을 고정 상수로 스케일링해 수치 안정성을 확보한다.

계산량 손익분기는 분모가 0이 아닐 때
`K* = (B gamma - alpha)/(beta - B eta)`다. 어느 쪽이 유리한지는 Delta의 부호로
판정한다. K*가 허용 범위 밖이거나 양의 근이 없으면 그대로 보고한다. 계수의 선형성,
캐시 효과, branch 평균 안정성이 맞지 않으면 이 식은 기각될 수 있는 예측 모형이다.

동일한 추가 prefix 연산만 DeltaP만큼 늘고 나머지 비용이 고정일 때의 특수식은
`new_gain = old_gain + (B-1) DeltaP`다. 같은 추가 suffix 연산은 상쇄된다.
이는 C1이 N을 이긴다는 식이 아니다. 두 방식이 같은 계산을 생략하면 C1-N의 추가
비용은 남을 수 있다. 이 특수 가정을 실제 계수 모형에 강제하지 않는다.

## 요구사항과 시험 설계

아래 SRS와 STD를 먼저 정의하며, 뒤의 측정 및 분석 설계가 이를 구현해야 한다.
시험은 해당 연구 주장을 직접 검증하는 것으로 한정한다. 자동 smoke, 설치 시험,
freeze-state 검증 체계 또는 전체 회귀시험을 실험 시작 조건으로 추가하지 않는다.

| SRS | 요구사항 | STD와 수용 기준 |
|---|---|---|
| BE001 | R/N/C1이 같은 모델 함수와 입력을 실행 | TC01 동일 source/config/input/action identity, 방식별 연산 삽입 금지 |
| BE002 | 복원 후 개입 의미 보존 | TC02 companion에서 시각·상태·보상·RNG·순서의 stepwise 정확 일치 |
| BE003 | 복원이 모델을 앞부분까지 재실행하지 않음 | TC03 prefix R=B,N/C1=1; restore N/C1=B; capture/restore model callback 벡터0 |
| BE004 | 분기는 실제로 다르고 서로 격리됨 | TC04 q별 실제 수량 결과 차이, snapshot 불변성, 직접 A/B/A 격리 시험 |
| BE005 | K가 실제 모델 계산이고 상태 수와 구별됨 | TC05 K×H 유효 계산, risk가 보상에 사용됨, 논리 상태 수·실제 snapshot bytes 기록 |
| BE006 | 관찰기 비용과 제품 비용 구분 | TC06 타이머 spy로 연속 구간 경계 확인, production validation 유지 |
| BE007 | 비용모형의 부호와 모든 근 상태 처리 | TC07 양수·음수·0 분모, 동률, 범위 밖 근, 다수 부트스트랩 무근 사례 |
| BE008 | 계수 추정과 예측 검증 분리 | TC08 중복 seed 없음, 선택 규칙 재현, validation이 적합 입력에 들어가지 않음 |
| BE009 | 불확실성과 반복 단위의 정합성 | TC09 whole-family 재표집, 6개 primary 다중비교, 미달 정밀도 보고 |
| BE010 | 실패와 미실행을 숨기지 않음 | TC10 실패·불일치·timeout·미실행 분모, 재시도와 성공 대체 없음 |
| BE011 | 기존 자료와 구현을 보존 | TC11 기존 core/vendor/v1 kernel/analyzer와 기존 results를 변경·재분석하지 않음 |
| BE012 | 예산과 실행 소유권을 준수 | TC12 명시한 작업만 종료, 새 output만 사용, 시간·저장 중단 기록 |
| BE013 | 예측 정확성을 실제 새 관측으로 평가 | TC13 사전 예측과 signed-seconds 잔차·CI·교차 상태를 모든 확인점에 출력 |
| BE014 | 주장 범위와 기여를 구분 | TC14 synthetic mechanism, 새 입력 구조, 인간 평가와 RL 비평가를 보고서에 명시 |

TC01/04/05/06/07/08/09/10/11/12와 TC13a는 필요한 직접 개발 시험이다. TC02/03은
직접 사례와 실제 companion 측정에 연결하고, TC13b는 실제 holdout 예측 검증이다.
직접 시험의 통과 수를 실제 실험 완료 횟수로 계산하지 않는다.
BE002를 만족하지 않는 사례는 비용상 유리하더라도 의미 보존 성공 사례가 아니다.

## 계산량과 상태를 구분하는 모델

### 버전이 분리된 재고 위험평가 모델

기존 inventory V1/V2와 Q/M을 수정하지 않고 새로운 `inventory-risk-v1`을 작성한다.
이는 분석자가 설계한 도메인 기반 기전 벤치마크이며 실제 창고 자료나 현장 검증을
사용한 산업 사례라고 부르지 않는다. 기존 공통 continuation core는 재사용한다.

단일 actor가 S개 제품 레코드를 관리한다. 레코드는 재고, 수요 regime, 판매·품절·입고
누적값 등 실제 미래 전이에 필요한 고정 수의 값을 갖는다. 한 사건은 제품 하나를
인덱스로 읽고 갱신한다. actor 수, event calendar 형태, demand 수를 S에 따라 늘리지
않는다. 현재 inventory의 pending 목록이나 전체 이력을 늘려서 state-only라고 부르지 않는다.

각 수요 전이 후 해당 제품의 8기간 재고 위험을 K개 시나리오로 평가한다.
미래 시나리오 수요·regime을 계산하고 품절 및 보유비용을 누적한 평균을 보상에 반영한다.

```text
for each actual demand event:
    update actual stock and lost sales
    total = 0
    for scenario in range(K):
        virtual_state = fixed_size_copy(current_product)
        for h in range(8):
            draw = forecast_draw(seed, demand_id, scenario, h)
            update virtual regime, demand, scheduled receipts and inventory
            total += lost_weight * virtual_lost + holding_weight * virtual_stock
    last_risk = total / K
    cumulative_risk += last_risk
at each decision boundary:
    reward = (fulfilled - previous_fulfilled)
             - risk_weight * (cumulative_risk - previous_cumulative_risk)
    update both previous-value baselines
```

forecast_draw는 모델 RNG와 분리된 명시적 deterministic counter 기반 함수다.
Python의 process-randomized hash를 사용하지 않는다. 과거 또는 실제 미래 수요를
읽지 않는 모형 기반 예측이며 forecast seed와 알고리즘을 동일하게 기록한다.
K개의 궤적을 저장하지 않고 scratch는 고정 크기다. risk 계산은 prefix와 suffix의
모든 수요 전이에 동일하게 적용하며 sleep, 사용하지 않는 반복문, prefix 전용 부하는 금지한다.

여러 수요가 한 decision step에 있어도 모든 risk를 누적한다. last_risk만 덮어써
이전 계산을 보상에서 버리지 않는다. cumulative_risk와 두 reward baseline도 복원한다.
계산 결과는 risk-adjusted reward 및 선언된 상태에 실제 반영된다. 모든 방법이 같은
imported callable을 호출한다. K가 바뀌면 선언한 forecast 분포의 근사 정밀도와 보상값은 달라질 수
있다. 정확 일치는 같은 K/config/seed의 R/N/C1 사이에서 요구한다. 위험값으로 행동을
자동 변경하지 않는 open-loop 평가이므로 K만 바꿀 때 외부 사건 구조는 유지한다.

각 호출에서 정수 비용 누적과 마지막 나눗셈을 사용하고, 별도 단순 scalar oracle로 작은 입력을
검사한다. 생산용 validation이 risk를 재계산한다면 이를 제거하지 않고 비용과 호출
위치를 보고한다. restore 중 모델 전이는 금지지만 상태 validation 계산은 비용이다.

### 계산 알고리즘의 구체적 계약

`forecast-h8-lcg64-v1`의 한 기간은 0.25이며 H8은 미래 2.0시간 단위다. 실제 발주의
lead time0.5와 같다고 부르지 않는다. lost_weight=4, holding_weight=1,
risk_weight=0.01로 고정한다. 각 scenario는 현재 stock/regime에서 시작한다.
기존 pending 입고의 product가 평가 대상 제품과 같으면 그 due를 포함하는 첫 forecast
기간에 그 수량을 더한다. 다른 제품의 입고를 더하지 않는다.
가상의 forecast 수요·입고는 실제 event calendar에 삽입하지 않는다.

시나리오 초기 state는 forecast_seed, demand_id, scenario_index를 아래 순서대로
섞어 uint64로 만든다. 모든 산술은 mod2^64이며 오른쪽 shift는 unsigned이다.

```text
mix64(x):
    x = ((x xor (x >> 30)) * 0xbf58476d1ce4e5b9) mod 2^64
    x = ((x xor (x >> 27)) * 0x94d049bb133111eb) mod 2^64
    return x xor (x >> 31)
x = mix64(forecast_seed xor 0x9e3779b97f4a7c15)
x = mix64(x xor demand_id)
x = mix64(x xor scenario_index)
for h in range(8):
    x = (6364136223846793005*x + 1442695040888963407) mod 2^64
    u = x >> 32
    if (u & 3) == 0: regime = 1-regime
    forecast_demand = 1 + ((u >> 2) & 3) + 3*regime
    stock += known_pending_receipt_for_this_period
    served = min(stock, forecast_demand)
    stock -= served
    cost += 4*(forecast_demand-served) + stock
```

period h의 입고 포함 조건은 `now+h*0.25 < due <= now+(h+1)*0.25`다. 현재 시각에
이미 처리한 입고를 다시 예측하지 않는다. regime은 두 상태이며 위 비트 규칙이
전이·수요 분포의 정의다. 물리적 난수 품질이나 현장 분포 적합성을 주장하지 않는다.
forecast 법칙은 실제 수요 tape의 생성·regime 갱신 법칙과 다른 가상 위험 시나리오다.
이 값은 그 선언된 분포 아래의 위험 점수이며 실제 simulator 미래손실의 조건부 기댓값이나
현장 예측 정확도를 검증한 값이 아니다.
scenario seed에 method, branch label 또는 physical object ID를 넣지 않는다.
같은 사건과 동일 상태에는 동일 forecast randomness를 사용한다. K의 작은 scenario
집합은 큰 K의 앞부분이다. 정수 합계는 정확하고 나눗셈/보상 누적은 Python binary64를
동일 순서로 사용한다. 서로 다른 K 사이의 reward 일치는 요구하지 않는다.

제품 레코드는 `stock=20, regime=i%2, fulfilled=0, lost=0, received=0`으로 시작한다.
construction RNG를 S번 소비하지 않으므로 처음 8개는 모든 S에서 동일하다. global
상태는 actual demand cursor, seed, cumulative risk, last risk, 전체 fulfilled/lost,
reward baseline과 하나 이하의 실제 pending order다. 자동 보충이나 다른 초기 입고는
없다. 실제 demand 처리 후 regime을 `(regime + quantity%2)%2`로 갱신한다.
ordinary demand 수량은 입력 materialization의 `random.Random(input_seed)`에서
각 row마다 `randrange(1,5)` 한 번으로 만든 뒤 아래 두 row만 override한다.
이는 bounded 기전 모델이지 최적화된 재고 위험계산 알고리즘이라는 주장은 아니다.

seed namespace는 `label + ':' + decimal(family_seed)`의 UTF-8 SHA256 첫 8byte를
unsigned big-endian으로 읽는 규칙이다. label은 `be-forecast-v1`, `be-actions-v1`,
`be-order-v1`로 구별한다. 완성된 input과 action 목록을 기록하므로 Python 버전의
shuffle 구현에 의존하는 재현성을 숨기지 않는다. action seed로 보완쌍을 shuffle하고
각 쌍 내부 순서를 getrandbits(1)로 정한 뒤, 선택한 목록을 한 번 shuffle한다.

정상 제품 observation은 clock, 전체 fulfilled/lost, cumulative risk와 action 대상
제품의 scalar stock/received만 반환한다. reward는 step 반환 tuple의 별도 scalar로
두고 observation에 중복 계산하지 않는다. observe 뒤 reward baseline을 갱신하는
기존 호출 순서와 재관찰을 모순 없이 유지한다. full product table과 전체
history를 observation에 복사하지 않는다. 생산 모델은 무제한 이벤트 이력을 보관하지
않고 cursor와 누적값을 갖는다. companion은 별도 observer로 매 step의 table, pending,
cursor, reward baseline 및 실제 event 순서 trace를 추출한다. 이 차이를 mode별 domain
알고리즘 변경으로 구현해서는 안 된다. scalar consumer 비용은 모든 방법의 timing에 포함한다.

### 고정 조건과 조작 변수

| 변수 | 설정 | 해석 |
|---|---|---|
| 계산량 K | 1, 16, 256, 4096 | domain estimator의 scenario 수, 성능 결과에 따라 상한 확대 금지 |
| 상태 레코드 S | 8, 512 | 논리 레코드 수, bytes 또는 실제 창고 규모로 환산하지 않음 |
| 분기 수 B | 4, 8, 16 | 서로 다른 양수 발주 개입 |
| prefix / suffix | 64 / 16 decision steps | 길이를 고정하여 계산량과 이력 길이를 분리 |
| delta / forecast horizon | 0.25 / 8 | 시간 단위와 예측 반복 깊이 고정 |
| 방법 | R, N, C1 | 모두 fresh process, 순차 실행, worker1 및 BLAS1 |
| 상태 보존 | 전체 제품 레코드, cursor, clock, reward baseline, RNG 및 aliases | N sidecar와 C1 선언 모두 포함 |

수요 제품은 처음 8개의 active 제품을 순환한다. 나머지 레코드도 유효 상태이며
후속 개입에서 접근할 수 있지만 주 시간 구간의 활동량은 고정한다. 큰 S가 드문
활동의 더 큰 보유 상태를 나타낸다는 한계를 명시한다. 직접 TC05에서는 마지막
레코드에 개입하여 복원 누락을 검출한다. bytes는 N/C1 각각 실측하고, 숫자 표현·
캐시·validation에 따른 잔여 결합이 있으므로 완벽한 bytes 독립성을 주장하지 않는다.

### 입력과 개입 목록

기본 수요 생성기는 80개 row에 j=0..79 ID를 부여하고, `time=(2*j+1)/8`,
`product=j%8`로 각 quarter-step 내부에서 제품 하나에 수요를 낸다.
수량은 seed로 결정하되 개수와 시각은 고정한다. 실제 설정 파일에 모든 수요를
실체화하고 해시를 기록한다. prefix와 suffix 모두 모델이 계속 활동하도록 한다.

cut은 t=16이다. 발주 q=1..16을 t=16에서 제품 0에 주입하고 t=16.5에 입고한다.
action schema는 `{"product": 0, "order": q}`이며 prefix와 이후 zero action은
`{"product": 0, "order": 0}`이다. 실제 pending은 product/order/due의 단일 레코드다.
기존 row63을 `(id63,time15.875,product0,quantity100)`으로, row67을
`(id67,time16.75,product0,quantity100)`으로 **교체**한다. 사건을 추가하지 않으므로
prefix64/suffix16의 수요 수를 유지한다. 초기 재고20과 자동 입고 부재로 row63 뒤
제품0은 비며, q입고와 row67 사이에는 다른 제품0 수요·입고가 없다. 따라서 row67의
fulfilled가 정확히 q가 된다. oracle로도 이를 검사하고 실패하면 BE004 미충족이다.
이후 action은 order0이다.
모든 q의 발주·입고 사건 수는 같고 fulfilled 결과는 실제로 달라야 한다.

(1,16), (2,15), ..., (8,9)의 여덟 보완쌍을 family별로 섞어 B4에는 앞 2쌍,
B8에는 앞 4쌍, B16에는 8쌍을 사용한다. 같은 family의 K/S 조건에는 같은 목록을
적용한다. 평균 발주량은 항상 8.5지만 비선형 suffix 비용까지 같다고 가정하지 않는다.
목록 구성·실행 순서는 seed로 확정하고 모든 방법에 같은 목록을 준다.

별도 입력 구조 이전 평가는 기본 수요의 개수와 제품·수량을 유지하고, ordinary
row j의 시각을 `floor(j/4)+(j%4+1)/32`로 바꾸는 burst 생성기를 사용한다. 강제 두 사건과
cut 개입·입고 시점은 유지한다. 기본 생성기의 새 seed를 구조 이전이라고 부르지 않는다.
이는 새 모델이 아니라 같은 모델의 새 입력 구조다. 입력은 `(time,demand_id)`로
정렬하며 arrival cursor 순서와 demand ID를 혼동하지 않는다. 동시 시각의 실제 입고는
수요보다 먼저 처리하고 같은 시각의 수요는 ID 순서다. 이 규칙은 직접 oracle로 확인한다.

## 측정 설계

### 연속 시간 구간

새 primary endpoint는 `workflow_wall_seconds`다. source/config/input/action 준비와
imports를 끝낸 뒤 backend 생성 직전에 시작해, source와 모든 branch runtime 종료
및 해당 snapshot 폐기가 끝난 직후 정지한다. fresh, prefix, snapshot 생성, B회
restore, suffix의 제품 observation/reward, 필수 admission/validation, 종료·폐기까지 포함한다.
프로세스 기동·import·프로비넌스·독립 물리 projection·연구 receipt/전송은 제외한다.
여기서 준비 제외는 연구용 manifest/provenance에 해당하며 C1 제품의 필수 source
검증과 bundle 등록은 setup 비용에 남긴다.
같은 시작·종료 경계의 CPU time도 보조 지표로 기록한다. 부모의 process wall은 별도다.

coarse phase는 setup, fresh, prefix, capture_write, source_close, restore_read,
suffix, branch_close, artifact_cleanup이다. invocation count와 합계 및
`unclassified_seconds = workflow_wall - sum(nonoverlapping_phases)`를 기록한다.
분기 계획은 타이머 밖에서 한 번만 생성한다. phase 시간을 뺀 값을 오염 없는 전체
실행시간이라고 부르지 않는다. 타이머 자체의 작은 비용도 이 계측 구현의 일부다.

production C1 검증을 꺼서 성능을 개선하지 않는다. N도 sidecar·시간/calendar·보상
상태의 정상 복원을 포함한다. cached local filesystem, no fsync 조건을 명시하며
새 프로세스를 cold cache라고 표현하지 않는다. 파일 캐시 비우기나 다른 프로그램의
종료를 자동화하지 않는다. 기본 구현에서는 기존 kernel의 O(BL) 행동 계획 재생성과
연구용 누적 이력 JSON 복사가 새 endpoint 안에 들어가지 않도록 한다.

### 의미 확인과 시간 측정의 분리

각 family/condition 셀은 R/N/C1의 **의미 확인 companion 3회와 시간 측정 3회**로
구성한다. companion은 stepwise physical projection과 callback 계수를 수집하지만
그 실행시간은 성능 근거로 사용하지 않는다. timing은 독립 observer 없이 제품의
정상 observation/reward가 이미 반환한 immutable scalar 값만 동일한 consumer로
보존한다. scalar 소비·보존 비용은 측정에 포함하고 digest·비교는 endpoint 뒤 수행한다.
추가 graph 순회·직렬화·상태 복사·post-close live-state 접근은 하지 않는다.
이 최종 정상 출력과 종료 여부를 companion에 대조한다. 사후 검사를 위해 timed lifecycle을 연장하지 않는다.

companion의 전체 trace는 세 방식 간 메모리에서 정확 비교하고 compact receipt를
보존한다. 시각, 상태, 보상, future RNG, ordered history와 branch 결과가 대상이다.
추가 A/B/A 직접 시험과 snapshot 불변성 검사를 구분한다. callback 계수는 실제
int/ext/output/con 벡터로 보존하며 합계를 unique event 수라고 부르지 않는다.
risk 호출과 scenario-stage 수도 실제 실행을 phase별로 계수한다. 정상 수요 전이와
제품 validation 재계산을 구분하며 K×H에서 추정한 수를 실측으로 쓰지 않는다. 새 모델의
callback을 계수하고 기존 Q/M 전용 counter만 설치하지 않는다. capture/restore의
모든 risk 계산이 0이어야 한다고 요구하지 않는다.

**timing 실행 자체의 전체 trace를 exact 검증했다는 주장은 하지 않는다.** 동일 소스·
설정의 별도 결정론적 실행을 확인한 근거다. 실험 모드 차이는 observer 유무로만
제한하고, 입력이나 모델 함수가 달라지면 셀을 성공으로 인정하지 않는다.

companion/timing의 선행 순서는 순환 균형화하고, 각 역할 안의 R/N/C1 여섯 순열은
6개 family block마다 한 번씩 사용한다. 조건 순서는 family 안에서 무작위화한다.
모든 수집은 순차 실행한다. 코드 개발·검토 에이전트는 병렬 가능하지만 실제 측정 중
시험·설치·다른 모델 실행을 동시에 수행하지 않는다.

## 코호트와 표본 설계

### 계수 추정

K 4수준 × S 2수준 × B 3수준 × family 6개다. 144개 셀, 시간 432회와 companion
432회, 합계 **864회**다. 6은 작은 탐색 계수 표본이며 충분한 정밀도나 확증력을
보장하지 않는다. family는 독립 input seed, 방법 순서와 개입쌍 순서를 공유하는
전체 24조건 묶음이다. 조건·분기·scenario 수를 독립 표본 수로 세지 않는다.

첫 family의 24조건은 동일한 무작위 순서로 수행하는 자원 예측용 파일럿도 겸한다.
이를 별도 표본으로 중복 계산하지 않는다. 완료 후 각 condition/method/role의
process wall에 1.5배 여유를 두어 남은 횟수와 분석 여유시간을 계산한다. 이것은
완료 보장이 아닌 보수적 운영 추정이다. 부족하면 부분 결과와 분모를 남기고 중단하며
grid·계수·상한을 임의 변경하지 않는다. 실패한 arm의 재시도는 없다.

### 독립 예측 검증

계수 추정에 쓰지 않은 K 후보는 **4, 64, 1024**다. 각 S에서 B8의 예측
`abs(Delta_C1,R)`가 가장 작은 후보를 하나 선택하고, 동률은 작은 K를 선택한다.
그 K에서 B4/8/16을 검사하므로 **6개 primary 조건**이 된다. 이 규칙은 교차를
예측하지 못해도 적용되며, 유리한 점이 나올 때까지 후보를 확장하지 않는다.
6개 calibration family 전체가 완료·수용된 경우에만 좌표와 N을 산출한다. 부분
calibration이면 validation은 미실행으로 남긴다. 선택 규칙 적용 뒤 6개 좌표의 R/N/C1
절대 시간 예측이 모두 finite하고 양수여야 한다. 하나라도 위반하면
`model_prediction_failed`로 보고하고 후보를 교체하거나 모형을 슬쩍 변경하지 않는다.

선택 좌표·계수·예측값·표본 수·실행순서를 validation 관측 전에 파일로 기록한다.
이는 연구 설계 기록이지 추가 freeze harness가 아니다. 기본 생성기의 완전히 새로운
family를 사용하고 계수를 다시 적합하지 않는다. old/pilot/validation 자료를 합쳐
확인 효과나 CI를 계산하지 않는다.

표본 수는 계수 추정의 24조건별 paired `T_C1-T_R` 표준편차 중 최댓값 s_max를
사용한다. 6개 확인점의 고정 replay 시간 예측을 t_R,j라고 하고 다음으로 정한다.

```text
z = standard_normal_quantile(1 - 0.05 / (2 * 6))
h_j = 0.10 * t_R,j
N_required = 6 * ceil(max(12, max_j((z*s_max/h_j)^2)) / 6)
allowed N = 12,18,24,30,36,42,48
```

10%는 평균 시간차 예측의 실용적 정밀도를 위한 **사전 공학적 목표**이며 논문 성능
향상 기준이나 보장된 검정력이 아니다. 작은 calibration의 분산 추정과 정규 근사를
사용한 계획식임을 명시한다. N_required>48이면 48로 잘라 실행하지 않는다.
새 설계 판단이 필요하다. 실제 CI가 목표보다 넓어도 사후 N을 늘리지 않고 보고한다.

검증은 6N개 셀, 시간 18N회, companion 18N회, 합계 **36N회**다.

### 입력 구조 이전

같은 6개 좌표를 burst 생성기의 새로운 family 6개로 평가한다. 36개 셀,
시간 108회와 companion 108회, 합계 **216회**다. 계수를 재적합하지 않으며
기본 검증과 합치지 않는다. 별도 탐색적 외적 타당성 평가다. 이 평가를 생략하거나
중단하면 완료되지 않았다고 명시하고 입력 구조 이전 주장을 하지 않는다.

전체 계획은 **180+6N개 조건 family 셀**, 시간 **540+18N회**, companion도 같은
횟수로, 합계 **1080+36N회**다. N12이면 1,512회, N48이면 2,808회다. 큰 횟수는
분기 수가 아니라 두 역할과 세 비교 방법을 포함한 실행 횟수다.

## 분석과 판정

### 적합과 불확실성

계수 추정은 모형별로 하지 않고 방법별·S별로 위의 절대시간 및 대응 차이식을 적합한다.
잔차를 K/B, 실행 순서, 전후반 family로 표시한다. 계수 추정 내부의 leave-one-K-level-out
예측도 보고하지만 이를 독립 validation이라고 부르지 않는다. 잔차의 패턴을 보고
모형을 고칠 경우 그 버전은 새 연구이며 현재 validation을 다시 훈련 자료로 쓰지 않는다.

calibration의 whole family를 20,000회 재표집해 모든 S/K/B와 비교를 함께 유지한다.
부트스트랩마다 F, D, B*, K*의 부호·범위·근 상태를 기록한다. 무근, 범위 밖 근,
singular fit을 버리고 유한 근만으로 좁은 CI를 만들지 않는다. 모든 draw 상태의
빈도를 보고하며, 제한된 finite-root 조건부 요약은 반드시 조건부라고 표시한다.
무근 draw 빈도는 실제 세계의 무근 사후확률이 아니다.

### 독립 검증의 primary 판정

primary estimand는 6개 좌표 각각의 **평균 signed time difference**
`Delta = T_C1-T_R`와 고정 예측의 차이 `e = Delta_observed-Delta_predicted`다.
validation family를 공동 재표집하며 각 좌표의 percentile bootstrap
**99.1666667% CI**를 계산한다. 6개에 Bonferroni를 적용한 familywise 95% 목표이나,
bootstrap 구간의 유한 표본 정확도와 독립 family 가정에 의존하며 정확 보장은 아니다.
이 보정은 두 endpoint 집합 전체의 동시 보장이 아니라 아래 두 판단군에 각각 적용한다.

1. 예측 정확성 판단군: e의 CI가 `[-0.10*t_R,j, +0.10*t_R,j]` 안에 완전히
   들어오는지 6개 모두 보고한다. 6개 모두 충족할 때만 해당 확인점 집합의 평균
   예측 오차가 설정한 공학적 허용범위 안이라고 판정한다.
2. 실행시간 방향 판단군: Delta의 CI 상한<0이면 C1 우위, 하한>0이면 R 우위,
   0 포함이면 불확실이다. 근처 한 점의 방향으로 연속 전체 구간의 우위를 선언하지 않는다.

두 판단군을 합친 단일 확증 성공확률 95%를 주장하지 않는다. 같은 validation의
고정 예측 오차 구간은 관측 Delta 구간을 상수 이동한 것이다. calibration 불확실성은
별도 계수/근 보고와 두 코호트를 독립 재표집하는 보조 예측오차 분석으로 제시한다.
보조 분석에서도 원래 선택된 6개 좌표는 고정하고 midpoint 선택을 반복하지 않는다.
이를 선택 절차까지 보정한 추론이라고 부르지 않는다.

실측 교차는 동일 S/K에서 검증한 B의 부호 변화와 불확실한 점을 이용한 **구간**으로
보고한다. B*의 정확한 실수값을 실측했다고 쓰지 않는다. finite range 내 양쪽 부호가
확인되지 않으면 `crossing_not_demonstrated`다. 모형 예측이 범위 밖인 경우
`outside_supported_domain`으로 구분한다. K 확인값은 S당 하나이므로 K*를 촘촘히
실측한 연구라고 주장하지 않는다.

N-R, C1-N의 절대 차이와 기하평균 시간비·95% pointwise CI는 보조 탐색 결과다.
N이 더 빠른 결과도 모두 보인다. source/callback/risk 연산량, 실제 snapshot bytes,
phase 합과 unclassified 비용을 함께 보고한다. bootstrap 재표집은 셀이나 branch가
아니라 family 단위다. host/order 민감도는 사전 선언한 전후반·방법 순서별 기술통계다.

### 실패와 전체 판정

정확 불일치, source 혼합, 정상 종료 실패, 실행 소실, resource 초과는 실패다.
값이 불리하거나 느리다는 이유로 제외하지 않는다. timeout을 상한의 실행시간 관측으로
대체하지 않는다. 프로세스 소유권 확인 없이 PID만 보고 다른 작업을 종료하지 않는다.
완전 family가 부족하거나 계획된 실행이 실패·미실행이면 해당 코호트의 전체 수용은
false다. 완성 셀의 기술통계는 부분 결과로 보존하되 계획을 완료한 확증으로 부르지 않는다.

## 자원과 재현성

다음은 **새 실행 승인 전 제안값**이다. 현재 실행을 시작하거나 자동화를 설정하지 않는다.

| 항목 | 제안 |
|---|---|
| 계수 추정 / 독립 검증 / 입력 이전 | 3,600 / 3,600 / 1,800초, 합계 최대 9,000초 |
| 각 단계 분석 여유 | 해당 단계 상한 안에 120초 확보, 보장값은 아님 |
| 단일 arm | companion과 timing 각각 120초 |
| 새 연구 저장량 | 세 단계와 transient를 합쳐 32MiB, 그 안 transient 8MiB |
| 기존 자료 | 그대로 보존, 신규 32MiB 상한 제외, 기존 사용량 별도 기록 |
| 프로세스 | 실험 worker1, BLAS1, 다른 프로그램을 강제 종료하지 않음 |
| 메모리 | lifetime peak 보장 없음, 관측한 값과 coverage를 분리 |

위 새 저장 상한은 과거 코호트를 제외한 세 단계 합계다. 실행 전 기존 사용량과 free
공간을 별도 기록하고 free1GiB 미만이면 시작하지 않는다. 경계 관측 제한은 OS quota나
peak 보장이 아니다. 부족하면 output을 삭제해 계속하지 않고 부분 결과를 보존한다.
각 단계는 새 폴더로 수행하며 종료·실패 후 자동 재시작하지 않는다.

Linux 전용 실험 PC를 우선 대상으로 제안한다. OS, Python 버전, filesystem, CPU,
전원 설정과 기타 부하의 알려진 상태를 기록한다. Windows 개발시험은 별개다. 과거
Windows/Linux 시간 자료와 합치지 않으며 idle이라는 이름만으로 무간섭을 인증하지 않는다.
기존 전용 venv와 의존성을 재사용할 수 있고 실행 전에 전체시험·smoke를 자동 실행하지 않는다.

시간 예산 예상은 실제 process wall과 companion 비용으로 계산한다. N이 허용되어도
예상시간·저장량이 제안 상한에 맞지 않으면 실행 가능한 정밀도인지 사용자와 다시
판단한다. 상한·N·K를 자동 완화하지 않는다. 분석기 설치·코드 수정은 실험 중 하지 않는다.

구현 전 산술 검산상 제품 validation의 재계산을 제외한 calibration 전체 forecast-stage는
2,952,744,960회이고 최대 R arm은 41,943,040회다. calibration 3,600초는 나머지 비용을
0으로 쳐도 stage당 약1.22마이크로초를 요구한다. 따라서 제안 예산은 빠듯할 수 있으며
실측 시간 추정이나 완료 약속이 아니다. 첫 family 자료로 실제 실행 가능성을 판단한다.

## 논문에 제시할 산출물

- 비용항 정의와 조건부 손익분기 명제 및 모든 부호 사례.
- 계산량 대 시간차 그림: 예측선, 독립 관측, 불확실 구간, 0 기준선.
- S별 손익분기 구간 표: 예측, 실측 bracket, 무근·범위 밖·불확실 포함.
- R/N/C1의 구간별 비용과 생략한 model/risk 연산량, snapshot 크기.
- 계수 추정과 validation의 예측오차 및 burst 입력 이전 결과를 분리한 표.
- 실패·미실행 분모, 호스트·메모리·synthetic 모델과 관찰기 분리의 한계.

유리한 경계가 없더라도 설계 실패로 바꾸지 않는다. 대신 적용 범위 안에서 추가 비용을
회수하지 못했다는 결과다. 수식은 성능 보장의 대체물이 아니며 새 조건의 예측을 얼마나
정확히 설명하는지가 검증 대상이다. 구현 전에 이미 새로운 novelty가 입증됐다고 쓰지 않는다.

## 방법론 근거

[ACM SIGSOFT의 Benchmarking 기준](https://www2.sigsoft.org/EmpiricalStandards/docs/standards)은
대표성·공정성·반복 가능성을 요구하고 특정 방법에 맞춘 벤치마크와 불충분한 반복을
경계한다. 이 문서의 통제용 도메인 벤치마크와 현장 사례의 구분은 그 취지에 따른다.

[Kalibera와 Jones의 성능평가 연구](https://kar.kent.ac.uk/33611/45/p63-kaliber.pdf)는
측정 변동성과 효과 크기의 불확실성을 다룬다. 위의 구체적 grid, 10% 정밀도 목표,
N 상한 및 두 역할 설계는 본 연구의 사전 선택이며 해당 문헌이 보장하는 수치가 아니다.
