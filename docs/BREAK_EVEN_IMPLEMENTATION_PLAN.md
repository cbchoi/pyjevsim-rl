# 손익분기 실험 구현과 추적 계획

이 문서는 [실험 설계](BREAK_EVEN_DESIGN.md)의 BE001부터 BE014까지를 구현할
SDD, IDD, 직접 시험 사례와 작업 순서를 정의한다. 2026년 10월 4일 현재 설계 문서만
작성했으며 당시의 새 모듈과 시험은 예정 산출물이었다. 2026년10월5일의 후속 실행
요청에 따른 구현·직접 시험·실행 상태는 [진행 기록](BREAK_EVEN_PROGRESS.md)에
분리한다. 과거 통과 시험을 새 구현의 통과 근거로 재사용하지 않는다.

## 현재 코드 감사에서 확인한 제약

| 현재 위치 | 관측 내용 | 설계상 처리 |
|---|---|---|
| `bench/continuation_study/run.py`의 `execute_arm` | source를 닫고 B회 복원, R은 B회 fresh | 실제 호출 수를 비용식에 포함 |
| 같은 함수의 application endpoint | projection, digest, receipt 및 cleanup 포함 | 과거 endpoint 보존, 새 명칭의 observer-free endpoint 구현 |
| `bench/continuation_study/cases.py`의 `physical_projection` | 매 step 누적 history를 JSON 복사 | companion에만 사용, timing에서 사후 차감하지 않음 |
| `bench/research/cost.py`의 `action_plan` | branch마다 쓰지 않는 prefix 목록도 다시 생성 | 새 plan을 한 번 만들어 전달 |
| `bench/research/cost_analysis.py` | v1의 330 arms/110 cells를 명시적으로 강제 | 새 schema와 별도 analyzer, old 분석기 수정 금지 |
| `bench/research/inventory_adapter.py` | closed config와 exact object shapes | 새 모델 버전과 새 adapter 필요 |
| `src/pyjevsim_bridge/rl/continuation/coordinator.py` | step admission, fresh/capture/restore 검증 | 제품 계약이므로 시간 측정에 유지 |
| `bench/worker.py`와 `bench/runner.py` | CPU/application/process endpoint가 서로 다름 | 새 대응 wall/CPU를 명시, process wall 별도 보존 |

이는 읽기 전용 코드 감사 결과다. 기존 application 시간을 곧바로 새 workflow 시간으로
재명명하거나 기존 자료의 projection 비용을 빼서 새 근거로 사용하지 않는다.

## 소프트웨어 설계

| SDD 구성요소 | 예정 파일 | 책임과 금지 사항 |
|---|---|---|
| D01 Domain | `bench/research/break_even_domain.py` | 제품 table, risk kernel, forecast draw, reward, input generator; 방법 이름으로 분기 금지 |
| D02 Bindings | `bench/research/break_even_adapter.py`, `break_even_native.py` | C1 선언과 N sidecar, 동일 모델·상태 의무, 독립 oracle와 제품 graph 분리 |
| D03 Design | `bench/research/break_even_design.py` | 프로토콜 해석, 명시적 action plan, family/순열/arm 분모 생성 |
| D04 Measurement | `bench/research/break_even_run.py` | companion 또는 timing 역할, 연속 wall/CPU, coarse phases, source 종료와 B복원 |
| D05 Analysis | `bench/research/break_even_analysis.py` | 계수·부호·근, family bootstrap, 독립 확인점 선택과 N, 검증 잔차 |
| D06 Integration | 기존 `run_research.py`, `bench/worker.py`, `bench/runner.py`의 작은 분기 | 새 experiment kind만 추가, v1 의미·분모 보존, 설치·시험 자동 실행 추가 금지 |
| D07 Reporting | 새 결과 폴더와 후속 결과 문서 | 모든 비교·실패·무근·한계, 기존 코호트와 분리 |

공통 core, native vendor, 기존 Q/M과 inventory V1/V2, v1 kernel 및 v1 분석기는
변경하지 않는다. 기존 worker/runner의 범용 소유권·종료 기능은 재사용하되 native
Windows Job과 Linux process group의 실제 지원 범위를 확인하여 보고한다.
기존 callback observer를 복사해 새 harness를 크게 만들지 말고 필요한 backend와
새 endpoint만 명시적으로 연결한다. timer 단위는 초, clock은 monotonic이다.

### 호출 관계

```text
run_research의 새 명시적 단계
  -> D03 protocol과 실행 목록
  -> 기존 sequential owned worker
       -> D04 role 선택
            -> D01 동일 domain 함수
            -> R fresh replay / D02 N restore / D02 C1 restore
       -> endpoint 종료 뒤 compact receipt
  -> companion의 정확 비교와 timing의 최종 요약 대조
  -> D05 계수 추정 또는 고정 예측 검증
  -> D07 결과와 실패 분모
```

companion과 timing은 같은 입력을 가진 서로 다른 실행이다. mode에 따른 차이가
observer·counter 이외에 domain 코드로 전파되어서는 안 된다. timing의 부하 자체를
덜어 주는 검증 비활성화나 모델 cache 도입은 허용하지 않는다.

## 인터페이스 정의

아래 함수명과 필드는 설계 당시의 구현 계약이다. 새 kind는 `break-even-v1`이다.
구현 진입점은 `run_research.py --stage break-even`이며 실제 Python 함수의 인자는
아래 대응표로 추적한다.

### 계획과 arm

```text
make_plan(protocol, stage, calibration_selection=None) -> Plan
make_input(input_structure, family_seed, factors) -> materialized InputTape
make_action_plan(action_seed, B) -> immutable PrefixActions and BranchActions
execute_arm(ArmSpec, InputTape, ActionPlan, runtime_services) -> ArmReceipt
compare_companions(three receipts, transient projections) -> ExactCellReceipt
fit_calibration(complete timing cells) -> FitReport
select_validation(FitReport, protocol) -> PredictionManifest
analyze_validation(PredictionManifest, new cells) -> ValidationReport
```

| 계약 | 실제 구현 |
|---|---|
| 입력 생성 | `break_even_domain.make_input(input_structure, input_seed)`; K/S는 configuration에 별도 포함 |
| 실행 | `break_even_run.execute_arm(spec, case, campaign, budget, modules=None)`; case 안에 입력·action plan |
| 셀 대조 | `break_even_campaign._cell_summary(spec, packets)`; 전체 값을 메모리에서 비교 |
| 분석 | `fit_calibration(plan, rows, cells, protocol=...)`, `select_validation(fit, protocol)`, `analyze_validation(predictions, plan, rows, cells, cohort=..., protocol=...)` |

collection·자원 accounting 연결은 새 `break_even_campaign.py`가 담당한다. 기존 worker는
직접 생성한 자식 프로세스 handle만 관리하며, 이번 경로에서 Windows Job/Linux process
group 전체 소유권을 제공한다고 주장하지 않는다. 모델이 손자 프로세스를 만들지 않는
단일 worker 연구이며 lifetime memory는 미확인으로 기록한다.

Plan에 schema, stage, endpoint revision, protocol hash, source identity,
input-generator version, forecast algorithm version, environment, seed 규칙,
method/role 순서, 계획된 셀·arm 수와 예산 상태를 기록한다. `execute_authorized=false`
상태의 설계 JSON을 단순히 넘겨 실험을 시작하지 않는다. 실제 실행 요청을 받으면
별도 명시적 stage 명세에서 승인된 자원값과 output 경로를 설정한다.

ArmSpec 필수 필드:

```text
arm_id, cell_id, stage, family_id, input_seed, forecast_seed, action_seed,
K, S, B, prefix_steps, suffix_steps, delta, method, role,
input_identity, action_identity, source_identity, global_order,
method_order, role_order, owned_scratch_root, remaining_budget
```

method는 R/N/C1, role은 companion/timing이다. 동일 cell의 6개 arm은 role과
method 및 실행순서 외의 workload가 같아야 한다. action/input 생성은 timing 밖이다.

### 결과와 분모

ArmReceipt는 status와 error, 종료 코드, cleanup 범위, source/config/input/action
identity 및 실제 정상 출력 요약을 포함한다. timing만 workflow wall, 같은 구간 CPU,
phase 합계·횟수·unclassified 값을 성능 필드로 기록한다. companion만 exact projection과
phase별 callback/risk counter를 연구 측정으로 기록한다. 역할에 맞지 않는 필드는 null이다.

관측된 final output은 제품 observation/reward가 이미 반환한 immutable scalar 값을
동일한 consumer로 보존한 것이다. 그 소비·보존 비용은 시간 안에 포함한다. 추가 graph
순회·직렬화·full-state 복사나 post-close 접근을 하지 않는다. 실제로 읽을 수 없는 필드는
null과 이유를 기록한다. observer-free 실행의 모든 step 물리 상태를 관측한 것처럼
`exact_trace=true`를 붙이지 않는다. endpoint 통계와 정상 종료는 별도 판정이다.

각 셀은 companion complete/exact, timing complete/normal_output_agreement,
identity agreement를 별도 표시한다. 모든 셀이 완료되고 오류가 없어야 코호트 전체
수용이 true다. source receipt는 실행 소스의 근거이지 완전한 환경 인증이 아니다.

FitReport는 방법·S별 계수, 설계행렬 rank, 잔차, bootstrap 설정 및 모든 root-state
분포를 포함한다. PredictionManifest에는 calibration hash, 선택된 6좌표와 선택 이유,
예측 T_R/Delta, s_max, N 계산 및 feasibility 판단을 넣는다. ValidationReport는
6개 primary의 paired seconds, 고정 예측오차와 CI, 방향/정밀도 판정, 보조 비교를 포함한다.

공통 상태 값은 `planned`, `succeeded`, `failed`, `unexecuted`로 둔다. 분석의 root 상태는
`finite_in_domain`, `outside_supported_domain`, `always_faster_in_domain`,
`never_faster_in_domain`, `tie`, `singular_fit`, `crossing_not_demonstrated`를 구별한다.
무근·singular 상태를 숫자 0이나 임의의 아주 큰 손익분기점으로 대체하지 않는다.

### 보관 형식

```text
results/<새 study id>/<calibration|validation|transfer>/
  protocol.json, environment.json, execution.json
  arms.jsonl, cells.jsonl
  fit.json 또는 predictions.json 및 validation.json
  findings.ko.md
  transient/  # 소유한 일시 snapshot와 projection, 완료 후 정리
```

성공한 큰 trace/snapshot은 보관하지 않는다. timing runtime와 snapshot은 endpoint
안에서 정리하고 보존한 scalar는 종료 후 비교한다. companion projection은 부모의
정확 비교 후 정리한다. timing snapshot/runtime을 비교까지 보존하지 않는다. 기록별
raw compact 시간값, 입력 재생 정보, model/source version, 정확 비교 receipt와
실패 원인을 남긴다. 해시만으로 원래 full trace를 복원할 수 없음을 알린다.
raw arm 시간값을 버리고 평균만 보관하지 않는다. 공개나 원격 push는 별도 요청 없이는 하지 않는다.

## 직접 시험 사례

아래는 구현 후 수행할 STD 세부 사례이며 현재 통과했다고 보고하지 않는다.

| 시험 | 입력과 절차 | 예상 결과 |
|---|---|---|
| TC01 | 작은 동일 tape/K/S/action을 세 backend에 전달하고 호출 identity를 대조 | 같은 kernel과 scenario 수, 방법별 별도 부하 없음 |
| TC02 | cut0, 수요 직후, 입고 대기 중, 입고·수요 동시 시각의 새 action을 비교 | R/N/C1의 full companion 관측 일치, 선언한 동시사건 규칙 유지 |
| TC03 | B4와 B16의 companion phase counter를 비교 | main prefix R=B,N=C1=1; restore N=C1=B; capture/restore model callback0 |
| TC04 | q1..16 결과, A/B/A, 마지막 제품 레코드 개입, 입력 snapshot hash 대조 | label이 아닌 fulfilled 차이, A 반복 일치, branch/source 격리 |
| TC05 | K1/16와 S8/512를 교차하고 독립 scalar oracle와 비교 | risk 보상 기여, 레코드 수 고정, scenario scratch 미저장, 마지막 레코드 보존 |
| TC06 | fake monotonic clock과 spy backend로 전체 lifecycle 수행 | 초기화·제품 검증·cleanup 포함; observer·receipt 제외; phase 비중첩 |
| TC07 | 알려진 F/D 부호 9조합과 integer boundary, zero slope, singular matrix | 수식의 정수 우위 집합 정확, finite root만 선택하는 오류 검출 |
| TC08 | seed 중복과 validation row를 fit 입력에 삽입, midpoint tie와 무근 생성 | 누수 거부, tie는 작은 K, 무근도 고정된 후보 규칙 유지 |
| TC09 | 각 family 내 큰 상관을 가진 합성 시간 배열과 일부 무근 bootstrap | family 단위 공동 재표집, 6개 primary 구간수준, 무근 draw 분모 보존 |
| TC10 | arm timeout, projection mismatch, 종료 실패, 중간 deadline 주입 | 재시도 없음, 부분 분모 보존, 전체 수용 false, 성공 통계로 위장하지 않음 |
| TC11 | 변경 파일과 v1 manifest/분모 경로를 확인 | 과거 결과 및 원본 bytes 보존, 새 kind 외 동작 변경 없음 |
| TC12 | 계획된 owned scratch, 이미 존재하는 output, 예산 소진 상태 | 기존 output 거부, 소유 범위만 정리, 미확인 memory를 pass로 쓰지 않음 |
| TC13a | 알려진 비용함수의 합성 calibration과 별도 biased validation 입력 | 참 계수 복구, 사전 예측 유지, 오차·불확실·모형 실패를 그대로 출력 |
| TC13b | 새 실제 family의 고정된 6개 holdout 좌표와 사전 예측 | 실제 signed-seconds 잔차와 구간, 교차/무교차/불확실을 모두 보고 |
| TC14 | no-root/negative/partial cohort 보고서 fixture | 실패·한계·N 비교 누락 없음, 실제 성능/인간/RL 범위 과장 없음 |

TC07/09/13a의 합성 숫자 시험은 분석기 정확성 시험이지 시뮬레이션 성능 결과가 아니다.
부하 범위가 너무 비싸거나 source 구현 오류가 나타나면 버전을 수정하고 원인을 기록한다.
측정 중 수정한 코드를 같은 코호트에 섞지 않는다.

## 마일스톤과 병렬 작업

| 마일스톤 | Task | 선행과 병렬성 | 완료 조건 |
|---|---|---|---|
| MS38 설계 | TASK207 | 현재 작업 | 비용식·SRS·STD·SDD·IDD·계획·독립 검토를 문서화 |
| MS39 구현 | TASK208 모델과 bindings | 설계 후 TASK209/210과 병렬 | D01/02, TC01..05와 새 field 의무 |
| MS39 구현 | TASK209 계측과 실행 연결 | TASK208의 IDD 공유, TASK210과 병렬 | D03/04/06, TC06/10/11/12, 실제 endpoint 명세 일치 |
| MS39 구현 | TASK210 수식과 통계 분석 | 순수 합성 자료로 병렬 | D05, TC07/08/09/13/14, 모든 root 상태 |
| MS39 검토 | TASK211 통합과 직접 검토 | TASK208..210 후 | 직접 관련 시험과 독립 검토, 기존 자료 보존, 실행 예산 판단 |
| MS40 계수 | TASK212 calibration | 새 실행 요청·예산 결정 후 순차 | 864회와 144개 companion exact 셀 또는 실패·부분 보고 |
| MS40 예측 | TASK213 validation과 transfer | 계수와 선택/N/예산 기록 후 순차 | 36N+216회, 고정 예측 검증, 미완료와 무근 보존 |
| MS41 논문 | TASK214 결과와 논문 반영 | 연구 수집 종료 후 검토 병렬 가능 | 계산식·실측·예측오차·제한 일치, 결과 문서와 commit |

각 task는 `MS번호-L횟수`로 Plan–Do–Review–Reflect를 기록한다. Plan은 코드·설계·
기존 reflect를 확인하고, Do는 해당 범위만 구현하며, Review는 직접 요구사항 증거를
검토하고, Reflect는 미해결 원인과 다음 수정을 기록한다. 같은 목적의 보완 task가
동일 milestone에서 10회 이상 추가되면 루프를 중단하고 판단을 요청한다.

설계 요청 당시에는 TASK207만 수행했다. 2026년10월5일의 실행 요청에 따라
후속 task를 진행한다. task 완료마다 명시 경로만
`cbchoi with claude <me@cbchoi.info>` 작성자와 committer로 commit한다. 다른 저장소의
변경·staged 상태에 접근하거나 push하지 않는다.

## 요구사항 추적성

| 요구사항 | 설계 | 직접 또는 연구 시험 | 작업 |
|---|---|---|---|
| BE001 | D01/02/03 | TC01 | TASK208/209 |
| BE002 | D02/04 | TC02 | TASK208/212/213 |
| BE003 | D04 | TC03 | TASK208/212/213 |
| BE004 | D01/02/03 | TC04 | TASK208 |
| BE005 | D01/02/04 | TC05 | TASK208/209 |
| BE006 | D04/06 | TC06 | TASK209 |
| BE007 | D05 | TC07 | TASK210 |
| BE008 | D03/05 | TC08 | TASK209/210 |
| BE009 | D05 | TC09 | TASK210/213 |
| BE010 | D04/06/07 | TC10 | TASK209/214 |
| BE011 | D06 | TC11 | TASK209/211 |
| BE012 | D03/04/06 | TC12 | TASK209/211 |
| BE013 | D05/07 | TC13a/13b | TASK210/213 |
| BE014 | D07 | TC14 | TASK210/214 |

## 현재 단계의 미확인 사항

새 K/S 모델의 실제 실행시간과 상태 bytes, N 산출값, 예산 내 완료 가능성, Linux
환경과 외부 부하, 시간 측정 실행의 full trace는 아직 확인하지 않았다. 모델의 단순
선형 비용식이 실제 구현에 맞는지도 결과가 아니다. 이 항목들은 설계가 누락한 성공
조건이 아니라 이후 연구에서 답해야 할 질문이다.

## 2026년10월5일 구현·실행 상태

| Task | 상태와 직접 근거 |
|---|---|
| TASK208 | 구현 완료, 새 domain·bindings·oracle 의미 시험12개 통과 |
| TASK209 | 구현 완료, planner/endpoint/실제6-arm/isolated worker20개 통과; collection/entry/기존 owned-process16개 통과 |
| TASK210 | 분석 구현 완료, 합성 통계 시험22개 통과 |
| TASK211 | 직접 검토 완료; 관련70개는 task별 시험 합계, 전체 시험 일괄 실행 아님 |
| TASK212 | 144/864회·24/144셀·1/6family 후 사전 자원 예측 기준에 따른 부분 중단, 실패 arm0 |
| TASK213 | 미실행; N/고정 예측 미선정, TC13b 미검증 |
| TASK214 | 부분 관측·비용·제한 보고; 논문 결론의 확증 승격은 보류 |

구현 시험은 실제 성능의 신뢰구간을 대신하지 않는다. 자세한 원자료 위치·중단 산식·
전체24개 조건은 [부분 결과](BREAK_EVEN_RESULTS.md), loop 기록은
[진행 기록](BREAK_EVEN_PROGRESS.md)을 참조한다. 새 실행이나 예산 변경은 승인 전 수행하지 않는다.
