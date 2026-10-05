# Continuation 실행 비용과 재사용성 개선 설계

2026년 10월 5일 사용자의 개선 계획 실행 요청에 따른 SRS, STD, SDD, IDD와 작업 계획이다.
현재 strict 구현과 과거 결과를 보존하면서 실행 비용, 모델 작성 부담, 의사결정 효용을
구분하여 개선한다. Darpan과의 비교는 연구 동기이며 새 기능의 독창성 증명은 아니다.
기존 의미 검증 18조건 및 재고 모델 54셀은 완료된 근거이고 재수행을 신규 성과로 세지 않는다.

## 요구사항과 직접 시험

| 요구사항 | 구현 의무 | 직접 시험과 연구 판정 |
|---|---|---|
| IMP001 | 기본 strict 실행 경로와 기존 snapshot 의미 유지 | 기존 API 기본값, 검사 호출, 원본 결과 보존 |
| IMP002 | 정상 step의 반복 검증 비용을 별도 진단 | cProfile 호출별 누적·자체 시간, 계측 시간은 성능 추정에서 제외 |
| IMP003 | 명시적 admitted 실행 계약 | 런타임 소유권·registry 세대·수명 검사, full 경계 검사 유지, 서로 다른 보장 범위 기록 |
| IMP004 | strict와 admitted의 정상 모델 동등성 | 같은 입력의 상태·관측·보상·종료·분기 격리 및 손상 snapshot 거부 |
| IMP005 | 명시적 단독 소유 값 필드 작성 도우미 | exact keys, detached values, 전체 validation 후 대입, alias/RNG/calendar 자동 추론 금지 |
| IMP006 | 도우미를 실제 모델 어댑터에서 사용 | 기존 inventory의 별도 profile, 원본 모델·어댑터 불변, N/oracle 동등성 |
| IMP007 | observer 민감도와 framework 거부 구분 | 실제 미래 calendar 손상과 거부·원본/형제 격리의 증거를 별도 기록 |
| IMP008 | 공정한 탐색 비용 비교 | R/N/strict C1/admitted C1A, 동일 model/input/action, 역할별 companion/timing 분리 |
| IMP009 | 의사결정의 독립 평가 | 공통 후보 순서, 실제 비용이 있는 목적함수, 평가 입력 독립, budget overrun과 실패 보존 |
| IMP010 | 기존 독립 손익분기 검증의 완결 준비 | 코호트 간 혼합 금지, 버전별 재보정, 본 실험 호스트·단계 예산 사용자 답변 반영 |
| IMP011 | 사람 연구를 실행 가능한 과제로 준비 | matched 신규 과제·판정표·지원 동일화, 실제 참여 없이 생산성 수치 생성 금지 |
| IMP012 | 계획과 실제 성과 구분 | task별 Plan Do Review Reflect, 실패·미실행·제한 및 명시 경로 commit |

## 실행 프로파일 설계와 인터페이스

`ContinuationCoordinator(registry, *, execution_profile="strict-v1")`를 제공한다.
허용 값은 `strict-v1`과 `admitted-runtime-v1`뿐이다. `RuntimeHandle.execution_profile`로
현재 계약을 읽는다. 기존 호출의 기본 동작은 변하지 않는다.

admitted 프로파일의 handle은 full fresh 또는 restore 수용 후 registry entry witness를
소유한다. 정상 step은 기존 runtime/env 잠금, READY 검사, registry/bundle/profile 객체와
세대·runtime 소유권 확인 후 기존 env.step을 실행한다. 기존 env의 action, clock,
observation, reward, termination, failure 동작을 변경하지 않는다. register, capture 전후,
restore 전체, inspect, semantic_view는 기존 full 검증을 유지한다. 기존 ValidationContext를
검증 결과 캐시로 바꾸지 않는다. 실행 로직은 이미 source-bound인 coordinator/registry에 둔다.

admitted는 trusted model 계약이다. handle 수명 중 설치 source/provider/callback/validator
변경 및 graph/engine/env 내부 직접 수정은 지원하지 않는다. 정상 전이가 선언된 불변식을
보존해야 한다. 임의 Python 변조를 봉쇄하는 sandbox가 아니며 full 경계 사이 내부 변조의
즉시 탐지를 주장하지 않는다. 외부 개입은 정상 action 경로로만 전달한다. 검증 주기와
보장 범위가 strict와 다르므로 단순한 동일 보장 최적화라고 보고하지 않는다.

프로파일은 snapshot 의미 payload를 변경하지 않고 실행 메타데이터에 명시한다.
bench의 C1은 strict, C1A는 admitted로 표시한다. N은 기존 fast native 그대로 유지한다.
필요한 추가 보장을 N에 구현하는 실험은 별도 방법이며 기존 N을 불리하게 교체하지 않는다.

## 값 필드 작성 지원 인터페이스

`OwnedValueField(name, validate)`와 `DeclaredValueFields(fields)`를 제공한다.
메서드는 `capture(owner)`, `validate(payload)`, `restore_into(owner, payload)`다.
명시한 직접 속성과 닫힌 유한 JSON 값만 처리하고 전체 payload 검증을 마친 뒤 대입한다.
공유 container root, RNG, scheduler, topology, clock, rebind는 모델의 명시적 책임으로 남는다.
이 도우미는 숨은 상태의 완전성을 증명하지 않는다. 별도 declared inventory adapter에서
실제 사용하되 기존 inventory를 새로운 held-out 모델이라고 부르지 않는다.

## 진단과 성능 연구

Root가 `run_improvement.py`와 `bench/research/improvement_study.py`를 연결한다.
진단은 cProfile을 사용하는 별도 실행으로 full admission, source 확인, export/복사,
실제 모델 계산의 호출 수와 시간을 기록한다. 진단 중 profiler가 만든 시간은 speedup으로
사용하지 않는다. 계측되지 않은 탐색 timing과 별도 companion 비교를 수행한다.

초기 탐색 grid는 K1/4096, S8, B4/16, prefix64, suffix16, 세 독립 family다.
R/N/C1/C1A와 companion/timing의 96 arms, 12 exact cells를 계획한다. 방법 순서는 family와
조건에 따라 교대하고 전체 균형 여부를 실제 보고한다. 작은 표본이므로 확증으로 부르지 않는다.
worker1/BLAS1, 새 탐색 시간 최대600초·저장16MiB, arm120초를 사용한다. 진단은 별도60초다.
기존 연구 스크립트의 모델·측정 함수를 backend injection으로 재사용하며 과거 analyzer와
코호트는 변경하지 않는다. 같은 연구 중 실행 코드를 수정하거나 실패 arm을 교체하지 않는다.
실행 중 모델·시험·설치·큰 분석을 병렬 실행하지 않는다. 현재 Windows는 condition unspecified,
외부 간섭 및 전체 lifetime memory 미확인이다. 큰 trace는 비교 후 보존하지 않는다.

기존 본 손익분기 연구의 총9000초·32MiB 안에서 단계4200/3000/1800초 재배분과 호스트는
사용자에게 질문한다. 답변 전에는 본 연구를 시작하지 않는다. 기존 부분 코호트는 재개하지 않는다.
최적화 버전을 평가하려면 새 실행 프로파일을 명시하고 calibration부터 다시 수행한다.

호스트 답변은 전용 Linux로 확정됐다. 단계 예산 재배분 답변과 접속/결과 전달 방식은
아직 없으며, Windows 본 연구를 대신 실행하지 않는다. `run_improvement.py`는 위의
작은 탐색 연구용이다. C1A를 포함하는 full calibration/validation/transfer의 독립 확증
설계와 파이프라인 확장은 별도 미완료 범위로 유지한다.

## 의사결정 사례연구

목적은 snapshot 기반 후보 평가가 실제 선택에 주는 효용이다. 기존 risk benchmark의
항상 큰 주문량을 유리하게 만드는 조건을 그대로 성과 근거로 사용하지 않는다. 기존 inventory
구성의 지원 범위에서 prefix를 고정하고 이후 수요를 분리한 새로운 문제와 구매·보유·품절 비용을
명시한다. 후보 행동은 실제 simulation 경로로 적용한다. 평가용 미래 입력은 후보 선택에 노출하지
않으며 예측 목적함수와 실제 평가 손실을 분리한다. 기존 모델 변경이 필요하면 별도 버전으로 둔다.

고정 후보 비교와 고정 wall 예산 비교를 분리한다. N/C1은 setup/prefix/capture를 한 번 비용에
포함하며 R은 후보마다 fresh/prefix를 비용에 포함한다. 시간 초과 후보는 선택에서 제외하고 이미
소비한 시간과 초과를 남긴다. 부드러운 경계 관측이며 hard realtime 보장이 아니다. 독립 평가와
oracle 생성 비용은 별도 기록한다. 이 사례는 정책 학습이나 수렴의 증거가 아니다.

## 마일스톤과 작업

| 마일스톤 | Task | 담당과 선행 | 완료 조건 |
|---|---|---|---|
| MS42 | TASK215 설계 및 baseline 진단 | Root, 변경 전 진단 | 계약·분모 명시, 실제 profile 기록 |
| MS43 | TASK216 admitted runtime | 실행 경로 agent | 명시 opt in, strict 기본 보존, 직접 시험 |
| MS43 | TASK217 모델 작성 지원 | 상태 계약 agent, TASK216과 병렬 | helper와 실제 adapter, exact/negative 시험 |
| MS43 | TASK218 의사결정 사례 | 연구 agent, 독립 파일 | 실질적 objective와 독립 평가, 직접 시험 |
| MS44 | TASK219 통합 비용 평가 | Root, 위 구현 종료 후 순차 | 96 arms/12셀 또는 부분 분모·오류 보존 |
| MS44 | TASK220 본 손익분기 연구 | 호스트·예산 판단 및 고정 버전 후 | calibration/validation/transfer 별 수용 또는 부분 보고 |
| MS45 | TASK221 개발자 과제 준비 | 결과 분석과 병렬 가능 | 신규 A/B task pack과 rubric, 실제 세션은 별도 조율 |
| MS45 | TASK222 의사결정 및 RL 효용 | 비용·의미 근거 후 | 의사결정 실제 결과, RL 추가 설계·구현은 해당 조건 확정 후 |

각 task는 `CONTINUATION_IMPROVEMENT_PROGRESS.md`에 MS번호와 loop 횟수의
Plan Do Review Reflect를 남긴다. 동일 목적 보완 task가 동일 milestone에서 10회 이상
추가되면 중단하고 판단을 요청한다. task 완료마다 명시 경로만 author/committer
`cbchoi with claude <me@cbchoi.info>`로 commit한다. push·공개·타 저장소 수정은 하지 않는다.
직접 관련 시험만 수행하며 자동 smoke/freeze/전체 시험을 실험 시작 조건으로 추가하지 않는다.

## 구현과 결과 추적 상태

2026년10월5일의 실제 상태다. 상세 분모와 실패는
[결과 보고서](CONTINUATION_IMPROVEMENT_RESULTS.md) 및
[loop 기록](CONTINUATION_IMPROVEMENT_PROGRESS.md)에 보존한다.

| 요구사항 | 근거 경로 | 현재 판정 |
|---|---|---|
| IMP001∼004 | coordinator.py, registry.py, test_admitted_runtime.py, test_break_even_domain.py | 구현과 직접 시험 완료, 성능 확증 아님 |
| IMP005∼006 | state_fields.py, declared_inventory_adapter.py, test_declared_state_fields.py | 기존 모델 공통화 완료, 신규 모델 일반화 아님 |
| IMP007 | test_future_calendar_contract.py | 명시 범위의 관측·거부·격리 시험 완료 |
| IMP008 | improvement_study.py, run_improvement.py, test_improvement_study.py | 55/96 실행과 exact6/12 부분 결과 보존, admission false |
| IMP009 | decision_utility.py, test_decision_utility.py | 탐색 24/24 완료, 학습 효용 미평가 |
| IMP010 | run_research.py, 본 설계의 Linux 이관 조건 | 본 연구와 C1A full 파이프라인 미완료 |
| IMP011 | docs/developer_tasks/ | 명세만 완료, 실행형 과제·사람 연구 미완료 |
| IMP012 | CONTINUATION_IMPROVEMENT_PROGRESS.md, task별 local commit | 실제 결과와 제한 기록, 공개·push 없음 |

파일명은 src/pyjevsim_bridge/rl/continuation, bench/research, tests 안의 해당 경로를
가리킨다. MS42/43은 완료, MS44는 탐색 부분 종료·본 연구 대기, MS45는 과제 명세와
의사결정 사례만 완료 상태다. 전체 계획의 완료로 올리지 않는다.
