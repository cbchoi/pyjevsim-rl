# 과제 B — 두 축열기 지연 가열 모델의 복원 통합

TASK221-B-v1 / **설계 명세; 실행 starter와 채점기는 아직 없음.**
공통 절차는 [README](README.md), 판정은 [RUBRIC](RUBRIC.md)을 따른다.

## 제공할 원본 모델

두 정적 축열기 `buffer0`, `buffer1`이 하나의 mutable `EnergyLedger`를 공유한다.
모델 생성·사건 처리·정상 관측·action 함수는 연구자가 검증한 원본으로 제공한다.
대응 과제 A와 상태 소유권 의무는 맞추되 가열량과 방열 법칙은 별도로 정의한다.
이는 matched 학습 과제이지 두 독립 산업 모델의 현장 타당성을 주장하는 사례가 아니다.

- 초기 열량 `[10, 8]`, 각 용량 `24`, 공통 전기 펄스 예산 `20`.
- action `{"p0": a, "p1": b}`의 각 값은 정수 `0..4`; 모두 0이면 no-op.
- 양수 p는 해당 축열기의 미완료 가열 작업이 없을 때만 허용한다. 합계 p가 예산
  이내인지 등 전체 action을 먼저 검사하고, 잘못된 입력은 상태를 바꾸지 않는다.
- 수락 시 예산 p를 즉시 차감하고 `pending=(request_id,p,due=t+1.5)`를 만든다.
  완료 시 열량 `2*p`를 추가하고 용량을 넘는 분량은 `vented`에 누적한다. 환불은 없다.
- 열수요 사건 `(time,tape_id,buffer,base)`마다 공통 LCG 상태를 한 번 진행시키고
  `heat_demand=2*base+(state & 1)`로 정한다. 실제 공급은 `min(heat,heat_demand)`,
  차이는 `unmet`이다. LCG는 `state=(1664525*state+1013904223) mod 2**32`.
- 동일 시각에는 가열 완료를 buffer ID 순서로, 그 뒤 수요를 tape ID 순서로 처리한다.
  cut은 같은 시각 사건이 이미 끝난 committed 경계이며 새 action은 그 이후에 주입한다.
- 관측은 clock, 각 열량, 예산, 누적 공급/미충족/방출, pending의 ID/양/완료시각이다.
  V1 목적량 `J=heat_served-2*unmet`; reward는 `J-last_J`, 이후 baseline을 갱신한다.
- delta는 0.25, 기본 종료시각은 4.0이다. 현재 난수 상태, 종료 여부, step cursor,
  관측 cache와 보상 baseline을 함께 보존한다. 내부 alias/RNG도 독립 observer가 검사한다.

## 공개 fixture 명세

| 항목 | 설정 |
|---|---|
| 초기 seed | 29 |
| prefix | t=0에 p0=4,p1=0 적용 후 t=1.0까지 no-op advance |
| 주요 cut | t=1.0: buffer0 가열이 t=1.5에 완료될 예정 |
| suffix A | cut에서 p0=0,p1=1, 이후 no-op |
| suffix B | cut에서 p0=0,p1=3, 이후 no-op |
| 추가 cut | t=0 초기 상태, t=1.5 동시 사건 처리 후 |

수요 tape `(time,id,buffer,base)`는
`[(0.5,0,0,2),(1.0,1,1,2),(1.5,2,0,3),(2.0,3,0,2),
(2.5,4,1,4),(2.5,5,0,1),(3.0,6,1,2),(3.5,7,0,3)]`이다.
cut 이후 p1 작업의 완료는 t=2.5 수요에 앞선다. 이 표는 앞으로 starter/oracle에
구현할 입력이지 이미 실행하여 검증한 결과가 아니다.

비공개 입력은 다른 seed, 두 축열기 활성 순서, 수요값, pending이 없거나 둘인 cut,
예산 한계, 종료 직전 cut을 포함한다. A/B의 비공개 test 범주·개수·평가 예산을 동일하게
준비한다. domain별 배율이 다르므로 A의 expected 값을 그대로 이름만 바꾸지 않는다.

## 단계 1 — 통합, 40분

1. committed cut에서 저장하고 별도의 runtime으로 복원할 것.
2. 복원 직후 정상 action으로 새 가열을 요청하고 모든 미래 사건을 원래 순서로 진행할 것.
3. 두 축열기가 같은 EnergyLedger 객체를 참조하는 관계를 복원할 것.
4. 원본·동시 분기 A/B·다시 복원한 A의 상태/난수/cache를 서로 격리할 것.
5. 저장·복원 중 전이/출력 callback과 난수 draw를 호출하지 않을 것.
6. 누락될 수 있는 pending·보상·종료·난수·별칭의 상태 소유권을 짧게 문서화할 것.

replay와 독립 oracle는 평가자가 사용한다. 참가자의 restore가 prefix를 다시 실행하면
정상 trace 일부가 일치해도 복원 과제로 수용하지 않는다. N과 C1 모두 동일 요구다.

## 단계 2 — 유지보수, 20분 후 별도 공개

V2는 수요 사건마다 축열기별 `discomfort_energy += unmet_this_event**2`를 누적한다.
공통 EnergyLedger에는 가열 완료마다 `(request_id,buffer,heat_added,vented)`를 순서대로
남기는 `completion_log`를 추가한다. 원장은 한 runtime 안에서 여전히 공동 소유한다.

V2 목적량은 `J2=heat_served-2*unmet-discomfort_energy_total`; reward는
`J2-last_J2`다. 신규 accumulator·log·baseline을 복원하고 새로운 요구를 제외한
기존 물리 전이와 V1 모드는 보존한다. V1 snapshot을 V2로 migration하는 문제는 아니며
fixture가 schema/version을 구분한다. 일반 적대적 artifact 방어 구현은 공통 과제 밖이다.

유지보수의 cut에는 신규 accumulator와 log가 이미 비어 있지 않은 경우가 포함된다.
초기화만 추가하는 구현이 아니라 **누적한 상태에서의 continuation**을 평가한다.

## Oracle 준비 책임

정수 열량과 quarter-tick 시간으로 독립 event-list oracle를 구현해야 한다.
pyjevsim, 참가자 구현, 모델 callback, snapshot codec을 oracle에서 호출하지 않는다.
정의된 완료→수요→새 action 순서, 난수 draw 위치, 예산 차감 시점, V2 reward delta를
독립 검토하고 hand-checked 예제를 준비한 뒤 사용한다. 원본 replay만으로 oracle를
대체하거나 adapter의 export 출력끼리만 비교하여 독립 정확성으로 보고하지 않는다.
