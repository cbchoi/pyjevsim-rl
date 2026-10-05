# 과제 A — 두 수조 지연 주입 모델의 복원 통합

TASK221-A-v1 / **설계 명세; 실행 starter와 채점기는 아직 없음.**
공통 절차는 [README](README.md), 판정은 [RUBRIC](RUBRIC.md)을 따른다.

## 제공할 원본 모델

두 개의 정적 수조 구성요소 `tank0`, `tank1`이 하나의 mutable `Meter`를 공유한다.
모델 생성·사건 처리·정상 관측·action 함수는 연구자가 검증한 원본으로 제공한다.
참여자의 첫 작업은 물리 모델 재구현이 아니라 지정 처리의 capture/restore 통합이다.

- 수위 초기값 `[5, 4]`, 수조별 용량 `12`, 공통 주입 예산 `20`.
- action `{"q0": a, "q1": b}`의 각 값은 정수 `0..4`. 두 값 모두 0이면 no-op.
- 양수 q는 해당 수조에 진행 중인 주입이 없을 때만 허용한다. 두 수조의 합계가 예산
  이내인지 등 전체 action을 먼저 검사하며, 잘못된 action은 아무 상태도 바꾸지 않는다.
- 수락한 주입은 예산 q를 즉시 차감하고 `pending=(request_id,q,due=t+1.5)`를 만든다.
  완료 시 수위에 q를 더하되 용량을 넘는 분량은 `spilled`에 누적한다. 예산은 환불하지 않는다.
- 유출 사건 `(time,tape_id,tank,base)`마다 공통 LCG 상태를 한 번 진행시키고
  `demand=base+(state & 1)`로 정한다. 실제 공급은 `min(level,demand)`, 차이는 `unserved`다.
- 난수 진행 규칙은 `state=(1664525*state+1013904223) mod 2**32`.
  최초 seed만 저장하여 cut까지 재실행하는 것은 금지한다. 현재 상태를 복원한다.
- 동일 논리시간에는 주입 완료를 수조 ID 순서로, 이어서 유출을 tape ID 순서로 처리한다.
  committed cut에서 새 action을 적용한 뒤 다음 advance를 실행한다. 이미 처리된 같은 시각
  사건보다 앞으로 action을 이동시키지 않는다.
- 관측은 clock, 각 수위, 예산, 누적 공급/미공급/넘침, pending의 ID/양/완료시각이다.
  V1 누적 목적량 `J=served-2*unserved`; step reward는 `J-last_J`이고 이후 `last_J=J`.
- delta는 0.25, 기본 종료시각은 4.0이다. 종료 여부·step cursor·관측 cache와 보상 baseline도
  연속 실행과 같아야 한다. alias와 난수 상태는 물리 비교용 observer가 별도로 확인한다.

## 공개 fixture 명세

| 항목 | 설정 |
|---|---|
| 초기 seed | 11 |
| prefix | t=0에 q0=4,q1=0 적용 후 t=1.0까지 no-op advance |
| 주요 cut | t=1.0: tank0 주입이 t=1.5에 완료될 예정 |
| suffix A | cut에서 q0=0,q1=1, 이후 no-op |
| suffix B | cut에서 q0=0,q1=3, 이후 no-op |
| 추가 cut | t=0 초기 상태, t=1.5 동시 사건 처리 후 |

유출 tape는 `(time,id,tank,base)` 목록
`[(0.5,0,0,2),(1.0,1,1,2),(1.5,2,0,3),(2.0,3,0,2),
(2.5,4,1,4),(2.5,5,0,1),(3.0,6,1,2),(3.5,7,0,3)]`다.
cut 이후의 q1 주입은 t=2.5에 완료되어 같은 시각 유출보다 먼저 적용된다.
fixture는 입력 명세이며 검증된 expected trace 파일이 이미 있다는 뜻이 아니다.

비공개 입력은 동일 문법·경계 내의 다른 seed, 두 수조 활성화 순서, 유출값,
pending 없는 cut, 두 pending이 있는 cut, 예산 한계 및 종료 직전 cut을 포함한다.
비공개 범주를 숨기지 않되 실제 값과 정답은 과제 제출 전에 제공하지 않는다.

## 단계 1 — 통합, 40분

양 처리 모두 다음을 제출한다.

1. committed cut의 전체 미래 관련 상태를 저장하고 별도 runtime으로 복원하는 코드.
2. 정상 action API로 새로운 개입을 주입하는 사용 예제와 상태 소유권 목록.
3. 원본과 동시에 살아 있는 분기 A/B/A가 서로 상태·원장·난수·cache를 공유하지 않는 구현.
4. 같은 분기 안의 두 수조는 **하나의 동일 Meter 객체**를 참조하도록 rebind하는 코드.
5. 저장·복원 중 모델의 전이/출력 callback이나 난수 draw를 실행하지 않았다는 직접 확인.

R replay는 evaluator의 기준이며 참가자의 restore 구현으로 사용하지 않는다.
모델의 과거를 재계산하거나 숨은 상태를 초기값으로 대체하면 출력 일부가 같아도 불합격이다.

## 단계 2 — 유지보수, 20분 후 별도 공개

V2는 유출 때마다 수조별 `shortage_energy += unserved_this_event**2`를 누적한다.
공통 Meter에는 주입 완료 시 `(request_id,tank,delivered,spilled)`를 순서대로 남기는
`completion_log`를 추가한다. 두 수조가 각각 새 원장을 복제하여 쓰면 안 된다.

V2 누적 목적량은 `J2=served-2*unserved-shortage_energy_total`이며
step reward는 `J2-last_J2`다. 신규 accumulator와 보상 baseline, 원장의 별칭을 저장·복원한다.
V1 저장 파일을 V2로 migration할 요구는 없으며 task/schema version mismatch는
fixture 선택 단계에서 구별한다. V1 회귀는 V1 모드로 별도 실행하고 V2 reward와 같다고
요구하지 않는다. snapshot의 외부 보안 형식 전체를 새로 구현할 의무는 없다.

유지보수 판정에는 이미 shortage와 completion_log가 비어 있지 않은 V2 cut을 포함한다.
신규 필드를 항상 0/빈 목록으로 초기화하여 통과하지 못하게 한다.

## Oracle 준비 책임

연구자는 위 정수·유리수 규칙을 독립 event-list 프로그램으로 구현해야 한다.
그 구현은 pyjevsim, 참가자 adapter/native codec, 모델 callback, 공통 snapshot serializer를
호출하지 않는다. exact 비교는 clock을 quarter-tick 정수로 정규화하고 실제 callback 순서를
정의된 사건 순서에 투영한다. 객체 주소는 비교하지 않되 alias 관계 자체는 `is`로 검사한다.
hand-checked fixture와 독립 구현 검토를 완료하기 전 정답으로 사용하지 않는다.
