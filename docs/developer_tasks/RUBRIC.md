# 개발자 파일럿 공통 평가 기준

TASK221-R-v1. [README](README.md)의 배정·시간·동의 절차와 함께 적용한다.
**이 기준에 대응하는 executable evaluator는 아직 구현되지 않았다.**

## 수용 판정과 진단 점수의 구별

통합 성공은 아래 I01–I08을 모두 충족한 제출이다. 유지보수 성공은 해당 V2의
I01–I08과 M01–M04를 모두 충족한 제출이다. 항목별 pass/fail/not-observed를 보존하며
여러 쉬운 항목의 점수 합계로 필수 상태 누락을 상쇄하지 않는다. 동일 제출을 처리마다
다른 테스트나 허용오차로 평가하지 않는다.

| ID | 양 처리의 공통 의무 | 평가자가 준비할 독립 검증 |
|---|---|---|
| I01 | cut 이후 정상 개입의 의미 | replay와 oracle의 매 step clock·관측·물리 상태·reward·종료 exact 비교 |
| I02 | pending과 동시 사건 순서 | 완료/수요가 같은 시각인 cut 전후, 실제 이벤트 순서와 다음 schedule 대조 |
| I03 | 미래 난수 상태 | capture 이후 두 개 이상 draw와 후속 결과를 replay/oracle에 대조; seed만 재설정한 mutant 검출 |
| I04 | boundary 상태 | step cursor·관측 cache·last_J·종료 직전/직후 동작 대조 |
| I05 | alias와 분기 격리 | runtime 안에서는 두 구성요소 원장 `is` 일치; 원본·A/B/A 사이 원장은 다름; 후속 행동으로 격리 확인 |
| I06 | 실제 prefix 재실행 생략 | 저장/복원 구간의 전이·출력 callback와 난수 draw 0; source 불변 비교 |
| I07 | 입력 계약과 자원 종료 | 허용 행동 정상, 잘못된 action은 원자적 거부, 생성된 모든 runtime close 확인 |
| I08 | 공통 제출 요구 | 지정 API, 필요한 상태 소유권 목록, 허용 경로만 수정; 재생 source 편집 등 금지된 우회 없음 |
| M01 | 신규 누적 상태 | 0이 아닌 accumulator의 capture/restore와 추가 누적을 독립 oracle에 대조 |
| M02 | 변경된 증분 보상 | 누적 J2와 last_J2 복원, 첫 suffix reward와 후속 reward의 exact 일치 |
| M03 | 신규 공동 log | 완료 log 순서·내용 보존, 단일 runtime alias와 분기 간 분리 유지 |
| M04 | 기존 계약 유지 | V1 모드 회귀와 V2 물리 전이, 명시적으로 변경된 reward는 V2 oracle로 따로 비교 |

exact의 시간 단위는 quarter-tick 정수, 수량/난수는 정수다. 런타임 고유 ID/객체 주소는
정의된 logical ID로 대응시키되 의미 있는 시각·순서·reward를 정규화로 지우지 않는다.
scalar projection이 같다는 것과 내부 alias/난수 상태가 같다는 것은 서로 별도 확인이다.

공개 3개 cut와 비공개 6개 cut 유형을 A/B 모두 같은 수의 입력으로 준비한다.
비공개에는 정상 범위 내의 다른 seed/순서/양을 사용하고 공개되지 않은 기능 요구를
추가하지 않는다. 상충·미정 의미는 참가자 탓으로 처리하지 말고 rubric defect로 기록한다.

## 평가기 자체의 민감도 확인

연구자는 제출 전 평가기 검증용으로 다음 고의 오류를 가진 별도 disposable 구현을 만든다.
현재 이 mutant 구현도 아직 없다.

- 미래 RNG state 대신 최초 seed 사용.
- pending 완료시각 또는 동시 사건 순서 변경.
- 증분 reward baseline 누락.
- 공동 원장을 두 객체로 복제하거나 형제 branch와 공유.
- V2 accumulator/log를 초기값으로 되돌림.

observer가 차이를 검출한 결과와 참가자 구현이 손상을 자체 거부한 결과를 혼동하지
않는다. 손상 snapshot 공격을 스스로 탐지하는 기능은 공통 필수 과제에 넣지 않는다.
framework 추가 기능은 별도 기술표에서만 보고한다.

## 제출 시각과 timeout

각 제출의 파일 내용 hash와 제출 시각을 기록한다. 평가기는 해당 제출의 불변 사본을
비동기 채점할 수 있다. 제한시간 안에 제출되어 이후 correct로 판정된 경우 completion
time은 **제출 시각**이다. evaluator 대기 시간을 participant 개발시간으로 더하지 않는다.
참여자가 결과를 기다린 실제 시간은 대기 지표로 따로 남긴다.

첫 correct 제출이 없으면 40분/20분에서 right-censored이며 완료율에서 미완료다.
timeout 숫자를 성공 completion time으로 바꾸거나 미완료 참여자를 제외하지 않는다.
최종 제출이 늦었으면 해당 예산 성공은 아니며 지연 제출로 별도 기록한다.
code runner가 멈추면 사전 fixture별 평가 timeout으로 종료하고 그 제출은 판정 불가/실패
사유를 남긴다. 제출 재시도는 참여자 활동으로 허용하되 실패 이력을 삭제하지 않는다.

## 기록 항목과 분석

필수 receipt: 익명 participant/slot, period, treatment, A/B/task version,
runtime/source/AI version, start/end, 제출별 hash/time/status/failed IDs,
도움 내용·시각, AI 대기/사용량, infrastructure pause, baseline 제공 여부,
timeout/withdrawal, consent/retention policy reference.

주요 결과는 단계별 time-to-first-correct와 budget 내 완료 여부다. active human time,
AI interaction time, 결함 종류/수정 횟수, patch의 바뀐 물리적 코드 줄 수와 파일 수는
보조 지표다. 자동 생성 코드도 제출 코드에 포함하며 comments/blanks 포함 기준을
미리 고정한다. 연구자가 작성한 starter와 재사용 helper는 참여자가 새로 작성한 줄 수와
분리한다. 코드 줄 수를 생산성 또는 유지보수 비용과 같은 단위로 취급하지 않는다.

가능하면 평가자는 participant/treatment 라벨 없이 결과를 판단한다. source 형식에서
처리를 알 수 있어 완전 blinding이 불가능하면 그 한계를 기록한다. 4명 각각의 결과,
순서, 처리, 과제, 노출과 baseline 제공을 모두 표로 보고한다. 효과는 기술적 paired
차이로 제시하고 8개 기간이나 여러 제출을 독립 표본 수로 부풀리지 않는다.
실제 자료가 없으면 빈 템플릿을 유지하며 성공률·평균시간·개선율을 채우지 않는다.
