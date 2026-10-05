# Continuation 개선 구현과 검토 기록

## MS42 L1 Plan

2026년 10월 5일 사용자 요청으로 TASK215부터 시작한다. 세 에이전트가 실행 경로,
상태 선언 지원, 의사결정 사례를 병렬 설계했고 root가 요구사항과 IDD를 고정했다.
초기 git 상태는 깨끗하며 branch는 codex/portable-benchmark다. 먼저 기존 strict
실행의 진단을 기록하고 이후 코드 변경을 진행한다. 기존 결과는 변경하거나 합치지 않는다.

## MS42 L1 Do

CONTINUATION_IMPROVEMENT_DESIGN.md에 요구사항, 검증 범위, 별도 admitted 계약,
실험 분모, 제한과 task를 작성했다. 현재 성능 개선·개발자 생산성·의사결정 이득은 결과가 아니다.

## MS42 L1 Review

설계 검토에서 기존 negative control이 observer 민감도인지 framework rejection인지 구분했다.
반복 admission을 생략하는 프로파일은 다른 검사 계약이며 동일 보장 최적화가 아니라는 점을
명시했다. 기존 risk 모델의 단조 주문 목적은 유의미한 의사결정 사례로 사용하지 않는다.

## MS42 L1 Reflect

baseline 진단과 직접 구현은 진행 가능하다. 본 손익분기 연구 호스트와 단계 예산,
사람 참여 세션과 실제 RL 과제는 미확정이다. 동일 목적 보완 task 추가 수0.

### TASK215 진단 결과

코드 변경 전 `tools/profile_continuation.py`를 기존 K1/S8/B4의128step에 실행했다.
`results/improvement-profile-20261005-baseline/profile.json`에 실제 source 해시와 호출별
시간을 보존했다. 계측 workflow3.106458초, admission135회 누적2.387557초,
source verification161회 누적1.663174초, 파일 open11,662회였다. 중첩 누적시간은
서로 더하지 않는다. 이는 한 조건의 계측 진단이며 정상 실행시간이나 speedup 근거가 아니다.
자원 제한60초/4MiB 안에서 성공했고 source runtime/snapshot 정리는 확인됐다.

사용자는 본 연구 호스트로 전용 Linux를 선택했다. 접속 정보와 단계 예산 재배분 답변은
아직 없으므로 Windows에서 본 손익분기 연구를 시작하지 않는다.

## MS43 L1 Plan

TASK216은 strict 기본값을 유지하고 명시적 trusted runtime 프로파일을 추가한다.
TASK217의 값 필드 도우미와 TASK218의 독립 미래 평가 사례는 별도 파일로 병렬 구현한다.

## MS43 L1 Do

TASK216: coordinator/registry에 실행 프로파일과 runtime 소유 witness를 구현했다.
정상 step에서는 admitted 계약에 한해 반복 admission 대신 registry 세대·소유권을
확인한다. full fresh/restore/capture/inspect/semantic_view는 유지한다.

## MS43 L1 Review

TASK216 신규 직접시험15개와 기존 관련 domain12개가 통과했다. 기본 strict 동작,
외부 handle/세대/종료/실패 witness, 명시적 source 변경 검출 시점, 동일 정상 결과,
손상 snapshot 거부 및 분기 격리를 확인했다. source 변경 시험은 실제 파일 변조가
아닌 verifier 실패 주입이다. 새 시험에서 stochastic RNG 일반화까지 확인한 것은 아니다.

## MS43 L1 Reflect

TASK216은27개 고유 직접시험을 통과했다. admitted는 같은 검증 보장이라는 주장을
하지 않는다. 성능 효과는 아직 미측정이며 독립 비계측 timing이 필요하다.
동일 목적 보완 task 추가 수0. TASK217/218은 구현 중이다.
