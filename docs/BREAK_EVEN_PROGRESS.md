# 손익분기 연구 구현과 실행 기록

## MS39 L1 Plan

2026년 10월 5일 사용자가 실험 설계대로 진행하도록 요청했다. TASK208 모델과 bindings,
TASK209 측정과 실행 연결, TASK210 분석을 병렬 구현한다. TASK211에서 직접 관련
시험·독립 검토를 수행한 뒤 실제 연구는 순차 실행한다. 원본 core/vendor/v1 kernel과
과거 결과는 변경하지 않는다. TASK207의 설계는 commit633e331에 보존되어 있다.

새 실행 요청은 제안된 전체9,000초, 단계3,600/3,600/1,800초, 단일 arm120초,
새 연구32MiB와 그 안 transient8MiB 범위로 적용한다. N 상한48·불리한 결과 보존·
재시도 금지를 유지한다. Linux 전용 PC와 현재 Windows 중 실행 호스트를 확인하는
질문을 보냈으며 답변 전에는 구현·직접 관련 검증을 진행한다. 설치나 smoke/freeze
검사를 시작 조건으로 추가하지 않는다.

## MS39 L1 Do

세 담당 에이전트가 domain/native/C1 bindings, 계획·연속 endpoint,
수식·family 통계를 각각 구현한다. Root는 기존 worker의 allowlist와 새 연구 전용
순차 collection을 연결한다. source identity는 실제 파일 SHA256을 worker가 독립
확인하며, full companion projection은 메모리 비교 뒤 버리고 compact 측정값을 남긴다.

## MS39 L1 Review

TASK210 완료: `break_even_analysis.py`와 직접 합성 시험22개를 구현했다. 알고 있는
비용식 복구, 모든 부호·무근 처리, family 공동 bootstrap, 검증 좌표와 N 고정,
부분 코호트·source·입력 누수 거부를 확인했다. 기본20,000 draw의 합성 계산도
실행 가능했으나 이는 시뮬레이션 성능 결과가 아니다.

통합 검토에서 collection이 부분 중단되더라도 기술 분석을 보존하고, collection 수용과
analysis 수용을 모두 만족해야 다음 단계로 이동하도록 보완했다. 아직 실제 연구
cohort 분모와 성능 결과는 없다. 나머지 구현과 직접 검증은 진행 중이다.

## MS39 L1 Reflect

### TASK208 완료 기록

Plan은 동일 모델을 R/N/C1에서 사용하고 완전한 복원·개입 의무를 유지하는 것이었다.
Do는 versioned domain, native sidecar, C1 bundle, 독립 forecast/event oracle 구현이다.
Review에서 직접 의미 시험12개가 통과했다. 초기·수요 직후·입고 대기·동시사건의 cut,
양수 개입의 실제 fulfilled 차이, A/B/A 격리, dormant 마지막 레코드, 손상된 보상/table
거부, smooth/burst 입력과 callback/scenario 계수를 포함한다.
Reflect: 구성 검증이80개 입력을 재생성하는 비용은 제품 검증으로 측정 안에 남는다.
Native journal은 신뢰한 로컬 artifact이며 적대적 변조 전체를 인증하는 보안 연구가 아니다.
성능 cohort 실행은 이 task에 포함하지 않았다.

초기 위험은 Python risk kernel의 계획된 총 계산량과 stage 상한이다. 첫 완전 calibration
family로 예측한 나머지 비용이 상한을 초과하면 설계대로 중단·보고한다. 유리한
손익분기점을 얻기 위한 K 변경이나 N 축소·예산 자동 연장은 하지 않는다.
동일 목적 보완 task 추가 수0. 10회 이상 추가하면 루프 중단 규칙을 적용한다.
