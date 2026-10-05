# 손익분기 연구 구현과 실행 기록

## MS39 L1 Plan

2026년 10월 5일 사용자가 실험 설계대로 진행하도록 요청했다. TASK208 모델과 bindings,
TASK209 측정과 실행 연결, TASK210 분석을 병렬 구현한다. TASK211에서 직접 관련
시험·독립 검토를 수행한 뒤 실제 연구는 순차 실행한다. 원본 core/vendor/v1 kernel과
과거 결과는 변경하지 않는다. TASK207의 설계는 commit633e331에 보존되어 있다.

새 실행 요청은 제안된 전체9,000초, 단계3,600/3,600/1,800초, 단일 arm120초,
새 연구32MiB와 그 안 transient8MiB 범위로 적용한다. N 상한48·불리한 결과 보존·
재시도 금지를 유지한다. Linux 전용 PC와 현재 Windows 중 실행 호스트를 확인하는
질문을 보냈으며 우선 구현·직접 관련 검증을 진행했다. 별도 Linux 접속 정보가 없어
현재 접근 가능한 Windows 호스트를 기본 실행 대상으로 준비한다고 사용자에게 알렸다.
실행 시 condition은 unspecified이며 전용 Linux의 통제된 결과로 해석하지 않는다.
설치나 smoke/freeze 검사를 시작 조건으로 추가하지 않는다.

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
cohort 분모와 성능 결과는 없다. 나머지 구현과 검증 결과는 아래 기록으로 갱신한다.

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

### TASK209 완료 기록

Plan은 기존 측정 의미를 보존하며 새 kind만 연결하고 별도의 연속 wall/CPU endpoint를
구현하는 것이었다. Do에서 순차 계획, 측정, worker 연결, 세 단계 collection과 compact
보관을 구현했다. Review에서 planner5 + fake endpoint13 + 실제6-arm 대조1 + 격리
worker1의20개 직접시험이 통과했다. K1/S512/B16의 R/N/C1 companion full-state와
6개 arm 정상 scalar 출력이 일치했고, 실제 worker에서 source identity·4MiB 전송·
소유한 scratch 정리도 확인했다. 이때 얻은 시간은 연구 결과로 보관·사용하지 않았다.

추가로 collection/entry/owned-process 관련16개 시험이 통과했다. effective plan hash,
실패 분모, provenance 기록 실패 시 attempted 보존, 부분 분석, 기존 cost 진입점과
direct-child timeout 경로를 확인했다. 관련70개 시험은 task별 실행 합계이며 전체 시험
모음을 실행했다는 뜻이 아니다. 실제 연구 TC13b는 아직 수행하지 않았다.

Reflect: 첫 family 이후의 feasibility는 관측 process wall과1.5 안전계수를 사용한다.
후속 단계는 calibration의 역할·방법별 최대 process wall을 사용한 보수적 추정이다.
예측된 완료 가능성은 보증이 아니며 Windows Job/process-group 전체 소유권, lifetime
memory, 외부 간섭 없음은 인증하지 않는다. old source/kernel/result는 그대로 유지했다.

초기 위험은 Python risk kernel의 계획된 총 계산량과 stage 상한이다. 첫 완전 calibration
family로 예측한 나머지 비용이 상한을 초과하면 설계대로 중단·보고한다. 유리한
손익분기점을 얻기 위한 K 변경이나 N 축소·예산 자동 연장은 하지 않는다.
동일 목적 보완 task 추가 수0. 10회 이상 추가하면 루프 중단 규칙을 적용한다.

### TASK211 완료 기록

Plan: 새 모델의 직접 의미 시험과 endpoint·분모·통계 시험만 검토하고, 연구 중 병렬
개발 부하를 만들지 않는다. Do: 세 담당자가 병렬 구현하고 root가 collection을 연결했다.
Review: 관련70개 시험 통과, 실제 isolated worker 실행 경로 확인, 원본 src/vendor/
continuation_study와 기존 cost·분석 파일의633e331 대비 변경 없음 확인. 전체 시험·
smoke/freeze 단계는 수행하지 않았다. README 실행 명령과 IDD 실제 함수 대응도 갱신했다.
Reflect: 전용 Linux 결과·호스트 무간섭·학습 성능·인간 생산성 증거는 확보한 것이 아니다.

## MS40 L1 Plan — TASK212

현재 접근 가능한 Windows에서 `condition=unspecified`인 독립 새 코호트
`results/break-even-20261005-01`을 한 번 실행한다. 다른 프로그램은 종료하지 않는다.
명령은 다음과 같다. 이 기록 시점에는 실행 직전이며 완료를 뜻하지 않는다.

```powershell
.\.venv\Scripts\python.exe -I -B run_research.py --stage break-even --condition unspecified --output results/break-even-20261005-01 --budget-seconds 9000 --max-mib 32
```

calibration864/864와144/144 셀이 수용되어야 사전 산식의 N과6개 예측 좌표를 정한다.
처음 완전 family의 feasibility가 실패하거나 arm 오류·시간·저장 제한에 도달하면
부분 기술 분석을 남기고 중단한다. N>48 또는 다음 단계 부적합이면 validation/transfer를
임의로 줄이거나 시작하지 않는다. 수집 중 추가 모델·시험·분석을 병렬 실행하지 않는다.
