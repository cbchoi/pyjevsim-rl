# Continuation 개선 구현과 Windows 탐색 결과

2026년 10월 5일 구현과 제한된 탐색 평가를 마쳤다. 기본 strict 경로를 보존한 채,
신뢰 모델을 전제로 반복 검증을 줄이는 C1A와 명시적 상태 필드 도우미, 독립 미래를
사용하는 의사결정 사례를 추가했다. **strict 대비 비용 절감 가능성은 관측했지만,
native 대비 일반적인 성능 우위나 논문의 독창성을 입증한 결과는 아니다.**

본 손익분기 연구는 사용자가 지정한 전용 Linux에서 별도로 진행한다. 아래 Windows
결과는 기존 손익분기 코호트나 향후 Linux 결과와 합치지 않는다. 시험·설치·다른 모델
연구를 timing과 병렬 실행하지 않았지만 외부 호스트 간섭과 전체 lifetime memory는
미확인이다. 실패·부분 결과를 보존했으며 재시도나 대체 실행은 하지 않았다.

## 구현과 검증 범위

| 항목 | 완료한 내용 | 주장하지 않는 내용 |
|---|---|---|
| strict C1 | 기존 기본값과 full admission 유지 | 기본 보장을 낮춘 최적화 |
| admitted C1A | 명시 opt-in, 소유권·세대·수명 검사, full fresh/restore/capture/inspection 유지 | 일반 step에서 임의 source·내부 상태 변조 즉시 탐지 |
| 상태 필드 도우미 | exact keys, detached JSON 값, 전체 검증 후 복원, 별도 inventory V1/V2 어댑터 | 숨은 상태·alias·RNG·calendar 완전성 자동 추론 |
| 오류 증거 | 미래 calendar 지연 관측, 의미 불일치 snapshot 거부, 원본·형제 격리 | 모든 DEVS의 동시 사건 순서 보장 |
| 의사결정 사례 | 실제 action 전이, 구매·보유·품절 비용, 선택 후 독립 미래 평가 | RL 학습·수렴 또는 실제 재고 시스템 검증 |
| 개발자 연구 | 신규 대응 과제 A/B와 판정표 명세 | 실행형 starter/oracle/evaluator 완성, 실제 생산성 개선 |

직접 관련 고유 시험 77개가 통과했다. 구성은 admitted와 기존 domain 27개,
상태 도우미·calendar 14개, 의사결정 11개, 탐색 계획·실제 child 15개,
기존 runner/entry 10개다. 이는 공학적 적합성 시험이지 독립 연구 표본 77개가 아니다.
전체 시험이나 추가 smoke/freeze gate는 수행하지 않았다.

C1A의 정상 step은 신뢰한 코드와 내부 상태가 handle 수명 중 임의 변경되지 않는다는
계약에 의존한다. 따라서 C1A/C1 비교는 **검증 계약의 차이를 공개한 비용 비교**이며,
동일한 변조 탐지 보장을 유지한 최적화라고 표현하지 않는다.

## 진단 실행

변경 전 strict 진단은 K1/S8/B4, 128 step에서 계측 workflow 3.106458초였다.
admission 135회 누적 2.387557초, source verification 161회 누적 1.663174초,
파일 open 11,662회를 기록했다. 중첩 누적시간을 더하지 않는다.

변경 후 admitted 진단은 60초의 연산 경계 예산 검사에서 실패했고 workflow는
76.168372초였다. 112 step까지 관측한 부분 자료이므로 baseline의 128 step과 분모도 다르다.
source verification 33회, 파일 open 2,318회로 호출은 적었으나
파일 작업 지연이 크게 관측됐다. 연산 도중의 hard timeout이 아니므로 60초를 넘을 수 있다.
두 진단은 profiler와 서로 다른 호스트 시점의 영향을 포함하므로 elapsed 비율로
speedup이나 성능 회귀를 계산하지 않는다. 실패 진단도 그대로 보존했다.

## 부분 성능 비교

R은 prefix를 다시 실행하는 native replay, N은 native snapshot, C1은 strict framework,
C1A는 admitted framework다. 조건은 K1/4096, S8, B4/16, prefix64, suffix16이었다.
계획은 3 family × 4조건 × 4방식 × companion/timing의 96회였다.

| 분모 | 계획 | 성공 | 실패 실행 | 미실행 |
|---|---:|---:|---:|---:|
| companion | 48 | 28 | 0 | 20 |
| timing | 48 | 27 | 0 | 21 |
| 전체 arm | 96 | 55 | 0 | 41 |

600초 예산에서 다음 실행 전 분석용 잔여 시간을 확보하는 규칙으로 종료했다.
총 597.516834초, `status=failed`, `study_admission=false`다. 55개 child는 정상
종료했고 정리가 확인됐다. 완성된 셀은 6/12개이며 모두 exact다. 다음 셀은 7/8 arm에서
끝났고 C1A timing이 없다. 4조건을 모두 포함한 완전 family는 1/3개뿐이다.
원자료 저장 누락·write error·정리 오류는 기록되지 않았다.

아래는 완성 셀만의 대응 기하평균 실행시간 비율이다. 1보다 작으면 분자의 시간이 짧다.
조건별 family 수가 다르고 표본이 작으므로 신뢰구간과 확증 판정을 제공하지 않는다.

| K | B | 해당 조건의 완료 family 수 | C1A/C1 | C1A/R | C1A/N |
|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 1 | 0.3334 | 22.8011 | 18.7650 |
| 1 | 16 | 2 | 0.3277 | 14.8897 | 15.4502 |
| 4096 | 4 | 2 | 0.4473 | 0.5194 | 1.5134 |
| 4096 | 16 | 1 | 0.1867 | 0.3181 | 0.0833 |

마지막 행의 native 대비 큰 비율 차이는 특히 일반화할 수 없다. 그 family에서 N은
wall 59.771189초에 process CPU 5.593750초, C1은 wall 26.676490초에 CPU
9.484375초였다. C1A는 wall 4.980018초에 CPU 4.859375초였다. 다른 family의 N은
3.5191초였지만 그 family의 C1A timing이 없으므로 서로 짝지으면 안 된다.
느린 관측치를 임의 제거하거나 다른 프로그램의 간섭을 원인으로 확정하지 않는다.

현재 근거는 반복 검증 비용을 줄이는 방향의 유용성과 무거운 prefix에서 R 대비 이득
가능성을 보여준다. native 우위, 정확한 손익분기점, 일반적인 병렬 우위는 입증하지 않는다.

## 의사결정 사례 결과

별도 seed 984000부터 3개 family에서 4방식 × 고정 후보/고정 예산의 24건을 실행했다.
후보 주문량 8개와 독립 평가 미래 8개를 사용했다. 계획 미래는 하나이며 평가 미래는
모든 선택을 확정한 뒤 생성했다. 평가는 미래별 전지적 선택이 아니라 평가 panel 평균에서
가장 좋은 고정 후보를 oracle 기준으로 사용한다.

24/24 의사결정과 3/3 평가 panel을 완료했고 `study_admission=true`, source 일관성 true,
실패·미실행·선택 없음은 0이었다. CLI 기록 시간은 14.451259초로 120초 예산 안이다.
고정 후보 8개를 모두 평가하면 3개 family 모두 네 방식의 후보 결과와 선택이 일치했다.
평가 oracle의 후보×미래 셀은 192/192개, 선택 행동의 native/oracle 비교는
40/40 trajectory에서 일치했다. 같은 행동을 선택한 방식은 평가 미래별 native 확인을 공유한다.

1초 soft decision 예산에서의 결과는 다음과 같다. 손실은 작을수록 좋다.

| family | R/N/C1A 각각의 유효 후보 수 | C1 유효 후보 수 | R/N/C1A 평가 손실 | C1 평가 손실 |
|---|---:|---:|---:|---:|
| 984000 | 8 | 4 | 12.5000 | 20.7500 |
| 984001 | 8 | 5 | 12.4375 | 23.5000 |
| 984002 | 8 | 5 | 24.0625 | 24.0625 |

따라서 C1A는 strict보다 3∼4개 더 많은 후보를 평가했고 2/3 family에서 평가 손실이
줄었다. 그러나 R과 N도 8개를 모두 평가하여 동일한 손실을 얻었으므로 native보다
좋은 결정을 했다는 결과는 아니다. 고정 후보 모드에서 C1A는 C1보다 0.6740∼0.9642초
짧았지만 N보다 0.5630∼0.7486초 길었다.

첫 두 family에서 좋은 후보 q=9는 사전에 섞은 후보 순서의 마지막에 있었고 C1은 거기까지
도달하지 못했다. 효용 차이는 속도와 이 후보 순서가 함께 만든 관측이다. 세 번째 family는
전 후보를 평가해도 계획 미래에서 q=4를 선택하여 평가 oracle보다 손실이 9.375 높았다.
빠른 실행이 미래 예측 오차까지 해결한다는 결과는 아니다.

C1의 예산 초과 후보는 선택에서 제외하고 실제 초과 시간은 남겼다. 이 예산은 hard
realtime 상한이 아니다. planning oracle·held-out oracle·선택 행동 native 확인 비용은
별도 기록했다. 3 family/4방식의 실행순서는 완전히 균형 잡히지 않았으며, 세 개의 작은
합성 사례에 대한 기술적 관측이다. 기존 inventory의 새 목적함수이지 신규 모델 일반화가 아니다.

## 남은 연구와 Linux 이관

1. 전용 Linux의 접속 또는 사용자 실행·결과 전달 방식을 정한다. 원격 push는 하지 않았다.
2. 기존 본 연구 9,000초/32MiB 내의 단계별 예산 재배분 답변을 반영한다. 답변 전 임의 변경하지 않는다.
3. 현재 `run_improvement.py`는 96회 탐색 비교와 24건 의사결정 실행용이다. 기존
   `run_research.py`의 본 손익분기 calibration/validation/transfer에 C1A를 연결한
   확증 설계·분석은 아직 완료하지 않았다. 새 계약을 평가할 때는 버전별 새 calibration이 필요하다.
4. 대응 개발자 과제의 starter·독립 oracle·오류 평가기를 구현하고 실제 4명 연구를 준비한다.
5. RL 효용은 별도로 과제·학습 알고리즘·비교 예산을 고정한 뒤 구현한다. 현재는 학습 결과가 없다.

현 단계의 학술적 기여 후보는 검증 계약을 명시한 continuation 비용 구조와 실험 가능한
복원·개입 의미다. 이번 결과만으로 SCIE급 완전한 contribution이 완성됐다고 판단하지 않는다.

## 원자료

모든 경로는 저장소 루트 기준이며 원자료는 local results에 보존했다. 공개하지 않았다.

- `results/improvement-profile-20261005-baseline/profile.json`
- `results/improvement-profile-20261005-admitted/profile.json`
- `results/improvement-performance-20261005-01/protocol.json`
- `results/improvement-performance-20261005-01/arms.jsonl`
- `results/improvement-performance-20261005-01/cells.jsonl`
- `results/improvement-performance-20261005-01/execution.json`
- `results/improvement-performance-20261005-01/analysis.json`
- `results/improvement-decision-20261005-01/protocol.json`
- `results/improvement-decision-20261005-01/decision.json`

성능 연구 source identity는
`97f4720f0e7c9cc26aaf959c14a1c229cb68db548df2f454a2264aff78883665`다.
결과의 해석과 구현 계약을 세 에이전트가 읽기 전용으로 검토했다. 후속 문서 수정은
측정 코드를 바꾸지 않으며, 과거 코호트의 실패나 admission을 소급 변경하지 않는다.
