# pyjevsim-rl: portable local continuation experiment

Windows와 Linux에서 같은 소스로 **native snapshot + 모델 전용 복원 보조 코드(N)** 와
**재사용 가능한 continuation framework(C1)** 를 비교하는 작은 실험 저장소입니다.
큐(Q)와 제조 시스템(M)을 사용하며 **고정 행동의 시뮬레이션 실행**을 측정합니다.
강화학습 학습/수렴 실험이나 gorti federation 실험은 아닙니다. RTI 서버는 필요하지 않습니다.

## 시작

Python **3.11 이상**, Git, 첫 설치 시 PyPI 접속이 필요합니다. Ubuntu/Debian은 해당
Python 버전의 `venv` 패키지도 필요합니다(예: `python3-venv` 또는 `python3.12-venv`).
스크립트는 시스템 패키지나 다른 프로젝트의 가상환경을 변경하지 않습니다.

원격에 이 구현이 push된 후:

```sh
git clone --branch codex/portable-benchmark https://github.com/cbchoi/pyjevsim-rl.git
cd pyjevsim-rl
```

Linux:

```sh
python3 run_experiment.py --setup-only
python3 run_experiment.py --preset idle-primary --condition idle
```

Windows PowerShell (Python launcher 사용 시):

```powershell
py -3.14 run_experiment.py --setup-only
py -3.14 run_experiment.py --preset idle-primary --condition idle
```

`--setup-only`는 선택 사항입니다. 생략해도 첫 실행에서 저장소 전용 `.venv`를 만들고,
해시가 고정된 `dill==0.4.1`만 설치한 뒤 실험합니다. PyJevSim 원본은 `vendor/`에
보존되어 있어 별도 pip 설치하지 않습니다. OS별 가상환경은 서로 복사하지 마세요.
`python3`/`py -3.14`는 사용할 Python 실행 파일로 바꿀 수 있습니다.
`--help`는 설치나 실험 없이 도움말만 출력합니다.

## 기본 실험

| 항목 | 값 |
|---|---|
| 모델 / 방식 | Q, M / N, C1 |
| 반복 단위 | 모델별 새 paired family 6개 |
| 실행 수 | 24 timing arms, 12 exact 비교 셀 |
| 작업량 | prefix64, branches8, suffix8, delta0.25 |
| 실행 | 방식별 새 프로세스, 순차 실행, worker1 / BLAS1 |
| 기본 상한 | 실행·분석 합계 600초, 결과·transient 저장 16MiB, arm120초 |
| 결과 | 모델별 C1/N 시간비, 20,000 paired-family bootstrap 95%CI, 실행순서/전후반 민감도 |

여기서 N도 snapshot을 사용합니다. **처음부터 매번 재실행하는 serial 기준선은 아닙니다.**
C1/N은 snapshot의 존재 여부가 아니라 framework의 추가 비용을 포함한 비교입니다.
비율이 1보다 작으면 C1이 더 빠릅니다. 실험은 작은 표본의 탐색 평가이며 확증 결과가 아닙니다.

```sh
python3 run_experiment.py --condition idle --output "results/linux-idle-01"
```

기존 output 폴더는 덮어쓰거나 재개하지 않습니다. 실패한 arm을 재시도/대체하지 않고,
실패 및 미실행 분모를 남깁니다. 필요하면 **실행 전에** `--budget-seconds`(최대7200),
`--max-mib`(최대128)를 명시하세요. 상한은 디스크 quota나 하드 실시간 보장이 아닙니다.
시간은 소유 worker 종료 및 작업 경계, 저장량은 경계/주기 관측으로 제한하며 관측 사이
transient peak와 전체 lifetime memory는 미확인입니다.

`--seed-offset`은 새 코호트의 사전 결정 seed를 바꾸는 옵션입니다. 같은 offset을 사용한
idle/busy 실행은 같은 입력을 쓰지만 독립 표본을 추가한 것으로 합치면 안 됩니다.
좋은 결과가 나올 때까지 seed나 상한을 바꾸어 재실행하지 마세요.

## 결과 확인

종료 시 출력되는 폴더의 `findings.md`와 `analysis.json`을 확인하세요.

- `execution.json`: 실행/실패/미실행 분모, exact 비교 수, study admission.
- `analysis-status.json`: 분석 완료 여부. exit0과 execution admission,
  analysis status `completed`가 모두 있어야 전체 성공입니다.
- `arms.jsonl`, `cells.jsonl`: 간결한 측정값과 동등성 비교 기록.
- `environment.json`, `provenance/`: OS/Python/의존성, 실제 사용 소스 및 선택적 호스트 관측.

성공한 전체 snapshot/trace는 보관하지 않습니다. 독립 실행의 물리적 결과를 메모리에서
직접 비교한 후 digest를 남기므로 **보관 결과만으로 원래 trace를 복구할 수는 없습니다.**
결과 폴더와 `.venv`는 Git에서 제외합니다. 코드를 공개하지 않고 결과만 공유하려면
결과 폴더를 별도로 전달하되 환경 기록의 사용자 경로 등도 확인하세요.

## 비교를 해석할 때

- `--condition idle`은 사용자 표시일 뿐 무간섭 인증이 아닙니다. 자동으로 다른 프로그램을
  종료하거나 부하를 만들지 않습니다. 호스트 CPU/I/O 합계에는 벤치마크 자체도 포함됩니다.
- 기존 application wall 구간을 유지합니다. CPU time은 backend 진입부터 kernel 반환까지의
  별도 구간이며, 그와 일치하는 wall도 기록합니다. 서로 다른 구간의 wall-CPU 차이를
  외부 프로그램의 간섭량이라고 단정하지 않습니다.
- Windows/Linux, Python 버전, 저장 파일시스템, 전원 설정이 달라지면 교란 요인이 됩니다.
  같은 CPU/저장장치라도 OS만 바꾼 비교로 외부 간섭의 원인을 확정할 수 없습니다.
  가능하면 동일 OS/인터프리터/파일시스템에서 idle/busy를 짝지어 비교하세요.
- WSL에서의 기능 시험은 별도 Linux PC의 성능을 대표하지 않습니다. Linux 성능실험은
  가능하면 `/mnt/c` 같은 Windows 마운트 대신 Linux 파일시스템에 clone하세요.
- RL 정책 품질·학습 수렴·gorti federation·일반적인 병렬 우위는 평가하지 않습니다.

## 복원 의미와 새 모델 및 실행 비용 연구

기존 24회 실험은 그대로 두고, 별도 진입점으로 세 평가를 실행할 수 있습니다.

```sh
python3 run_research.py --stage all --condition unspecified
```

Windows에서는 `python3` 대신 `py -3.14` 또는 준비된 `.venv/Scripts/python.exe`를
사용합니다. 개별 평가는 `--stage semantics`, `--stage transfer`, `--stage cost`입니다.
실행마다 새 결과 폴더를 사용하며 과거 결과나 실패 실행을 덮어쓰지 않습니다.

- `semantics.json`: Q/M 18개 조건의 R/N/C1 복원·개입 비교, 162개 분기 궤적,
  난수·보상·시각·분기 격리 및 관측기의 오류 검출 검사.
- `transfer.json`: 새 재고·보충 모델 V1/V2의 54개 세 방법 비교 셀과 독립 도메인
  계산 비교. AI가 수행한 공학 사례연구이며 사람의 개발시간 절감 측정은 아닙니다.
- `cost/findings.md`: Q/M × prefix16/64/256 × branches1/4/16 × 6 family × R/N/C1,
  324회 시간 측정과 별도 6회 콜백 계수. R은 분기마다 앞부분을 다시 실행합니다.
- `workflow.json`: 전체 완료·실패·미실행 단계. `protocol.json`은 실행 소스와 환경 기록.

기본 전체 실행·분석 상한은 600초·새 결과와 일시 파일 16MiB입니다. 비용 분석용
30초를 남겨 두고 워커1·BLAS1로 순차 실행합니다. 상한으로 중단되면 부분 결과를
보존하며 재시도하지 않습니다. 간섭 없는 호스트나 전체 수명 메모리 상한 준수를
인증하지 않습니다. 저장 공간은 경계 관측이며 관측 사이 최대 사용량은 미확인입니다.
새 연구의 실행시간은 기존 Linux 자료와 합치지 않습니다.

요구사항·설계·추적성은 [연구 설계](docs/RESEARCH_DESIGN.md), 조건부 의미 보존
논증은 [의미 보존 논증](docs/SEMANTICS_ARGUMENT.md)에 있습니다. 확보 가능한
학부 졸업반 4명을 위한 [AI 보조 개발 파일럿 절차](docs/DEVELOPER_STUDY.md)도
준비했지만, 실제 참여자 자료는 아직 수집하지 않았습니다.

현재 수행 결과는 [연구 결과](docs/RESEARCH_RESULTS.md)에 정리했습니다. 의미 보존과
새 모델 사례는 완료됐고, 승인된 별도 비용 코호트도 330/330회 완료했습니다.
앞부분 재실행 생략은 확인했지만, 시험한 18개 조건에서 C1의 실행시간 우위는
관찰되지 않았습니다. 개발자 생산성 향상도 아직 측정하지 않았습니다.

후속 [손익분기 예측 실험 설계](docs/BREAK_EVEN_DESIGN.md)는 비용식으로 계수를
추정하고 새 조건에서 예측을 검증하는 계획입니다. [구현·시험 추적 계획](docs/BREAK_EVEN_IMPLEMENTATION_PLAN.md)과
[구조화된 프로토콜](docs/break-even-protocol.json)을 함께 제공합니다. 구현과 실행 상태는
[손익분기 진행 기록](docs/BREAK_EVEN_PROGRESS.md)을 참조하세요. 설계 당시의 JSON은
변경하지 않고, 명시적 실행 요청에서 별도의 실행 명세를 결과 폴더에 기록합니다.

### 손익분기 예측 연구 실행

```sh
python3 run_research.py --stage break-even --condition unspecified
```

준비된 환경을 바로 쓰려면 Linux는 `.venv/bin/python`, Windows는
`.venv/Scripts/python.exe`로 같은 스크립트를 실행합니다. 실험 시작 시 자동 시험이나
별도 freeze 검사는 하지 않습니다. `--condition idle`은 실제 전용 환경을 사용자가
관리한 경우의 표시이며 부하가 없다는 보증은 아닙니다.

이 단계는 기존 `--stage all`과 분리되어 있습니다. 새 inventory-risk-v1 모델의
K×S×B에서 R/N/C1을 비교하며, calibration 864회 → 사전 산식으로 정한 N의
validation 36N회 → burst 입력 transfer 216회를 순차 실행합니다. 각 셀의
정확성 관찰 실행(companion)과 시간 측정 실행(timing)은 별개입니다.

전체 상한은 9,000초(2.5시간), 단계별 3,600/3,600/1,800초, arm당120초입니다.
새 세 단계 전체 저장량32MiB 안에 transient8MiB를 포함합니다. `--budget-seconds`와
`--max-mib`로 상한을 줄일 수 있지만 자동 확대·N 축소·실패 실행 재시도는 하지 않습니다.
첫 완전 calibration family의 관측 비용으로 남은 계산의 실행 가능성을 판단합니다.
N이48을 넘거나 시간·저장량·모형 조건에 맞지 않으면 부분 자료를 보존하고 중단합니다.

`workflow.json`은 전체 상태, `calibration/fit.json`은 비용식과 근의 불확실성,
`predictions.json`은 새 좌표·N·고정 예측, 각 단계의 `findings.ko.md`는 요약입니다.
`arms.jsonl`의 시간 원자료와 `cells.jsonl`의 동등성 기록을 보존합니다. 큰 trace는
메모리에서 비교 후 버리고 snapshot은 소유한 arm 경로에서 정리합니다.
새 `workflow_wall_seconds`에는 제품 검증·종료·snapshot 폐기가 포함되지만 연구용
projection·해시·receipt는 제외됩니다. 기존 application 시간을 이 값으로 바꾸거나
Windows/Linux 코호트를 합치지 않습니다. 측정 구간이 다른 결과의 직접 비교도 피하세요.

2026년10월5일 첫 실행은144회/24개 셀을 정상 완료한 뒤, 남은 예상시간이 단계 예산을
넘는다는 사전 규칙으로 조기 중단됐습니다. 큰 계산량에서 C1의 R 대비 이익은 관측했지만
독립 family1개뿐이므로 확증 결과가 아닙니다. [부분 결과와 한계](docs/BREAK_EVEN_RESULTS.md)에
전체24조건과 중단 이유를 보존했습니다. Validation/transfer는 아직 수행하지 않았습니다.

## 구조와 출처

```text
run_experiment.py           # 준비 → 실행 → 분석의 단일 진입점
configs/idle-primary.json   # 사전 설정
bench/                     # 플랫폼 독립 실행·CPU/호스트 관측·분석
bench/continuation_study/   # 기존 측정 kernel, byte-preserved
src/pyjevsim_bridge/        # 기존 framework 소스
src/rti1516e/               # import에 필요한 SDK 소스; 서버 실행 안 함
vendor/pyjevsim/            # 실제 실험에 사용한 native 소스
provenance/source-import.json
docs/DESIGN.md              # SRS/STD/SDD/IDD 및 추적성
docs/IMPLEMENTATION.md      # Plan–Do–Review–Reflect 및 검증 범위
tests/                     # 개발 검증용; 실험 시작 시 자동 실행하지 않음
```

원래 gorti 저장소의 이력/실험 데이터는 포함하지 않은 scoped working-tree export입니다.
`provenance/source-import.json`에 원본 HEAD와 파일별 SHA256을 기록했습니다.
imported source를 수정하거나 줄바꿈을 바꾸면 native qualification/출처가 달라집니다.
MIT 라이선스는 `LICENSE`, native 라이선스는 `licenses/pyjevsim-LICENSE`에 보존했습니다.

검증된 플랫폼과 남은 한계는 [구현 기록](docs/IMPLEMENTATION.md)을 참고하세요.
