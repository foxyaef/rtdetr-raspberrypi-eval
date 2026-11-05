# RT-DETR Raspberry Pi 공통 평가기

**`test.py` 하나만 복사하면 폴더·전용 가상환경·공통 COCO 이미지를 자동 준비합니다.** 한 SSH 계정의 `queue/inbox/`에 여러 사용자의 모델을 넣고 실행하면, 한 번에 하나씩 평가하고 모델별 결과 ZIP을 저장합니다. 기본 설정에서는 모델명에 맞는 GitHub 사용자 브랜치에도 모델·로그·ZIP을 올립니다.

## 처음 한 번 준비

[최신 test.py 다운로드](https://raw.githubusercontent.com/foxyaef/rtdetr-raspberrypi-eval/main/code/test.py)를 작업 폴더의 `test.py`로 저장하고 실행합니다. 기존 사용자는 이 파일만 교체하면 됩니다. 저장소 clone이나 다른 Python 파일 복사는 필요하지 않습니다.

```bash
python3 test.py --setup-only
```

필요한 Python 라이브러리와 COCO 데이터를 준비하고 아래 폴더를 만듭니다. **64-bit Raspberry Pi OS, 64-bit Python 3.11~3.13**이 필요합니다. OS 도구가 없다면 관리자가 한 번 설치합니다.

```bash
sudo apt update
sudo apt install -y python3-venv python3-dev build-essential git git-lfs gh
```

GitHub 업로드를 사용할 계정에서 처음 한 번 로그인합니다. 로컬 저장만 하면 이 과정과 Git/Git LFS가 필요하지 않습니다.

```bash
gh auth login
gh auth setup-git
```

HTTPS 인증을 선택합니다. 실행 계정에는 `foxyaef/rtdetr-raspberrypi-eval` 쓰기 권한이 필요합니다. `user1`~`user4`는 모델 소유자를 구분하는 실험 라벨이며 GitHub 계정명이나 SSH 로그인명이 아닙니다.

## 대기열에 모델 넣고 실행

```text
작업폴더/
├── test.py                       복사할 파일 하나
├── queue/
│   ├── inbox/                    여기에 평가할 모델을 넣음
│   │   ├── user1_v1.onnx
│   │   ├── user2_v1.onix
│   │   └── user3_encoder_v2.onnx
│   ├── state.json                작업 상태·완료 이력 (자동 생성)
│   └── summary.csv               모델별 상태·결과 경로 (자동 생성)
├── models/                       기존 단일 모델 실행용
├── data/coco/                    모두가 공유하는 이미지 한 벌
├── results/
│   ├── user1_v1_<실행ID>/         측정값·로그
│   ├── user1_v1_<실행ID>.zip      결과 ZIP
│   ├── user2_v1_<실행ID>/
│   └── user2_v1_<실행ID>.zip
├── .rtdetr_venv/                 공통 전용 Python 환경
└── .rtdetr_runtime/              자동 복원되는 평가 코드
```

모델을 `queue/inbox/`에 넣은 뒤 다음 중 하나로 실행합니다. 파일 상단의 `MODEL_NAME`은 대기열 모드에서 사용하지 않습니다.

```bash
# inbox 모델 자동 인식 → 이름순으로 평가 → ZIP 저장 + GitHub 업로드
python3 test.py

# 동일한 평가를 실행하고 로컬 결과·ZIP만 저장
python3 test.py --no-upload
```

`test.py`가 `code/` 안에 있다면 모든 명령에서 `python3 code/test.py`를 사용합니다. 폴더는 `code/`와 같은 층위에 생성됩니다. 터미널의 현재 위치와 관계없이 스크립트 위치를 기준으로 경로를 정합니다.

**동작 순서:** 대기열 등록 → 공통 이미지 한 번 검증/준비 → 모델 검사 → CPU 추론 → COCO mAP 계산 → 결과 ZIP 저장 → 선택적 업로드 → 다음 모델. 추론과 정확도 계산은 별도 프로세스여서 이전 모델의 추론 메모리를 해제한 뒤 다음 모델을 실행합니다. 같은 작업 폴더에서는 평가 잠금으로 대기열/단일 평가의 중복 실행을 막습니다.

처음에는 3장으로 동작을 확인하고, 정상 동작하면 1,000장을 평가하세요.

```bash
python3 test.py --queue --limit 3 --no-upload
python3 test.py --queue
```

3장 실행과 1,000장 실행은 별도 작업입니다. 모델 파일 SHA256·모델명·이미지 수·반복/스레드 수·데이터/결과 경로·프로토콜·평가 코드 지문을 조합해 작업을 식별합니다. **동일한 모델과 설정으로 완료한 작업은 재실행해도 건너뜁니다.** 코드나 설정이 바뀌면 다시 평가합니다. 모델을 복사하거나 자동 삭제하지 않으며, 처리한 모델은 `inbox/`에 그대로 남습니다.

파일명은 `user1_`~`user4_`로 시작해야 합니다. `.onix`도 받지만 표준 확장자는 **`.onnx`**입니다. 모델 내용은 RT-DETR deploy 입력 `images`, `orig_target_sizes`와 출력 `labels`, `boxes`, `scores`를 가진 **self-contained ONNX**여야 합니다. 외부 가중치 파일을 요구하는 모델은 지원하지 않습니다.

## 새 모델을 계속 받는 감시 모드

```bash
python3 test.py --watch
# 로컬 저장만 하려면
python3 test.py --watch --no-upload
```

15초 간격으로 `queue/inbox/`를 확인하고 새 모델을 순서대로 처리합니다. 실행 중 추가한 모델도 다음 탐색 때 등록합니다. 일반 `python3 test.py`는 시작할 때 발견한 모델들을 처리한 뒤 종료합니다. `--watch`는 계속 기다리며, `Ctrl+C`로 종료합니다. SSH 접속을 닫아도 계속 실행하려면 `tmux` 세션 안에서 실행하세요.

**업로드 중인 파일을 잘못 읽지 않도록** 전송할 때는 `.part`로 끝나는 이름을 쓰고, 전송이 끝난 뒤 같은 폴더에서 최종 이름으로 바꾸세요. `.part` 파일은 평가하지 않습니다.

```text
전송 중: queue/inbox/user2_v2.onnx.part
전송 완료 후 이름 변경: queue/inbox/user2_v2.onnx
```

완료한 파일을 덮어쓰기보다 `user2_v2.onnx`, `user2_v3.onnx`처럼 버전을 올리는 것을 권장합니다. 모델 등록 직후와 평가 후 파일 지문을 확인해 실행 중 변경도 검사합니다.

## 상태 확인·실패 재시도·순서 지정

```bash
# 평가나 설치를 실행하지 않고 저장된 대기열 상태 표시
python3 test.py --queue-status

# 실패한 모델도 다시 시도; 성공한 모델은 유지
python3 test.py --queue --retry-failed

# inbox 안의 모델 중 지정한 것만 이 순서로 평가
python3 test.py --queue --models user3_v1.onnx user1_v1.onnx
```

`completed`, `failed`, `interrupted`, `pending`, `running`을 `queue/state.json`에 기록하고 `summary.csv`에 결과 폴더·ZIP 경로를 정리합니다. 한 모델이 실패해도 오류 로그/가능한 결과 ZIP을 남긴 뒤 다음 모델로 넘어갑니다. 파일명/경로가 잘못된 항목은 `rejected`에 기록합니다. 공통 데이터 준비 자체가 실패하면 모델 평가는 시작하지 않고, 원인을 해결한 다음 다시 실행합니다.

중단된 모델은 다음 대기열 실행에서 **그 모델의 평가를 처음부터** 다시 시작합니다. 완료한 모델은 유지합니다. 추론 중간 이미지부터 이어 돌리는 방식은 아닙니다. OS가 평가기 전체를 강제 종료하거나 저장공간이 고갈되면 해당 실행의 ZIP이 완성되지 않을 수 있으며, 저장된 작업 상태로 다시 시도할 수 있습니다.

GitHub 업로드가 실패해도 평가 성공은 `completed`, 업로드는 `failed`로 따로 기록합니다. 이 경우 ZIP은 이미 로컬에 있습니다. **측정 없이 업로드만 재시도**합니다. 원본 모델은 입력 폴더에 남아 있어야 합니다.

```bash
python3 test.py --upload-only results/<실행폴더이름>
```

업로드 여부를 바꿨다는 이유로 완료한 대기열 작업을 다시 평가하지 않습니다. 이전에 `--no-upload`로 완료한 결과를 나중에 올릴 때도 `--upload-only`를 사용합니다. 대기열 CSV는 평가 당시의 업로드 상태를 담으며, 재업로드의 최신 상태는 해당 결과 폴더의 `upload_status.json`에서 확인합니다.

## 결과 저장과 GitHub 경로

항상 로컬에 `results/<모델명>_<실행ID>/`와 같은 이름의 `.zip`을 먼저 만듭니다. ZIP은 **측정 결과·예측·로그**이며 모델 파일 자체는 포함하지 않습니다. 기본 업로드 경로는 다음과 같습니다.

```text
user2_v1.onnx → user2 브랜치
├── models/user2_v1.onnx                       Git LFS로 저장한 모델
└── results/user2/user2_v1_<실행ID>/
    ├── results.zip                           로컬 ZIP과 동일한 결과 묶음
    ├── metrics.json                          mAP·파일 크기·자원 측정
    ├── provenance.json                       모델/데이터/환경 지문·이미지 ID
    ├── quantization_audit.json                dtype·Q/DQ 검사
    ├── predictions.json
    ├── class_metrics.csv
    ├── latency.csv
    ├── resource_samples.csv
    ├── system_info.json
    ├── evaluation_state.json
    └── audit.log / infer.log / score.log
```

`user1_`→`user1`, `user2_`→`user2`, `user3_`→`user3`, `user4_`→`user4`로 분기합니다. `main`에는 결과를 올리지 않습니다. 업로드는 임시 clone에서 수행하며 현재 작업 폴더의 Git 브랜치를 바꾸지 않습니다. 업로드 실패·평가 실패 시 로컬 자료는 보존합니다. 모델은 Git LFS의 계정별 저장/전송 한도를 사용합니다.

`upload_status.json`은 업로드 후 만들어지는 로컬 영수증으로 commit·결과 링크를 담습니다. 재시도의 ZIP 내용과 원격 결과를 동일하게 유지하기 위해 **이 영수증은 ZIP/원격 묶음에서 제외**합니다. 평가 실패 자료는 자동 업로드하지 않습니다.

## 공통 데이터와 측정 조건

- 공통 Colab과 같은 seed 42의 COCO 선택: calibration 512장, 겹치지 않는 evaluation 1,000장. 평가기는 calibration으로 모델을 재양자화하지 않습니다.
- 공식 ZIP 약 1.1GB를 내려받고 선택 JPEG **1,512장만** 풉니다. 공식 ZIP SHA256·JPEG CRC·annotation/manifest 지문을 검사합니다. 데이터와 환경은 작업 폴더당 한 벌을 모든 모델이 재사용합니다.
- CPUExecutionProvider, ONNX Runtime 1.22.1, batch 1, 640×640 RGB bilinear/255, 기본 CPU 스레드 4개, 별도 warmup 최대 10장 뒤 모든 평가 이미지 측정.
- 추론 반복은 `--repeats 3`처럼 지정합니다. AP 예측·전처리·출력 변환·E2E는 이미지당 첫 반복을 쓰고, 추론 시간은 모든 반복을 기록합니다. 반복해도 정확도 표본 수가 늘지는 않습니다.

| 측정값 | 정의 |
|---|---|
| mAP50:95 | pycocotools bbox, IoU 0.50~0.95; `metrics.json`은 0~1 |
| 파일 크기 | 실제 ONNX bytes·MiB; ZIP/LFS 포인터 크기 제외 |
| 전처리 지연 | JPEG 읽기·RGB·resize·배열 변환, mean/p50/p95 ms |
| 출력 변환 지연 | 모델 출력을 COCO category ID·xywh 예측으로 변환 |
| 추론 / E2E 지연 | session.run / 전처리+첫 추론+출력 변환 |
| CPU peak | 20ms 샘플; 코어 하나=100%와 전체 코어 정규화 값을 함께 기록 |
| 메모리 peak | 추론 프로세스 RSS 샘플 peak와 Linux high-watermark 중 큰 값 |
| 전력·에너지 | 외부 전력계가 없으므로 `null`(미측정) |

CPU·RSS 범위는 모델 로딩·warmup·추론·예측 파일 쓰기입니다. 데이터 준비·모델 raw 검사·COCO 채점·압축·업로드는 제외합니다. 지연값에서는 예측 저장/로그 쓰기를 제외합니다. 온도·가능한 throttling 상태도 보조 기록합니다. 같은 Pi의 냉각·전원·동시 작업 조건을 통일하세요. 2GB 기기는 3장 확인부터 진행하세요. **현재 검증은 PC CPU 기능 검사이며 Pi 실기기 성능을 보장하지 않습니다.** 과거 Colab GPU의 속도/RSS와 Pi 값을 직접 합쳐 비교하지 마세요.

기존 데이터 폴더를 재사용할 수도 있습니다.

```bash
python3 test.py --queue --data-dir /path/to/data/coco --no-download
```

## 기존 단일 모델 실행 / 개발

`models/`에 모델을 넣는 기존 방식도 유지합니다. `--model`을 지정하면 대기열보다 우선하며 그 모델 하나만 평가합니다.

```bash
python3 test.py --model user1_v1.onnx
```

개발용 원본은 `code/benchmark.py`, `code/model_queue.py`, 데이터/평가/업로드 보조 모듈입니다. 이를 수정했다면 `python tools/build_standalone.py`로 배포용 `code/test.py`를 다시 생성합니다. 일반 사용자는 이 빌드가 필요하지 않습니다.

```bash
python -m unittest discover -s tests -v
```

검증 범위는 [VALIDATION.md](VALIDATION.md)에 정리했습니다. 참고: [공통 Colab](https://github.com/foxyaef/haiteam4_finetune/blob/main/colab/COCO_Eval.ipynb).
