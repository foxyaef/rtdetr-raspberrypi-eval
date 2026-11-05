# RT-DETR Raspberry Pi 평가기

`models/`에 ONNX 모델을 넣고 **`code/test.py`를 실행**하면 공통 COCO 이미지로 평가하고, 모델과 결과 폴더를 GitHub의 해당 사용자 브랜치에 올립니다.

```text
rtdetr-raspberrypi-eval/
├── code/
│   ├── test.py             실행·측정·결과 저장
│   ├── github_upload.py    사용자 브랜치 업로드
│   ├── coco_data.py        공통 이미지 준비·검증
│   └── coco_metrics.py     COCO mAP 계산
├── models/
│   └── user1_v1.onnx       사용자가 넣을 모델
├── data/coco/              첫 실행에 자동 준비, Git 업로드 제외
├── results/                실행별 결과·ZIP, main 업로드 제외
├── requirements.txt
└── protocol.json
```

`test.py`만 따로 복사하지 말고 저장소 전체를 사용하세요. FP32·양자화 모델 모두 같은 평가기를 사용합니다. 모델은 공식 RT-DETR deploy 형식의 입력 `images`, `orig_target_sizes`와 출력 `labels`, `boxes`, `scores`를 가진 단일 ONNX여야 합니다. 외부 가중치 파일을 요구하는 모델은 지원하지 않습니다.

## 1. 라즈베리파이 준비

**64-bit Raspberry Pi OS와 64-bit Python 3.11~3.13**, CPU용 ONNX Runtime **1.22.1**을 사용합니다. Pi의 모델·RAM에 따라 RT-DETR R50이 느리거나 메모리가 부족할 수 있으므로 2GB 기기는 반드시 아래 3장 smoke부터 확인하세요. 현재 제공한 검증은 PC CPU 기능 검사이며 실제 Raspberry Pi 하드웨어 성능을 측정한 결과는 아닙니다.

```bash
sudo apt update
sudo apt install -y git git-lfs gh python3-venv python3-dev build-essential
git clone https://github.com/foxyaef/rtdetr-raspberrypi-eval.git
cd rtdetr-raspberrypi-eval
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
git lfs install
```

전원 공급·냉각·CPU 설정을 같은 조건으로 맞추고 다른 무거운 작업은 종료한 상태에서 비교하세요. CPU 온도와 가능한 경우 throttling 상태도 기록합니다. 전력계는 현재 없으므로 전력·에너지는 **`null`(미측정)**입니다. CPU 사용률이나 기기 정격 전력으로 실제 소비전력을 대신 계산하지 않습니다.

## 2. GitHub 로그인 — 처음 한 번

```bash
gh auth login
gh auth setup-git
```

GitHub.com의 HTTPS 인증을 선택하고 안내에 따라 로그인합니다. 토큰을 Python 파일·모델명·명령 인자에 적지 마세요. 이미 Git HTTPS 인증이 설정돼 있으면 위 과정을 다시 할 필요가 없습니다.

`user1`~`user4`는 **실험용 라벨**이며 GitHub 계정명이 아닙니다. 실행하는 GitHub 계정은 이 저장소에 쓰기 권한이 있어야 합니다. 팀원의 별도 GitHub 계정을 사용할 경우 소유자가 저장소의 협업 권한을 설정해야 합니다. 파일명만으로 GitHub 권한이 생기지는 않습니다.

## 3. 모델 이름과 실행

모델을 `models/`에 넣습니다. 파일명은 `user1_v1.onnx`, `user2_encoder_both.onnx`처럼 **`user1_`~`user4_`로 시작**해야 합니다. 질문의 `.onix`도 입력 이름으로 받아들이지만 표준 확장자는 `.onnx`입니다. 모델 내용은 실제 ONNX여야 합니다.

가장 간단한 방법은 `code/test.py` 상단만 바꾸는 것입니다.

```python
MODEL_NAME = "user1_v1.onnx"
GITHUB_REPOSITORY = "foxyaef/rtdetr-raspberrypi-eval"
UPLOAD_TO_GITHUB = True
CPU_THREADS = 4
LATENCY_REPEATS = 1
```

먼저 동작만 확인합니다. 이 실행은 GitHub에 올리지 않습니다.

```bash
python code/test.py --model user1_v1.onnx --limit 3 --no-upload
```

정상 동작하면 전체 1,000장을 평가하고 자동 업로드합니다.

```bash
python code/test.py
# 또는 파일명을 명령에서 지정
python code/test.py --model user2_v1.onnx
```

시간 측정을 반복하려면 `--repeats 3`을 추가합니다. 추론 시간은 모든 반복에서 기록하고, AP·전처리·출력 변환·E2E는 이미지당 첫 반복을 사용합니다. 같은 이미지를 반복해도 정확도 평가 표본 수가 늘지는 않습니다.

이미지만 미리 준비하거나 업로드 없이 측정할 수도 있습니다.

```bash
python code/test.py --prepare-only
python code/test.py --model user1_v1.onnx --no-upload
# 기존 공통 데이터 폴더를 읽기 전용으로 검증·사용
python code/test.py --data-dir /path/to/data/coco --no-download --no-upload
```

첫 데이터 준비에는 공식 COCO 이미지·annotation ZIP 약 1.1GB를 내려받습니다. 선택된 JPEG 1,512장과 ZIP을 보관하므로 여유 공간을 준비하세요. 다운로드는 재개 가능하며 공식 ZIP SHA256·JPEG CRC를 검증합니다. 데이터는 GitHub에 업로드하지 않습니다.

## 측정값의 의미

| 항목 | 기록 방법 |
|---|---|
| **mAP50:95** | pycocotools bbox, IoU 0.50~0.95, maxDets 1/10/100. `metrics.json` 값은 0~1 |
| **파일 크기** | 실제 self-contained ONNX 파일 bytes·MiB. ZIP이나 LFS 포인터 크기 제외 |
| **전처리 지연** | JPEG 읽기·RGB 변환·640×640 bilinear resize·배열 변환 |
| **출력 변환 지연** | labels/boxes/scores를 COCO category ID·xywh 예측으로 변환 |
| **추론 / E2E 지연** | `session.run` / 전처리+첫 추론+출력 변환. mean·p50·p95, 단위 ms |
| **CPU peak** | 20ms 간격 프로세스 CPU 샘플. 코어 하나=100%(4코어 최대 약400%)와 전체 코어로 정규화한 값 둘 다 기록 |
| **메모리 peak** | 평가 추론 프로세스의 RSS. 20ms 샘플 peak와 Linux OS high-watermark 중 큰 값, 원값도 각각 기록 |
| **전력·에너지** | 외부 측정 장치가 없어 `null`, 이유도 기록 |

RSS·CPU는 모델 로딩·별도 warmup·추론·예측 파일 쓰기를 포함합니다. 데이터 준비, raw ONNX 검사, COCO 정확도 계산, GitHub 업로드는 **다른 단계/프로세스**여서 모델 측정값에 포함되지 않습니다. 지연 통계에서는 예측 JSON 저장·로그 쓰기를 제외합니다. 시스템 전체 CPU·메모리, 단계별 프로세스 peak, 온도도 보조 자료로 남깁니다. 짧은 순간의 CPU/RSS 샘플 peak는 샘플링 간격 영향을 받습니다.

공통 Colab과 이미지 선택·전처리·decode·COCO AP 계산 규칙을 맞췄습니다. seed 42, calibration 512장과 겹치지 않는 평가 1,000장, batch 1, 별도 warmup 10장 뒤 모든 평가 이미지를 측정합니다. 현재 GitHub main의 옛 노트북은 처음 10장을 timing에서 제외했지만 이 평가기는 최근 수정본과 같이 **별도 warmup 후 1,000장 모두** 측정합니다. 원본 파일 지문은 `reference_sources.json`에 있습니다.

메모리를 줄이기 위해 예측을 파일에 순차 저장하고, 모델 추론이 끝난 뒤 별도 프로세스에서 COCO AP를 계산합니다. 이 저장 방식과 CPU 플랫폼이 다르므로 과거 Colab GPU의 RSS·지연과 절대값을 직접 합쳐 점수화하지 마세요. 이번 장치에서 FP32·기본 PTQ·최적화 모델을 같은 설정으로 비교합니다.

## 결과와 자동 업로드

```text
user1_v1.onnx  → user1 브랜치
user2_v1.onnx  → user2 브랜치
user3_v1.onnx  → user3 브랜치
user4_v1.onnx  → user4 브랜치
```

성공한 평가 후 선택한 모델과 해당 실행의 로그 폴더만 올립니다. 모델은 Git LFS로 저장합니다. GitHub 일반 Git은 100MiB 초과 파일을 거부하므로 약136MB ONNX도 LFS가 필요합니다. LFS의 저장·전송 한도는 계정 정책을 따르며 이 프로그램은 유료 용량을 구매하지 않습니다.

```text
user1 브랜치/
├── models/user1_v1.onnx                 LFS로 저장한 실제 모델
└── results/user1/user1_v1_<실행ID>/
    ├── metrics.json                    정확도·자원 측정
    ├── provenance.json                 모델·데이터·평가기 지문, 이미지 ID
    ├── quantization_audit.json          dtype·Q/DQ 증거
    ├── predictions.json                COCO 예측
    ├── class_metrics.csv
    ├── latency.csv                     이미지별 단계 시간
    ├── resource_samples.csv            20ms 측정의 1초 간격 요약 로그
    ├── system_info.json                실행 환경·온도·전력 미측정 이유
    ├── run_config.json
    ├── evaluation_state.json
    ├── errors.json
    └── audit.log / infer.log / score.log
```

각 결과 경로는 실행 ID로 구분하고, 업로드는 임시 clone에서 수행해 현재 작업 브랜치를 바꾸지 않습니다. `main`이나 다른 사용자의 브랜치에는 결과를 올리지 않습니다. 재사용한 모델명이 과거 결과와 혼동되지 않도록 반환 URL은 commit에 고정됩니다. `models/`와 `results/`는 main에서 기본 무시되지만 업로더가 선택한 파일만 명시적으로 추가합니다.

인증 실패·네트워크·LFS 문제 등으로 업로드가 실패해도 로컬 결과와 ZIP은 보존됩니다. 측정을 다시 하지 않고 업로드만 재시도할 수 있습니다.

```bash
python code/test.py --upload-only results/<실행폴더이름>
```

업로드 직후 생성되는 로컬 `upload_status.json`에는 commit·브랜치·결과 링크가 있습니다. 자기 자신의 commit 정보를 포함하는 파일이므로 원격 최초 결과 묶음에는 넣지 않습니다. 로컬 ZIP은 이 receipt까지 포함해 마지막에 생성됩니다. 모델 평가가 실패하면 `evaluation_state.json`과 오류 로그를 남기고 자동 업로드는 건너뜁니다. 3장 smoke나 부분 평가는 `full_evaluation=false`로 구분됩니다.

## 개발 검증

```bash
python -m unittest discover -s tests -v
```

데이터 선택·무결성 검사, COCO 출력 변환, 실제 작은 ONNX의 추론·AP 계산, 격리된 Git/LFS 업로드와 재시도 경로를 검사합니다. 시험용 모델은 테스트 중 임시 생성하며 실험 성능 결과로 사용하지 않습니다.

참고: [공통 Colab](https://github.com/foxyaef/haiteam4_finetune/blob/main/colab/COCO_Eval.ipynb), [ONNX Runtime ARM64 wheel](https://pypi.org/project/onnxruntime/1.22.1/#files), [GitHub 대용량 파일 정책](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).
