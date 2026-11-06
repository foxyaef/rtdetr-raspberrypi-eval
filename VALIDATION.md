# 검증 기록

2026-11-05, Windows x86-64 CPU / Python 3.12.14에서 검증했습니다. 설치 버전은 requirements.txt와 같으며 ONNX Runtime 1.22.1입니다. 이 기록은 Raspberry Pi ARM64 실기기 성능 측정을 뜻하지 않습니다.

- 초기 평가기 34개 검사에 단일 파일 자동 준비 검사 16개를 추가했습니다. **전체 50개 중 48개 통과, 2개 Windows symlink 생성 권한 제한으로 건너뜀**입니다.
- 실제 공식 Encoder 전체 W8A8 ONNX로 공통 COCO 이미지 3장: 모델 검사 → CPU 추론 → COCO AP 계산 → 단계별 시간·CPU·RSS → 결과 ZIP 저장까지 성공. `full_evaluation=false`, 전력 값은 null. 3장 AP를 최종 정확도 결과로 사용하지 않았습니다.
- 별도의 작은 ONNX·정답 fixture로 실제 ONNX Runtime과 pycocotools 실행, 입력 전처리·category ID·좌표 변환, 반복 3회에서도 AP 예측은 이미지당 한 번만 사용하는 것을 검증했습니다. fixture는 실제 연구 결과가 아닙니다.
- 기존 공식 COCO ZIP으로 selection을 다시 생성해 calibration/evaluation annotation과 manifest가 기존 Colab과 바이트 단위로 같음을 확인했습니다. 공식 archive SHA256와 JPEG 1,512개의 CRC를 확인했습니다. 기존 데이터는 수정하지 않았습니다.
- Git/LFS는 임시 로컬 bare 저장소에서 실제 payload 전송, main 보존, 사용자 브랜치 이력 보존, 동일 업로드 재시도, 동시 push의 fetch/rebase 경로를 검증했습니다. 이 검사에서 실제 GitHub 사용자 브랜치에 시험용 모델을 게시하지 않았습니다.

공통 평가 annotation SHA256:

```text
274a6265f8d4fccbe7ef08be3118c10b428389795b71d3c22d3565a341e660fd
```

공통 manifest SHA256:

```text
98f5798b6b039770eba78a518d5d2d9f9b2630dc463dd4f5590863d4d50295ba
```

남은 실기기 확인은 Raspberry Pi의 ORT 실행·최대 메모리·발열/스로틀링과 실제 계정의 LFS 쓰기 권한입니다. 전력은 외부 전력계가 없어 측정하지 않습니다.

## 단일 계정 모델 대기열 업데이트

- 대기열/ZIP 업로드 검사 9개를 추가했습니다. 전체 **59개 중 57개 통과, Windows symlink 권한 제한 2개 건너뜀**입니다. 기존 공통 COCO 폴더는 읽기 전용으로 사용했습니다.
- 빈 작업 폴더에 배포용 `test.py` 하나만 복사하고, `queue/inbox/`에 작은 실제 ONNX 두 개와 잘못된 ONNX 한 개를 배치했습니다. 별도 `--queue` 없이 자동 인식하여 정상 → 실패 → 정상 순서로 처리하고 세 실행의 로그·ZIP을 저장했습니다. 정상 모델은 실제 ONNX Runtime CPU 추론과 pycocotools 채점을 거쳤습니다. 이는 제어 흐름 검사용 fixture이며 RT-DETR 모델의 실기기 성능 결과가 아닙니다.
- 같은 명령을 다시 실행해 완료/실패 작업을 자동 반복하지 않는 것을 확인했습니다. 실패 재시도, 파일 내용 변경 시 새 작업 등록, 지정 순서, `.part` 무시, 중단된 작업 재시작, 감시 모드에서 새 파일 인식도 확인했습니다.
- 별도 프로세스에서 평가 잠금을 획득하려 할 때 차단되며, 기존 작업 종료 후 잠금이 해제되는 것을 검사했습니다.
- 임시 로컬 Git/LFS 저장소에 모델·로그·`results.zip`을 실제 전송했습니다. 원격 ZIP이 로컬 ZIP과 바이트 단위로 같고, 영수증 생성 후 재업로드도 같은 commit으로 처리됨을 확인했습니다. ZIP과 로그가 다르면 업로드를 거부합니다. 실제 공개 저장소에는 시험 모델/결과를 올리지 않았습니다.
- 소비전력 측정과 Raspberry Pi ARM64 실기기 확인 범위는 앞선 기록과 같습니다.

## 단일 파일 자동 준비 업데이트

- 빈 작업 폴더에 생성된 `test.py` 하나만 복사해 별도 `.rtdetr_venv` 생성, 고정 Python 패키지 자동 설치, 실제 공통 데이터 읽기 검증까지 실행했습니다. 저장소나 보조 Python 파일을 해당 폴더에 미리 복사하지 않았습니다.
- 같은 복사본과 새로 설치된 전용 가상환경으로 실제 Encoder W8A8 모델 3장 평가를 끝까지 실행해 mAP 계산·자원 측정·ZIP 저장을 확인했습니다. `full_evaluation=false`, 전력 값은 null입니다. 실기기 Raspberry Pi 벤치마크나 전체 1,000장 결과라는 주장은 하지 않습니다.
- 단일 파일 검사 16개는 작업 경로 판별, 원본 사용자 파일 보존, runtime 손상·경로 탈출 차단, `sys.prefix` 검증, pip 설치 경로 설정 격리, 공백·Unicode 경로, 복사본의 실제 오프라인 데이터 준비를 포함합니다.
- Debian의 pip/ensurepip 누락 시 공식 PyPA fallback 설치 경로는 mock 검사로 확인했습니다. 실제 새 가상환경 설치 검사는 이 Windows 호스트의 ensurepip 경로를 사용했습니다.

## 기본 상시 대기·현재 상태 TXT 업데이트

- 추가 검사 6개를 포함해 전체 **65개 중 63개 통과, Windows symlink 권한 제한 2개 건너뜀**입니다. Python 3.12.14 / Windows x86-64 CPU에서 기존 공통 COCO 이미지와 임시 로컬 Git/LFS 저장소를 사용했습니다. Pi 실기기 검증은 포함하지 않습니다.
- 모델이 없는 기본 CLI도 계속 감시하도록 연결되고, `--once`는 현재 목록 처리 후 종료하도록 분리됨을 확인했습니다. 기존 복사본/실제 ONNX/COCO 통합 검사는 `--once`로 실행해 정상·오류·정상 순차 평가와 ZIP 저장을 다시 확인했습니다.
- 작업 중 다른 모델을 추가한 상황에서 별도 상태 갱신 스레드가 현재 실행 모델·대기 1건·진행 장수를 TXT에 표시하고, 현재 작업 종료 후 추가 모델을 실행하는 것을 확인했습니다. 이 스케줄링 검사는 제어용 callback을 사용하며 실기기 성능 측정이 아닙니다.
- 유휴 상태의 “바로 사용 가능”, `.part` 전송 건수, 잘못된 이름 구분, 완료/실패 모델의 대기 건수 제외, 정상 중단 후 “종료됨” 표시를 검사했습니다. 상태 파일은 원자적으로 교체하며, Windows에서 동시에 읽는 순간 발생하는 짧은 파일 잠금은 재시도합니다.
- 파일 크기·수정 시각 등이 같은 모델은 유휴 탐색 때 재해시하지 않고, 변경되면 다시 해시하는 것을 확인했습니다. 추론 중 TXT 갱신은 파일 메타데이터만 사용합니다.

## 2026-11-06 종료 신호·터미널 I/O 오류 및 파일명 제한 개선

- 전달받은 `user1_v1_20261105T154507_800853Z_5c9e701c.zip`은 `evaluation_state.status=interrupted`, `stage=infer`입니다. `infer.log`의 마지막 진행은 700/1000이며 `accuracy.json`·`metrics.json`이 없어 최종 평가 완료 자료가 아닙니다. `failure.log`에는 기존 종료 신호 handler에서 발생한 KeyboardInterrupt가 있습니다. 기존 코드가 SIGHUP과 SIGTERM을 같은 예외로 기록해 어느 신호인지는 확정할 수 없고, 별도 안내의 errno 5 발생 위치도 이 ZIP에는 없습니다.
- 새 상시 모드는 SIGHUP을 무시하고, 터미널에서만 발생한 EIO/EPIPE/EBADF/ENXIO는 화면 출력을 분리해 계속 실행합니다. `queue/service.log`에 진행 기록을 보존합니다. 실제 데이터/로그 파일의 I/O 오류는 무시하지 않습니다. Ctrl+C·SIGTERM 중단은 유지하며 신호 이름을 기록합니다. SSH 로그인 세션 정책·전원 차단은 별도 환경 요인입니다.
- 대기열 예외는 `queue/service_failure.log`와 state의 `service.failure`에 errno와 traceback을 남깁니다. 터미널 EIO를 주입해 다음 모델까지 완료되고 로그가 남는지, 실제 파일 쓰기 EIO는 전파되는지, SIGTERM 이름과 signal handler 복원, 실패 traceback 저장을 검사했습니다.
- 사용자 접두사를 필수로 요구하지 않습니다. 일반 이름은 shared 브랜치, 선택적 userN_ 접두사는 해당 브랜치로 매핑합니다. 한글·공백·대문자 확장자를 허용하고 실제 모델명은 provenance에 남깁니다. 결과 실행 폴더는 기존 업로더와 호환되도록 안전한 ASCII 이름으로 정리합니다.
- 최종 전체 **70개 중 68개 통과, Windows symlink 권한 제한 2개 건너뜀**입니다. 작은 실제 ONNX를 일반 대문자 파일명과 한글·공백 파일명으로 저장해 공통 COCO 추론·채점을 확인했습니다. 임시 로컬 Git/LFS 저장소에서 한글·공백·대문자 모델의 shared 브랜치 업로드와 실제 LFS 포인터도 확인했습니다. 이번 수정본의 Raspberry Pi 재실행은 아직 하지 않았습니다.
