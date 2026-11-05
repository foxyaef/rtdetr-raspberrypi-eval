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

## 단일 파일 자동 준비 업데이트

- 빈 작업 폴더에 생성된 `test.py` 하나만 복사해 별도 `.rtdetr_venv` 생성, 고정 Python 패키지 자동 설치, 실제 공통 데이터 읽기 검증까지 실행했습니다. 저장소나 보조 Python 파일을 해당 폴더에 미리 복사하지 않았습니다.
- 같은 복사본과 새로 설치된 전용 가상환경으로 실제 Encoder W8A8 모델 3장 평가를 끝까지 실행해 mAP 계산·자원 측정·ZIP 저장을 확인했습니다. `full_evaluation=false`, 전력 값은 null입니다. 실기기 Raspberry Pi 벤치마크나 전체 1,000장 결과라는 주장은 하지 않습니다.
- 단일 파일 검사 16개는 작업 경로 판별, 원본 사용자 파일 보존, runtime 손상·경로 탈출 차단, `sys.prefix` 검증, pip 설치 경로 설정 격리, 공백·Unicode 경로, 복사본의 실제 오프라인 데이터 준비를 포함합니다.
- Debian의 pip/ensurepip 누락 시 공식 PyPA fallback 설치 경로는 mock 검사로 확인했습니다. 실제 새 가상환경 설치 검사는 이 Windows 호스트의 ensurepip 경로를 사용했습니다.
