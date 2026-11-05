# 검증 기록

2026-11-05, Windows x86-64 CPU / Python 3.12.14에서 검증했습니다. 설치 버전은 requirements.txt와 같으며 ONNX Runtime 1.22.1입니다. 이 기록은 Raspberry Pi ARM64 실기기 성능 측정을 뜻하지 않습니다.

- 전체 34개 테스트: **32개 통과, 2개 Windows symlink 생성 권한 제한으로 건너뜀**.
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
