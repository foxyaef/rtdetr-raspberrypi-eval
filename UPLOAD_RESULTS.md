# 기존 결과 ZIP만 GitHub에 업로드

[`code/upload_results.py`](https://raw.githubusercontent.com/foxyaef/rtdetr-raspberrypi-eval/main/code/upload_results.py) **한 파일만** 복사하면 됩니다. Python 표준 라이브러리와 GitHub CLI `gh`를 사용하며 가상환경 활성화나 pip 설치가 필요하지 않습니다. 기존 `test.py`의 `UPLOAD_TO_GITHUB=False`와 독립적으로, 이 파일을 실행했을 때만 지정 ZIP을 올립니다.

기본 ZIP은 요청한 다음 파일입니다.

```text
/home/user1/results/user1_v1_20261106T011021_267952Z_28425c8c.zip
```

## 라즈베리파이 SSH 터미널에서 준비

```bash
sudo apt update
sudo apt install -y gh tmux
cd /home/user1
mkdir -p code
curl -fL https://raw.githubusercontent.com/foxyaef/rtdetr-raspberrypi-eval/main/code/upload_results.py -o code/upload_results.py
tmux new -s upload-test
```

위 경로는 평가 ZIP에 기록된 작업 폴더 기준입니다. 다른 작업 폴더를 사용한다면 그 폴더로 변경합니다. `upload_results.py`를 `code/` 안에 넣으면 그 위 폴더를 작업 경로로 인식하고, 단독으로 두면 그 파일 옆의 `results/`를 사용합니다.

이미 `upload-test` 세션이 있으면 `tmux attach -t upload-test`로 들어갑니다. tmux는 프로그램을 유지하는 도구이며 GitHub 인증은 같은 라즈베리파이 OS 계정에 저장된 gh 로그인 설정을 사용합니다.

## tmux 화면 안에서 로그인

```bash
gh auth login --hostname github.com --git-protocol https --web
gh auth status --hostname github.com
```

화면의 일회용 코드와 URL을 확인하고 PC의 웹브라우저에서 로그인합니다. Pi에 웹브라우저가 없어도 PC에서 인증할 수 있습니다. 이미 같은 계정으로 gh 로그인이 되어 있으면 status 확인만 하면 됩니다. 저장소에 쓰기 권한이 있는 GitHub 계정을 사용해야 합니다. 토큰을 Python 파일이나 명령 인자에 적을 필요는 없습니다. 이 도구는 gh API 인증을 사용하므로 Git credential helper나 Git LFS를 별도로 설정하지 않습니다.

공식 안내: [gh auth login](https://cli.github.com/manual/gh_auth_login), [gh api](https://cli.github.com/manual/gh_api).

## 점검 후 실제 업로드

```bash
# ZIP 검사 + 로그인 계정 + 저장소 쓰기 권한 조회. 업로드하지 않음.
python3 code/upload_results.py --check-only

# 위 점검이 성공하면 실제 업로드
python3 code/upload_results.py
```

업로드는 아래 경로에 원본 ZIP 이름으로 저장됩니다. 일반 평가기의 전체 모델/로그 업로드 경로와 겹치지 않게 zip_uploads 폴더를 사용합니다.

```text
foxyaef/rtdetr-raspberrypi-eval
└── user1 브랜치
    └── results/user1/zip_uploads/
        └── user1_v1_20261106T011021_267952Z_28425c8c.zip
```

출력의 `status: uploaded`, `github_login`, `github_zip_url`을 확인하세요. ZIP 옆에는 `<원본ZIP이름>.github_upload.json` 영수증도 저장됩니다. 같은 ZIP 재시도는 이미 업로드된 것으로 처리하고, 같은 경로에 내용이 다르면 덮어쓰지 않습니다. main은 변경하지 않으며 다른 원격 파일도 유지합니다. 실패해도 원본 ZIP을 지우지 않습니다.

다른 ZIP은 다음처럼 지정합니다. 상대 경로는 터미널 위치가 아닌 스크립트 작업 폴더 기준입니다.

```bash
python3 code/upload_results.py --zip results/user2_v2_20261106T120000_000000Z_abcd1234.zip
```

실제 파일은 평가기가 생성한 영문 실행 ID 이름을 유지하세요. 기본값을 바꾸고 싶으면 Python 파일 상단의 `ZIP_NAME`만 수정하면 됩니다. 모델에 userN_ 접두사가 있으면 그 브랜치를 쓰고, 일반 모델명은 shared 브랜치를 사용합니다.

## 저장공간과 결과 의미

- 모델 파일은 필요하지 않습니다. 모델 업로드·재추론·COCO 다운로드·ZIP 재압축·ZIP 압축 해제·저장소 clone을 하지 않습니다.
- **ZIP만** 백업합니다. ZIP 안의 개별 로그는 GitHub 폴더에 따로 풀지 않으며 원본 모델 파일도 올리지 않습니다.
- GitHub 전송용 base64 요청 파일을 시스템 임시 폴더에 잠깐 만들고 자동 삭제합니다. 대략 ZIP 크기의 1.34배와 여유 1MiB가 필요합니다. 현재 Pi의 `/tmp`는 RAM 기반이므로 SD카드 대신 그 공간을 사용합니다. gh 자체의 RAM도 사용합니다.
- 기본 임시 폴더가 부족하면 `--temp-dir /다른/실제/임시폴더`를 지정할 수 있습니다. ZIP은 최대 90MiB까지 지원합니다. 원본 파일을 이동하거나 삭제해 공간을 확보하는 기능은 없습니다.
- ZIP CRC와 평가 metadata를 확인합니다. `complete`이면 metrics.json도 있어야 하며, interrupted/failed ZIP은 진단 자료로 구분해 그대로 업로드합니다. 업로드 성공은 모델 평가 완료를 뜻하지 않으므로 `evaluation_status`도 확인하세요.
- 공개 저장소에 선택한 ZIP 전체를 올립니다. credential/configuration 경로가 들어 있으면 거부합니다.

tmux에서 빠져나오려면 Ctrl+B를 누르고 손을 뗀 뒤 소문자 d를 누릅니다. 재접속은 `tmux attach -t upload-test`입니다. 업로드 테스트는 성능 측정과 자원을 공유하므로 비교용 모델 평가가 끝난 뒤 진행하는 편이 좋습니다.

이 도구의 GitHub API 흐름·원본 바이트 보존·같은 파일 재시도·동시 브랜치 갱신은 모의 API 검사로 검증했습니다. 실제 Pi의 새 ZIP과 GitHub 쓰기 성공은 위 명령을 실행해 확인해야 합니다.
