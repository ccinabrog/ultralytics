
#!/bin/bash

# YOLO Region Counter 실행 스크립트
# 가상환경 활성화 후 YOLO 모델 실행

# 에러 발생 시 스크립트 중단
set -e

# 변수 설정
BASE_DIR="${HOME}/github/ultralytics"
VENV_PATH="${BASE_DIR}/.ul/bin/activate"
SCRIPT_PATH="${BASE_DIR}/examples/YOLOv8-Region-Counter/yolov8_region_counter.py"
SOURCE_PATH="/dev/video0"
WEIGHTS_PATH="${BASE_DIR}/dxnn_model/yolov8n.onnx"

# 색상 코드 (선택사항)
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}YOLO Region Counter 시작...${NC}"
echo "기본 디렉토리: $BASE_DIR"

# 기본 디렉토리 존재 확인
if [ ! -d "$BASE_DIR" ]; then
    echo -e "${RED}오류: 기본 디렉토리를 찾을 수 없습니다: $BASE_DIR${NC}"
    echo "다음을 확인해보세요:"
    echo "1. 경로가 올바른지 확인: $BASE_DIR"
    echo "2. ultralytics 저장소가 클론되어 있는지 확인"
    exit 1
fi

# 가상환경 활성화 상태 확인
if [ -n "$VIRTUAL_ENV" ]; then
    echo -e "${GREEN}가상환경이 이미 활성화되어 있습니다: $VIRTUAL_ENV${NC}"

    # 현재 활성화된 가상환경이 목표 가상환경과 같은지 확인
    TARGET_VENV_DIR="${BASE_DIR}/.ul"
    if [ "$VIRTUAL_ENV" != "$TARGET_VENV_DIR" ]; then
        echo -e "${RED}경고: 다른 가상환경이 활성화되어 있습니다.${NC}"
        echo "현재: $VIRTUAL_ENV"
        echo "목표: $TARGET_VENV_DIR"
        echo "현재 환경을 그대로 사용하시겠습니까? (Y/n)"
        read -r response
        if [[ "$response" =~ ^[Nn]$ ]]; then
            echo "스크립트를 중단합니다. 먼저 'deactivate' 명령으로 현재 환경을 비활성화하세요."
            exit 1
        fi
    fi
    SKIP_ACTIVATION=true
else
    SKIP_ACTIVATION=false
    # 가상환경 존재 확인 및 디버깅
    if [ ! -f "$VENV_PATH" ]; then
        echo -e "${RED}오류: 가상환경을 찾을 수 없습니다: $VENV_PATH${NC}"
        echo "다음 경로들을 확인해보세요:"
        echo "1. ${BASE_DIR}/.venv/bin/activate"
        echo "2. ${BASE_DIR}/venv/bin/activate"
        echo "3. ${BASE_DIR}/.ul/bin/activate"

        # 실제 존재하는 activate 파일 찾기
        echo -e "${GREEN}존재하는 activate 파일들:${NC}"
        find "$BASE_DIR" -name "activate" -type f 2>/dev/null || echo "activate 파일을 찾을 수 없습니다."
        exit 1
    fi
fi

# Python 스크립트 존재 확인
if [ ! -f "$SCRIPT_PATH" ]; then
    echo -e "${RED}오류: Python 스크립트를 찾을 수 없습니다: $SCRIPT_PATH${NC}"
    exit 1
fi

# 웨이트 파일 존재 확인
if [ ! -f "$WEIGHTS_PATH" ]; then
    echo -e "${RED}경고: 웨이트 파일을 찾을 수 없습니다: $WEIGHTS_PATH${NC}"
    echo "계속 진행하시겠습니까? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "스크립트를 중단합니다."
        exit 1
    fi
fi

# 가상환경 활성화 (필요한 경우에만)
if [ "$SKIP_ACTIVATION" = false ]; then
    echo "가상환경 활성화 중... ($VENV_PATH)"
    # 가상환경 활성화 시 에러 처리 (sh 호환성을 위해 . 사용)
    if ! . "$VENV_PATH"; then
        echo -e "${RED}가상환경 활성화 실패!${NC}"
        echo "다음을 시도해보세요:"
        echo "1. 수동으로 가상환경 생성: python -m venv ${BASE_DIR}/.ul"
        echo "2. conda 환경 사용: conda activate ultralytics"
        echo "3. 시스템 Python 사용 (가상환경 없이)"
        echo ""
        echo "가상환경 없이 계속 진행하시겠습니까? (y/N)"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            echo -e "${GREEN}시스템 Python으로 실행합니다...${NC}"
            # 가상환경 활성화 건너뛰기
        else
            exit 1
        fi
    else
        echo -e "${GREEN}가상환경이 활성화되었습니다: $VIRTUAL_ENV${NC}"
    fi
fi

echo "YOLO Region Counter 실행 중..."
echo "사용 중인 Python: $(which python)"
echo "Python 버전: $(python --version)"
if [ -n "$VIRTUAL_ENV" ]; then
    echo "가상환경: $VIRTUAL_ENV"
fi
echo "----------------------------------------"

python "$SCRIPT_PATH" \
    --source="$SOURCE_PATH" \
    --weights="$WEIGHTS_PATH" \
    --view-img

echo -e "${GREEN}스크립트 실행 완료${NC}"

