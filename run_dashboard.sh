#!/usr/bin/env bash
# =============================================================================
#  TSMAE Experiment Dashboard — 바로가기 런처 (shortcut launcher)
# -----------------------------------------------------------------------------
#  한 번에 대시보드를 띄웁니다.  ./results/experiments 를 자동 인식해서 8개 뷰 +
#  GIF Studio 를 http://127.0.0.1:<PORT>/ 에 서빙합니다.
#
#  사용법:
#     ./run_dashboard.sh              # 포트 8000, 브라우저 자동 오픈
#     ./run_dashboard.sh 8123         # 포트 지정
#     ./run_dashboard.sh --no-open    # 브라우저 자동 오픈 안 함
#     ./run_dashboard.sh --setup      # venv/dist 강제 재설치(없으면 자동 설치됨)
#     Ctrl-C 로 종료.
#
#  안전: 격리된 UI/.venv (dc_vis 와 무관) 만 사용, results/.trash 는 read-only,
#        GPU 미사용.  실행 중인 학습/콘다 환경을 전혀 건드리지 않습니다.
# =============================================================================
set -euo pipefail

# --- 0) 경로/옵션 ------------------------------------------------------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PORT="${TSMAE_PORT:-8000}"
# Bind 0.0.0.0 by default (2026-06-11): on WSL2 a service bound to 127.0.0.1 *inside*
# WSL is not always reachable from the Windows browser (localhost-forwarding quirk →
# "running but ERR_CONNECTION_REFUSED"). 0.0.0.0 is reliably reached via Windows
# localhost and stays NAT-isolated to the host. Override with TSMAE_HOST=127.0.0.1.
HOST="${TSMAE_HOST:-0.0.0.0}"
OPEN_BROWSER=1
FORCE_SETUP=0
BASE_PY="/home/ykio/anaconda3/bin/python3"   # NON-dc_vis base python (격리 venv 용)

for arg in "$@"; do
  case "$arg" in
    --no-open)       OPEN_BROWSER=0 ;;
    --setup)         FORCE_SETUP=1 ;;
    --help|-h)       sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    ''|*[!0-9]*)     ;;                      # 숫자가 아니면 무시
    *)               PORT="$arg" ;;          # 숫자면 포트
  esac
done

VENV="$ROOT/UI/.venv"
PYBIN="$VENV/bin/python"
DIST="$ROOT/UI/frontend/dist/index.html"

say() { printf '\033[1;36m[dashboard]\033[0m %s\n' "$*"; }
warn(){ printf '\033[1;33m[dashboard]\033[0m %s\n' "$*"; }

# --- 1) 격리 venv 보장 (dc_vis 미접촉) --------------------------------------
if [[ $FORCE_SETUP -eq 1 || ! -x "$PYBIN" ]]; then
  say "격리 venv 생성/설치 중 (UI/.venv — dc_vis 와 무관)…"
  [[ -x "$BASE_PY" ]] || BASE_PY="$(command -v python3)"
  "$BASE_PY" -m venv "$VENV"
  "$PYBIN" -m pip install -q -U pip
  "$PYBIN" -m pip install -q -r "$ROOT/UI/requirements.txt"
  say "venv 준비 완료."
fi

# --- 2) 프론트엔드 dist 보장 (없으면 빌드; npm 없으면 placeholder 로 진행) ----
if [[ $FORCE_SETUP -eq 1 || ! -f "$DIST" ]]; then
  if command -v npm >/dev/null 2>&1; then
    say "프론트엔드 빌드 중 (npm; 빌드타임 전용, venv 미오염)…"
    ( cd "$ROOT/UI/frontend" && (npm ci || npm install) && npm run build )
    say "dist 빌드 완료."
  else
    warn "npm 없음 → dist 미생성. 백엔드는 뜨지만 / 는 placeholder 입니다 (API 는 정상)."
  fi
fi

# --- 3) 포트 점유 확인 -------------------------------------------------------
if "$PYBIN" - "$HOST" "$PORT" <<'PY' 2>/dev/null; then :; else
import socket,sys
h,p=sys.argv[1],int(sys.argv[2])
s=socket.socket(); s.settimeout(0.3)
sys.exit(0 if s.connect_ex((h,p))!=0 else 1)
PY
  warn "포트 $PORT 이 이미 사용 중입니다. 다른 포트로:  ./run_dashboard.sh 8123"
  exit 1
fi

URL="http://$HOST:$PORT/"

# --- 4) 브라우저 자동 오픈 (best-effort, 백그라운드) -------------------------
if [[ $OPEN_BROWSER -eq 1 ]]; then
  ( sleep 2
    if command -v explorer.exe >/dev/null 2>&1; then explorer.exe "$URL" >/dev/null 2>&1
    elif command -v wslview     >/dev/null 2>&1; then wslview "$URL"     >/dev/null 2>&1
    elif command -v xdg-open    >/dev/null 2>&1; then xdg-open "$URL"    >/dev/null 2>&1
    fi ) &
fi

# --- 5) 서버 기동 (Ctrl-C 로 종료) ------------------------------------------
say "TSMAE Experiment Dashboard  →  $URL"
say "  health: ${URL}api/health   ·   API docs: ${URL}docs"
say "  종료: Ctrl-C"
exec "$PYBIN" -m uvicorn app.main:app --app-dir "$ROOT/UI/backend" --host "$HOST" --port "$PORT"
