# SOTA Baseline 10개 통합 체크리스트

**연동**: `SOTA_BASELINE_10_INTEGRATION_PLAN.md`

상태: ⬜ pending / 🟨 in_progress / ✅ completed / ❌ blocked

---

## Phase 0: 사전 준비

- ⬜ `./.trash/260519/` 백업 디렉토리 생성
- ⬜ 6개 핵심 파일 백업 완료
- ⬜ `dc_vis` 환경 검증 (torch, sklearn, performer-pytorch)
- ⬜ `scripts/verify_baseline_integration.py` skeleton 생성

## Phase 1: TFMAE + NPSR

### TFMAE
- ✅ `comparison/baselines/tfmae/` 생성
- ✅ Upstream `LMissher/TFMAE` `model/{MTFAE,attn,embed}.py` vendoring → `model.py` (358 lines, device-agnostic 수정)
- ✅ `wrapper.py` (`TFMAEBaseline`) — adversarial KL training loop, max-aggregation 점수
- ✅ `__init__.py`
- ✅ `comparison/baselines/__init__.py` import 등록
- ✅ `comparison/baseline_common.py`: HAS_TFMAE try/except + SOTA_MODELS/SOTA_AVAILABILITY/BASELINE_MODELS + MODEL_PRESETS + create_model dispatch
- ✅ `comparison/experiment_configs.py`: STANDARD_BASELINES key 추가
- ✅ Import + instantiate verification 통과 (`from comparison.baselines import TFMAEBaseline` + `create_model('tfmae', n_features=8)` OK)
- ⏸ Phase 1 small smoke test — 실행 정책에 따라 후속 일괄 진행

### NPSR
- ⬜ `comparison/baselines/npsr/` 생성
- ⬜ Upstream `andrewlai61616/NPSR` `models/NPSR.py` vendoring
- ❌ **BLOCKER**: `performer-pytorch` 패키지 미설치. 환경 `dc_vis`에 `pip install performer-pytorch` 필요. 현재 정책 "스크립트 실행 금지" 때문에 보류. → 사용자 승인 또는 별도 세션 필요.
- ⬜ `wrapper.py` (`NPSRBaseline`) — 2-model (M_pt + M_seq) 학습 (Blocker 해결 후)
- ⬜ `__init__.py`
- ⬜ register × 3
- ⬜ Import-only verification
- ⬜ Phase 1 small smoke test

## Phase 2: TimesNet + DCdetector

### TimesNet
- ✅ `comparison/baselines/timesnet/` 생성
- ✅ Upstream `thuml/Time-Series-Library` `models/TimesNet.py` + `layers/{Conv_Blocks, Embed}.py` vendoring (anomaly_detection 태스크만, 단일 model.py)
- ✅ `wrapper.py` (`TimesNetBaseline`) — MSE recon, max-aggregation
- ✅ `__init__.py`
- ✅ register × 3
- ✅ End-to-end verification 통과 (1 epoch fit + predict shape=(N,) dtype=float32)
- ⏸ Phase 2 small smoke test — 실행 정책에 따라 후속 일괄

### DCdetector
- ⬜ `comparison/baselines/dcdetector/` 생성
- ⬜ Upstream `DAMO-DI-ML/KDD2023-DCdetector` `model/*.py` vendoring (LICENSE 없음 → attribution)
- ⬜ `wrapper.py` (`DCdetectorBaseline`) — KL discrepancy only
- ⬜ `__init__.py`
- ⬜ register × 3
- ⬜ Import-only verification
- ⬜ Phase 2 small smoke test

## Phase 3: MEMTO + ModernTCN

### MEMTO
- ⬜ `comparison/baselines/memto/` 생성
- ⬜ Upstream `gunny97/MEMTO` `model/*.py` vendoring
- ⬜ `wrapper.py` (`MEMTOBaseline`) — 2-phase: K-means init + main training
- ⬜ `__init__.py`
- ⬜ register × 3
- ⬜ Import-only verification (sklearn.cluster.KMeans 의존 확인)
- ⬜ Phase 3 small smoke test

### ModernTCN
- ⬜ `comparison/baselines/moderntcn/` 생성
- ⬜ Upstream `luodhhh/ModernTCN` `ModernTCN-detection/models/*.py` vendoring (MIT)
- ⬜ `wrapper.py` (`ModernTCNBaseline`) — pure conv recon
- ⬜ `__init__.py`
- ⬜ register × 3
- ⬜ Import-only verification
- ⬜ Phase 3 small smoke test

## Phase 4: CAROTS + AnomalyBERT

### CAROTS
- ⬜ `comparison/baselines/carots/` 생성
- ⬜ Upstream `kimanki/CAROTS` `models/carots/*.py` vendoring (carots backbone만)
- ⬜ `wrapper.py` (`CAROTSBaseline`) — causality-aware contrastive (pos/neg augmenter)
- ⬜ `__init__.py`
- ⬜ register × 3
- ⬜ Import-only verification
- ⬜ Phase 4 small smoke test

### AnomalyBERT
- ⬜ `comparison/baselines/anomalybert/` 생성
- ⬜ Upstream `Jhryu30/AnomalyBERT` `models/{transformer, anomaly_transformer}.py` + degradation 함수 vendoring
- ⬜ `wrapper.py` (`AnomalyBERTBaseline`) — 4-type synthetic outlier injection, BCE loss
- ⬜ `__init__.py`
- ⬜ register × 3
- ⬜ Import-only verification (win_size=512 GPU 검증)
- ⬜ Phase 4 small smoke test

## Phase 5: CrossAD + CATCH

### CrossAD
- ⬜ `comparison/baselines/crossad/` 생성
- ⬜ Upstream `decisionintelligence/CrossAD` `models/CrossAD/*.py` vendoring
- ⬜ `wrapper.py` (`CrossADBaseline`) — cross-scale recon + query library
- ⬜ `__init__.py`
- ⬜ register × 3
- ⬜ Import-only verification
- ⬜ Phase 5 small smoke test

### CATCH
- ⬜ `comparison/baselines/catch/` 생성
- ⬜ Upstream `decisionintelligence/CATCH` `ts_benchmark/baselines/catch/*` vendoring (TAB framework 의존 제거)
- ⬜ `wrapper.py` (`CATCHBaseline`) — channel-aware frequency patching
- ⬜ `__init__.py`
- ⬜ register × 3
- ⬜ Import-only verification (relative import 정리 확인)
- ⬜ Phase 5 small smoke test

## Phase 6: 문서 + 검증 마무리

- ⬜ `comparison/MODELS.md` 신규 섹션 10개 추가 (#17~#26), 상단 count update
- ⬜ `comparison/GUIDE.md` 디렉토리 구조 + 모델 분류 update
- ⬜ `docs/CHANGELOG.md` 새 entry 추가
- ⬜ 전체 10개 모델 import-only verification 일괄 통과
- ⬜ `python -c "from comparison.baselines import *; print('OK')"` 통과
- ⬜ `git add docs/CHANGELOG.md && git commit && git push` (comparison/는 gitignored)

## Phase 7: Q1/Q3 실험 실행 계획 (Notion subpage 별도 작성, 실행 ❌)

- ⬜ Notion `MAE for Anomaly Detection` 아래 신규 subpage 생성: "10개 신규 SOTA × Q1/Q3 실험 실행 계획"
- ⬜ 26 models × 7 dataset-groups × 2 conditions 매트릭스 정의
- ⬜ 결과 디렉토리 구조 (기존 results/experiments/{N}_...과 정합)
- ⬜ 모니터링 명세: CPU/GPU/GPU mem/sys mem, 20분 간격
- ⬜ 모니터링 스크립트 spec (실행 ❌, code만 작성)
- ⬜ Phase별 실행 순서, ETA 추정
- ⬜ Failure recovery 절차 (resume from checkpoint)
- ⬜ MAE 결과 합치기 (`add_mae_results.py`) 절차

---

## 모델별 위험 fast-reference

| 모델 | Critical risk | 완화 |
|---|---|---|
| TFMAE | Adversarial KL stopgrad 수식 오류 가능 | TFMAE 논문 Eq.15 정확히 구현 |
| NPSR | `performer-pytorch` 의존 | install or vendor Performer attention |
| TimesNet | FFT GPU memory | batch_size=128, d_model=64 |
| DCdetector | LICENSE 없음 | attribution만 명시, 상업적 사용 불가 |
| MEMTO | K-means init 시간 | 10% 샘플링 사용 (원본 방식) |
| ModernTCN | hparam 많음 | 원본 SWaT.sh 정확히 복제 |
| CAROTS | 3-component (pos/neg augmenter + scorer) | default carots backbone만 사용 |
| AnomalyBERT | win_size=512 GPU OOM | WaDi (D=123)에서 batch_size 32로 |
| CrossAD | Query library memory | batch_size 64로 조정 |
| CATCH | TAB framework relative import | 모든 import 직접 호출로 변환 |

---

## 진행률 추적

- Phase 0: 0/4 항목
- Phase 1: 0/18 항목 (TFMAE 9 + NPSR 9)
- Phase 2: 0/14 항목 (TimesNet 7 + DCdetector 7)
- Phase 3: 0/14 항목 (MEMTO 7 + ModernTCN 7)
- Phase 4: 0/14 항목 (CAROTS 7 + AnomalyBERT 7)
- Phase 5: 0/14 항목 (CrossAD 7 + CATCH 7)
- Phase 6: 0/6 항목 (문서 + 최종 검증)
- Phase 7: 0/8 항목 (Notion 실험 계획)

**전체: 0/92 항목**
