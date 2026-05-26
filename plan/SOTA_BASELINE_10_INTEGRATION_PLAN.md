# SOTA Baseline 10개 통합 실행 계획 — 종합 (2026-05-19)

**문서 목적**: [Notion subpage](https://www.notion.so/36487856b20781a29441e1ddf95900a0)의 통합 계획을 코드-베이스로 실행하기 위한 step-by-step plan.

**원칙** (사용자 명시 2026-05-19):
1. `comparison/` 및 `mae_anomaly/` 파이프라인 100% 일관성
2. 각 모델 default 하이퍼파라미터는 기존(Notion 명세) 그대로 사용
3. 코드 수정 전 `./.trash/260519/`에 원본 백업
4. 오류 발생 시 임시방편 ❌ → 근본 원인 식별
5. **현재 실행 금지** — Q1/Q3 실험은 별도 plan으로 (Notion에)

**Plan audit 후 확정사항 (2026-05-19 사용자 결정)**:
- THOC 불일치(MODELS.md 18 vs code 17)는 무시 (사용자 의도된 상태로 유지)
- NPSR blocker: `pip install performer-pytorch` 승인. **단 설치 타이밍은 사용자의 GPU 실험이 진행 중이지 않은 시점에 진행** (사용자 실험과 충돌 시점만 회피)
- LSTM-NDT 추가 안 함 → 10개 plan 그대로
- Phase 우선순위 재정렬 없음 → 어차피 10개 다 통합. 기존 Phase 1→5 순서 유지
- 결과 디렉토리는 **기존 results/experiments/{N}_... 디렉토리에 force 추가** (분리 안 함)
- 필요 dependency 모두 정상 설치 (CAROTS의 yacs, performer-pytorch 등)
- 실행 시간 추정 무시, 진행하면서 측정
- 백업 정책 유지 (`./.trash/<YYMMDD>/`)

## ⚠️ 설치 타이밍 주의 (2026-05-19 사고 교훈)

`pip install performer-pytorch`는 dependency cascade로 torch/CUDA/numpy 등을 메이저 업그레이드한다. **이는 정상 동작**이며 다음 설치 시 그대로 진행한다.

**유일한 제약**: 사용자의 GPU 실험이 진행 중인 시점에는 설치 보류 (실험 crash 방지). 사용자 실험 종료 확인 후 즉시 진행.

확인 명령:
```bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader
ps aux | grep -E "python.*run_(base|baseline)" | grep -v grep
```

GPU 사용량이 낮고 (≤ 100MB), python 학습 프로세스가 없을 때 설치 진행.

각 모델 dependency:
- NPSR: `pip install performer-pytorch` 필요 (cascade로 torch 등 동시 업그레이드 — 정상)
- CAROTS: `pip install yacs` 또는 omegaconf (config용)
- MEMTO: sklearn ✅ (이미 있음)
- 나머지 (ModernTCN/DCdetector/AnomalyBERT/CrossAD/CATCH): 표준 PyTorch만으로 운영 가능 추정

**기준 모델 목록** (10개):

| Phase | 모델 | 학회 |
|---|---|---|
| 1 | TFMAE | ICDE'24 |
| 1 | NPSR | NeurIPS'23 |
| 2 | TimesNet | ICLR'23 |
| 2 | DCdetector | KDD'23 |
| 3 | MEMTO | NeurIPS'23 |
| 3 | ModernTCN | ICLR'24 Spot |
| 4 | CAROTS | ICML'25 |
| 4 | AnomalyBERT | ICLR'23 WS |
| 5 | CrossAD | NeurIPS'25 |
| 5 | CATCH | ICLR'25 |

---

## 0. 사전 준비

### 0.1 백업 (모든 수정 전 1회만)

```bash
mkdir -p ./.trash/260519/comparison/baselines
mkdir -p ./.trash/260519/docs

cp comparison/baselines/__init__.py ./.trash/260519/comparison/baselines/
cp comparison/baseline_common.py ./.trash/260519/comparison/
cp comparison/experiment_configs.py ./.trash/260519/comparison/
cp comparison/MODELS.md ./.trash/260519/comparison/
cp comparison/GUIDE.md ./.trash/260519/comparison/
cp docs/CHANGELOG.md ./.trash/260519/docs/
```

### 0.2 환경 검증

```bash
conda activate dc_vis
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import sklearn; print(sklearn.__version__)"  # MEMTO K-means용
python -c "import performer_pytorch" 2>&1 | head -1   # NPSR용 (없으면 설치 필요)
```

---

## 1. 모델별 통합 절차 (10회 반복)

각 모델 통합 시 다음 7단계를 수행:

### Step A: 디렉토리 생성

```bash
mkdir -p comparison/baselines/<model_name>/
```

### Step B: model.py 작성

- 공식 repo의 핵심 model 파일을 단일 `model.py`로 vendoring
- 상단 docstring에 attribution 명시:
  ```python
  """
  <Model> for Baseline Comparison
  
  Based on: "<Paper Title>"
  Paper: <Venue Year>, <arXiv/DOI link>
  Original code: <github URL> (<License>)
  
  Modified for:
  - Device-agnostic operation
  - Integration with comparison framework (sliding window, epoch_callback)
  """
  ```
- 의존 layer/embed/attn 파일은 동일 model.py 내부에 통합 (subdirectory 금지 — 기존 패턴 준수)

### Step C: wrapper.py 작성

- `class <Model>Baseline:` 정의
- 인터페이스 계약 (전 baseline 공통):
  ```python
  def __init__(self, **hparams, device=None, verbose=True): ...
  
  @property
  def name(self) -> str: ...
  
  def fit(self, train_X: np.ndarray, epoch_callback=None) -> '<Model>Baseline':
      # train_X: (N_train, n_features), normalized
      # train_loss_history list 유지
      # 각 epoch 끝마다 epoch_callback(self, ep+1) 호출 (있으면)
      return self
  
  def predict(self, test_X: np.ndarray) -> np.ndarray:
      # returns: (N_test,) float32 anomaly score per timestep
      ...
  
  def save(self, save_dir: Path) -> None: ...  # optional
  def load(self, save_dir: Path) -> '<Model>Baseline': ...  # optional
  ```
- 기존 `anomaly_transformer/wrapper.py`, `usad/model.py`, `omnianomaly/model.py` 패턴 그대로 차용

### Step D: __init__.py 작성

```python
"""
<Model> for Baseline Comparison

Based on: "<Paper Title>"
Paper: <Venue Year>, <arXiv/DOI link>
Original code: <github URL>
"""

from .model import <Model>  # main model class
from .wrapper import <Model>Baseline

__all__ = ['<Model>', '<Model>Baseline']
```

### Step E: comparison/baselines/__init__.py 업데이트

```python
# Add to existing imports:
from .timesnet import TimesNetBaseline
from .dcdetector import DCdetectorBaseline
# ... (10개)

# Add to __all__ list
```

### Step F: comparison/baseline_common.py 업데이트

3개 위치 수정:

**1. Optional import 블록** (line ~60 부근, try/except 패턴):
```python
try:
    from comparison.baselines import TimesNetBaseline
    HAS_TIMESNET = True
except ImportError:
    HAS_TIMESNET = False
# ... (10개 모델)
```

**2. SOTA_MODELS 리스트** (line ~112):
```python
SOTA_MODELS = [
    'gcn_lstm', 'anomaly_transformer', 'tranad', 'usad', 'dagmm', 'gdn',
    'omnianomaly',
    # 신규 10개:
    'timesnet', 'dcdetector', 'memto', 'moderntcn', 'tfmae', 'npsr',
    'anomalybert', 'carots', 'crossad', 'catch',
]
```

**3. BASELINE_MODELS, SOTA_AVAILABILITY** (동일 line 부근):
- BASELINE_MODELS에도 10개 key 추가
- SOTA_AVAILABILITY dict에 `'timesnet': HAS_TIMESNET, ...` 10개 추가

**4. _get_default_model_params()** (line ~144): MODEL_PRESETS 등록:
```python
'timesnet': {
    'win_size': 100, 'd_model': 64, 'd_ff': 64,
    'e_layers': 3, 'top_k': 3, 'num_kernels': 6,
    'dropout': 0.1, 'lr': 1e-4,
    'train_stride': 1, 'epochs': 10, 'batch_size': 128,
},
# ... (10개)
```

(전체 hyperparam 명세는 §3 참조)

### Step G: comparison/experiment_configs.py 업데이트

`STANDARD_BASELINES` 리스트에 10개 key 추가 (line 24~29):
```python
STANDARD_BASELINES = [
    'random', 'sensor_range', 'pca_error', 'l2_norm', 'nn_distance',
    'mlp', 'mlpmixer', 'transformer', 'gcn_lstm',
    'anomaly_transformer', 'tranad', 'usad', 'dagmm', 'gdn', 'omnianomaly',
    # === 신규 10개 (2026-05-19) ===
    'timesnet', 'dcdetector', 'memto', 'moderntcn',
    'tfmae', 'npsr',
    'anomalybert', 'carots',
    'crossad', 'catch',
]
```

---

## 2. 모델별 Vendoring 명세 (10개)

### 2.1 TimesNet
- **Upstream**: `thuml/Time-Series-Library` (MIT)
- **파일**:
  - `models/TimesNet.py` (Model class + TimesBlock)
  - `layers/Conv_Blocks.py` (Inception_Block_V1)
  - `layers/Embed.py` (DataEmbedding)
- **단일 `model.py`로 통합**: 위 3개 파일을 import 순서대로 합치고 `class TimesNet(nn.Module)` export

### 2.2 DCdetector
- **Upstream**: `DAMO-DI-ML/KDD2023-DCdetector` (LICENSE 없음 → attribution만)
- **파일**:
  - `model/DCdetector.py`
  - `model/RevIN.py`
  - `model/attn.py` (AnomalyAttention, DAC_structure)
  - `model/embed.py` (DataEmbedding, TokenEmbedding)
- **단일 `model.py`로 통합**

### 2.3 MEMTO
- **Upstream**: `gunny97/MEMTO`
- **파일**:
  - `model/Transformer.py` (Encoder)
  - `model/attn_layer.py` (FullAttention)
  - `model/embedding.py` (DataEmbedding)
  - `model/ours_memory_module.py` (MemoryUnit)
  - `model/loss_functions.py` (gathering_loss)
- **추가 의존**: `sklearn.cluster.KMeans` (Phase 1 init)
- **단일 `model.py`로 통합**

### 2.4 ModernTCN
- **Upstream**: `luodhhh/ModernTCN` (MIT)
- **파일**:
  - `ModernTCN-detection/models/ModernTCN.py`
  - `ModernTCN-detection/models/ModernTCN_Layer.py`
- **단일 `model.py`로 통합**

### 2.5 TFMAE ★
- **Upstream**: `LMissher/TFMAE` (MIT)
- **파일**:
  - `model/MTFAE.py` (dual encoder/decoder)
  - `model/attn.py` (FullAttention)
  - `model/embed.py` (DataEmbedding)
- **단일 `model.py`로 통합**

### 2.6 NPSR ★
- **Upstream**: `andrewlai61616/NPSR`
- **파일**:
  - `models/NPSR.py` (M_pt + M_seq + Performer attention)
- **추가 의존**: `performer-pytorch` package (또는 Performer 코드 vendoring)
- **단일 `model.py`로 통합**

### 2.7 AnomalyBERT
- **Upstream**: `Jhryu30/AnomalyBERT`
- **파일**:
  - `models/transformer.py` (PreLN Transformer)
  - `models/anomaly_transformer.py` (rel pos bias encoder)
  - `train.py`의 4-type degradation 함수 추출 (`soft_replacement`, `uniform_replacement`, `length_adjustment`, `peak_noise`)
- **단일 `model.py`로 통합**

### 2.8 CAROTS
- **Upstream**: `kimanki/CAROTS`
- **파일** (carots backbone만):
  - `models/carots/encoder.py`
  - `models/carots/modeling_carots.py`
  - `models/carots/modeling_positive_augmentor.py`
  - `models/carots/modeling_negative_augmentor.py`
  - `models/carots/scorer_carots.py`
  - `models/carots/transform_layer.py`
- **iTransformer/TimesNet variant는 제외**
- **단일 `model.py`로 통합**

### 2.9 CrossAD
- **Upstream**: `decisionintelligence/CrossAD`
- **파일**:
  - `models/CrossAD/Basic_CrossAD.py`
  - `models/CrossAD/Attention_Blocks.py`
  - `models/CrossAD/Context_Blocks.py`
  - `models/CrossAD/EncDec.py`
- **단일 `model.py`로 통합**

### 2.10 CATCH
- **Upstream**: `decisionintelligence/CATCH` (TAB framework 일부)
- **파일** (catch/ subdir만):
  - `ts_benchmark/baselines/catch/models/CATCH_model.py`
  - `ts_benchmark/baselines/catch/layers/RevIN.py`
  - `ts_benchmark/baselines/catch/layers/channel_mask.py`
  - `ts_benchmark/baselines/catch/layers/cross_channel_Transformer.py`
  - `ts_benchmark/baselines/catch/utils/ch_discover_loss.py`
  - `ts_benchmark/baselines/catch/utils/fre_rec_loss.py`
- **TAB 프레임워크 의존 제거**: relative import (`from ts_benchmark...`)를 같은 파일 내 직접 호출로 대체
- **단일 `model.py`로 통합**

---

## 3. MODEL_PRESETS 등록 명세 (10개)

Notion subpage §2.1-§2.10에서 명시된 그대로:

```python
# comparison/baseline_common.py:_get_default_model_params() 추가

# Phase 1
'tfmae': {
    'win_size': 100, 'd_model': 128, 'n_heads': 8, 'e_layers': 3,
    'temporal_mask_ratio': 0.5, 'freq_mask_threshold': 'auto',
    'lr': 1e-4, 'train_stride': 1, 'epochs': 10, 'batch_size': 64,
},
'npsr': {
    'win_size': 100, 'induction_length': 16,
    'd_model': 256, 'n_heads': 4, 'e_layers': 4, 'dropout': 0.1,
    'theta_N': 0.985,
    'lr': 1e-4, 'train_stride': 1, 'epochs': 10, 'batch_size': 64,
},

# Phase 2
'timesnet': {
    'win_size': 100, 'd_model': 64, 'd_ff': 64,
    'e_layers': 3, 'top_k': 3, 'num_kernels': 6,
    'dropout': 0.1, 'lr': 1e-4,
    'train_stride': 1, 'epochs': 10, 'batch_size': 128,
},
'dcdetector': {
    'win_size': 105, 'patch_size': [3, 5, 7],
    'd_model': 256, 'n_heads': 1, 'e_layers': 3,
    'dropout': 0.0, 'lr': 1e-4,
    'train_stride': 1, 'epochs': 10, 'batch_size': 128,
},

# Phase 3
'memto': {
    'win_size': 100, 'd_model': 512, 'n_heads': 8, 'e_layers': 3,
    'n_memory': 10, 'shrink_thres': 0.0025, 'lambda_entropy': 0.01,
    'phase1_epochs': 3, 'lr': 1e-4,
    'train_stride': 1, 'epochs': 10, 'batch_size': 64,
},
'moderntcn': {
    'win_size': 96, 'stem_ratio': 6, 'downsample_ratio': 2,
    'ffn_ratio': 1, 'patch_size': 8, 'patch_stride': 4,
    'num_blocks': [1], 'large_size': [13], 'small_size': [5],
    'dims': [32], 'head_dropout': 0.0, 'dropout': 0.3,
    'lr': 1e-4, 'train_stride': 1, 'epochs': 10, 'batch_size': 128,
},

# Phase 4
'carots': {
    'win_size': 100, 'd_model': 256, 'n_heads': 8, 'e_layers': 3,
    'patch_size': 10, 'patch_stride': 10,
    'contrastive_margin': 1.0,
    'pos_aug_strength': 0.1, 'neg_aug_strength': 0.3,
    'lr': 1e-4, 'train_stride': 1, 'epochs': 10, 'batch_size': 64,
},
'anomalybert': {
    'win_size': 512, 'd_model': 512, 'n_heads': 8, 'e_layers': 6,
    'dropout': 0.1, 'lr': 1e-4, 'degradation_ratio': 0.15,
    'train_stride': 1, 'epochs': 10, 'batch_size': 64,
},

# Phase 5
'crossad': {
    'win_size': 100, 'd_model': 256, 'n_heads': 8, 'e_layers': 3,
    'n_scales': 3, 'query_lib_size': 64,
    'lr': 1e-4, 'train_stride': 1, 'epochs': 10, 'batch_size': 64,
},
'catch': {
    'win_size': 96, 'patch_size': 24, 'patch_stride': 12,
    'd_model': 128, 'n_heads': 8, 'e_layers': 2,
    'ch_mask_ratio': 0.3, 'lambda_ch_discover': 0.5,
    'lr': 1e-4, 'train_stride': 1, 'epochs': 10, 'batch_size': 128,
},
```

---

## 4. 검증 (구현 후, 실행 전 — 간단한 코드만)

각 모델 통합 후 import-only verification (실제 학습/평가 실행 ❌):

```python
# scripts/verify_baseline_integration.py (신규 생성 권장)
"""Import-only verification for new baselines (no training)."""

def verify(model_key, baseline_class):
    print(f"\n=== {model_key} ===")
    # 1. Class instantiation
    cfg = MODEL_PRESETS['default']()[model_key]
    model = baseline_class(**cfg)
    print(f"  ✅ instantiated: {model.name}")
    
    # 2. Build internal model with tiny dummy (D=8, win_size aligned)
    import numpy as np
    win = cfg.get('win_size', cfg.get('seq_len', 100))
    dummy_train = np.random.randn(win * 10, 8).astype(np.float32)
    dummy_test = np.random.randn(win * 5, 8).astype(np.float32)
    
    # 3. fit() with 1 epoch only (override)
    model.epochs = 1
    model.fit(dummy_train)
    print(f"  ✅ fit() 1 epoch passed")
    
    # 4. predict() shape check
    scores = model.predict(dummy_test)
    assert scores.shape == (len(dummy_test),), f"shape mismatch: {scores.shape}"
    assert scores.dtype in (np.float32, np.float64), f"dtype: {scores.dtype}"
    print(f"  ✅ predict() shape=(N,) dtype={scores.dtype}")

if __name__ == '__main__':
    from comparison.baselines import (
        TimesNetBaseline, DCdetectorBaseline, MEMTOBaseline, ModernTCNBaseline,
        TFMAEBaseline, NPSRBaseline,
        AnomalyBERTBaseline, CAROTSBaseline,
        CrossADBaseline, CATCHBaseline,
    )
    
    targets = [
        ('timesnet', TimesNetBaseline), ('dcdetector', DCdetectorBaseline),
        ('memto', MEMTOBaseline), ('moderntcn', ModernTCNBaseline),
        ('tfmae', TFMAEBaseline), ('npsr', NPSRBaseline),
        ('anomalybert', AnomalyBERTBaseline), ('carots', CAROTSBaseline),
        ('crossad', CrossADBaseline), ('catch', CATCHBaseline),
    ]
    for key, cls in targets:
        try:
            verify(key, cls)
        except Exception as e:
            print(f"  ❌ {key} FAILED: {e}")
```

**검증 단계 통과 기준**:
1. Class instantiation 성공
2. 1-epoch fit() 무오류
3. predict() shape == (N_test,)
4. dtype float32 또는 float64

**실패 시**: 임시방편 금지. 원본 repo 코드와 비교, 통합 단계 어디서 오류 발생했는지 추적.

---

## 5. 문서 업데이트 (구현 완료 후)

### 5.1 `comparison/MODELS.md`
- 상단 count: 16 → 26 baselines (5 simple + 3 neural-simple + 8 SOTA + 10 신규 SOTA)
- 신규 섹션 추가 (#17~#26): 각 모델 1개 섹션 (anomaly_transformer 패턴과 동일 구조)

### 5.2 `comparison/GUIDE.md`
- §2 디렉토리 구조: 신규 10개 디렉토리 추가
- §7 모델 분류: SOTA 카테고리에 10개 추가
- §11 실험 큐: 신규 모델 명시

### 5.3 `docs/CHANGELOG.md`
- 새 entry: "Feat: Add 10 SOTA baselines (TimesNet, DCdetector, MEMTO, ModernTCN, TFMAE, NPSR, AnomalyBERT, CAROTS, CrossAD, CATCH)"

---

## 6. Phase별 실행 순서

각 Phase 완료 후 verification → 다음 Phase 시작.

**Phase 1** (TFMAE + NPSR) — 사용자 MAE 직접 비교 + F1\* 철학 일치
**Phase 2** (TimesNet + DCdetector) — Top-tier SOTA 필수, 통합 쉬움
**Phase 3** (MEMTO + ModernTCN) — Memory + Conv 카테고리
**Phase 4** (CAROTS + AnomalyBERT) — Contrastive causality + Self-supervised
**Phase 5** (CrossAD + CATCH) — 최신 2025 decisionintelligence 그룹

---

## 7. Q1/Q3 실험 실행 계획 (Notion에 별도 작성, 지금 실행 금지)

별도 Notion subpage로 정리:
- 26 models × 7 datasets (Simulation, SWaT, WaDi×2, SMD-28, PSM, Exathlon-6) × 2 conditions (Q1, Q3) 실험 매트릭스
- 모니터링: CPU/GPU 사용량, GPU memory, system memory, disk I/O (20분 간격)
- 결과 디렉토리 통합 (기존 results/experiments/{N}_... 구조)
- MAE 결과 합치기 (`add_mae_results.py`)

---

## 8. 위험 & 완화 (Notion subpage §5 참조)

10개 위험 항목과 완화 전략은 [Notion subpage §5](https://www.notion.so/36487856b20781a29441e1ddf95900a0) 참조.

---

**다음 단계**: `/plan/SOTA_BASELINE_CHECKLIST.md`에 모델별 체크리스트.
