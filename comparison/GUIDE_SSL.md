# GUIDE_SSL.md — Weakly-supervised 베이스라인 실험 세팅 가이드

> 대상: 2026-05-29/30 official-repo porting으로 추가된 **4개 weakly-supervised 모델**
> (`deepmil`, `wetas`, `treemil`, `nrdetector`).
> 이 가이드의 목적은 **실험 세팅을 위한 파라미터 정리** — 각 파라미터가 *무엇이고, 무엇을 의미하며,
> 사용자가 무엇을 결정해야 하는지*를 ground-truth 코드 기준으로 안내한다.
> label-사용 학습 pipeline은 **이미 코드에 구현**(각 wrapper)되어 있고, 원본 논문/repo와 대조 검증을 마쳤다(§2, §6).
>
> **현재 상태(중요): GPU 미실행.** 코드/정규화/preset porting은 완료됐으나 어떤 데이터셋에 대해서도
> 학습/추론을 돌린 적이 없다 — `comparison/results/` 아래 weak result artifact는 **0건**.
> 본 가이드는 "어떻게 세팅하고 돌리는가"의 ground-truth이며 **성능 수치는 포함하지 않는다(아직 없음, 과대표현 금지).**

---

## 1. 개요 — weak 4 vs unsupervised 22

핵심 차이는 한 가지로 요약된다: **weak 4개는 anomaly label을 *학습에 직접 활용*한다.** unsupervised 22개는 label을 학습에 쓸 방법이 없다.

이 차이가 Q1/Q3 축의 의미를 다시 정의한다. 기존 22개 unsupervised에서 Q1(train에 anomaly 포함) vs Q3(`*_normalonly`, normal-only train)은 *비지도 방법이 anomaly label을 쓸 수 없으니, 학습 시 anomaly를 제외할지 여부*를 정하는 임시방편 축이었다. weak 4개는 anomaly label을 *각 방법론이 제안하는 방식대로* 활용하므로, **normal-only(Q3) 실험은 애초에 필요 없다** — weak는 labeled(anomaly 포함, full-train) 데이터에서 학습하는 것이 정상 동작이다. 따라서 weak 모델에 `*_normalonly`(Q3)를 넘기면 wrapper가 `RuntimeError`를 raise하는데, 이는 "한계/결함"이 아니라 **비지도 전용 조건(Q3)이 weak 방법론에 적용 대상이 아님(N/A)**을 뜻할 뿐이다. weak 실험에는 **labeled-train(=anomaly가 포함된 일반) experiment**를 쓴다.

| 모델 (CLI key) | 논문 | Venue | anomaly label 활용 방식 (핵심) |
|----------------|------|-------|--------------------------------|
| **`deepmil`** | DeepMIL (Sultani et al.) | CVPR 2018 (arXiv:1801.04264) | MIL ranking: positive/negative bag 쌍의 max-instance score hinge |
| **`wetas`** | WETAS — Weakly Sup. Temporal Anomaly Segmentation (Lee et al.) | ICCV 2021 | weak BCE + soft-DTW alignment 기반 weak segmentation |
| **`treemil`** | TreeMIL (Chen et al.) | ICASSP 2024 (arXiv:2401.11235) | N-ary-tree pyramid MIL → window-level BCE |
| **`nrdetector`** | NRdetector — Noise-Resilient PWAD with Weak Segment Labels (Wang et al.) | KDD 2025 (arXiv:2501.11959) | PU(Positive-Unlabeled) learning + reveal/label-propagation |

> 모든 모델의 공통 1차 가공: point label → **window/bag weak label = `max(train_y over window)`**, **train split에서만** 계산(leak-free). 세부 활용 방식은 §2.

레지스트리 격리: weak 4개는 `WEAK_SUPERVISED_MODELS`로 분기되며 `--model all`에 포함되지 않는다. 반드시 `--model <name>`으로 명시 호출한다(§4).

정규화 원칙(중요): weak 4개는 모두 자기 wrapper 안에서 정규화한다(§5.3). multi-entity 데이터셋(SMD·MSL·SMAP·Exathlon concat)에서는 **per-entity(=per source file) 정규화**가 적용된다 — 각 entity의 scaler를 그 entity의 **train slice에만 fit**하고 자기 test slice를 `.transform`한다(leak-free). 단일 entity 데이터셋(PSM/SWaT/WaDi/simulation, smd_simple 등)에서는 per-entity ≡ whole-array NO-OP라 이전과 bit-identical. 이 원칙은 **데이터셋 설정 방식과 무관하게 일관 적용**된다(§5.3에서 코드 경로 상술).

---

## 2. 방법론별 — anomaly label을 어떻게 활용하는가

각 방법은 point label을 window/bag weak label로 coarsen(`max`)한 뒤, 자신의 학습 목적에 쓴다.
아래는 **원본 논문/repo의 메커니즘**과 그것을 따른 **우리 wrapper의 활용**, 그리고 **fidelity 검증 결과**다.
원본 file:line은 그대로 인용한다(증거표준 G-A).

### 2.1 DeepMIL — MIL ranking (faithful=true)

- **원본 label 프로토콜.** video-level binary weak label(이상이 어딘가 포함된 video = positive bag, 없는 video = negative bag; PAPER §3.1 *"only video-level labels indicating the presence of an anomaly is needed"*). 각 video = 하나의 BAG을 `n_seg=32` non-overlapping segment(C3D fc6 4096-d instance)로 분할. MIL ranking은 positive bag과 negative bag을 **segment에 대한 MAX instance score**로 비교: hinge `l(Ba,Bn)=max(0, 1 - max_i f(Va_i) + max_i f(Vn_i))`, margin 1 (PAPER Eq.4-5). official code는 30×30 normal-vs-abnormal 모든 쌍에 대해 hinge를 합(all-pairs sum)한 뒤 `.mean()`, + abnormal half에만 smoothness/sparsity `8e-5` 각각, + Keras `l2(0.001)`. optimizer는 **FROZEN C3D feature 위의 3-layer FC head**에 Adagrad(lr=0.01). TS-canonical 형태는 WETAS(ICCV'21 p.7360, *"DeepMIL employs the same model architecture with WETAS (i.e., DiCNN)"*)가 정의 — video bag이 stream 위 고정 window가 되고 DiCNN(7 dilated layer, kernel 2, RF=2^7=128, d=128)이 dense하게 score.
- **우리 wrapper의 활용.** `run_weak_sota_baseline_with_epoch_eval`(`baseline_common.py:1323-1328`)가 train-split point label `train_y`를 `model.fit(train_X, train_y=...)`로 forward — unsupervised 경로 대비 **유일한 동작 차이**. BAG = StandardScaler-정규화 stream 위 길이 `seq_len`의 window 하나(segment-boundary-safe). 정규화는 공유 leak-free 커널 `_per_file_norm`을 통해 **source file(entity)마다 별도 StandardScaler를 그 file의 TRAIN slice에만 fit**하고(`wrapper.py:168` `fit_transform_train_per_file`), test는 같은 scaler로 `.transform`만 한다(`wrapper.py:295` `transform_test_per_file`, fit-on-test 금지). multi-entity는 per-entity, single-file은 whole-array NO-OP. bag weak label = `max(train_y[s:s+seq_len])`(`wrapper.py:155-157`, train split only → leak-free). `pos_idx`/`neg_idx`로 분리(`wrapper.py:158-159`), 매 MIL iteration마다 30 positive + 30 negative window를 복원추출 샘플. `mil_ranking_loss`(`model.py:180-206`)는 dense per-timestep sigmoid score를 **MAX-over-timesteps**로 reduce(`model.py:194-195`; segment-max의 dense analog) 후 paired hinge `clamp(margin - pos_max + neg_max, 0).mean()` + positive bag에만 smoothness/sparsity + L2.
- **fidelity.** **faithful=true** (head/loss는 Sultani clean-room). **주요 deviation:** (a) MIL reduction이 all-pairs sum이 아닌 paired-min(P,Nn) hinge — 동일 ranking semantics, score 영향 없음. (b) encoder가 official이 아니라 **WETAS DiCNN(DERIVATIVE_CITED)** — Sultani 원저엔 학습형 TS encoder가 없어 WETAS가 정의한 canonical을 차용. (c) optimizer = **Adam lr=1e-4**(WETAS encoder의 자체 optimizer): Sultani의 Adagrad 0.01은 frozen-C3D shallow head용이라 deep trainable DiCNN에서 발산(logits→-200/-440, score collapse)하므로 encoder source의 optimizer를 따름. head/loss는 그대로 Sultani. 실무 주의: deepmil은 weak 4개 중 초기(특히 epoch-1) score collapse 위험이 가장 큰 모델이라(deep DiCNN + MIL hinge), best-epoch=`pak_auc_f1` 자동 선택(§5.2)이 collapse한 초기 epoch을 걸러내는 안전장치 역할을 한다 — 단일 마지막-epoch 수치로 판단하지 말 것.

### 2.2 WETAS — weak BCE + soft-DTW alignment (faithful=true)

- **원본 label 프로토콜.** weakly supervised. window-level weak label = 각 non-overlapping `split_size` chunk에 대한 point label의 max-pool. official(@cb149dc): (1) per-recording StandardScaler z-score(`timeseries.py:38-39`), front zero-pad non-overlapping chunking(`timeseries.py:43,48`); (2) `dlabel=label`, `wlabel=torch.max(label,dim=1)[0]`(`timeseries.py:52-53`) — **학습 label-사용 메커니즘 전부**; (3) weak BCE `bce(wscore,wlabel)`(`train_classifier.py:127`); (4) DTW-alignment hinge: min-max-normalized actmap에서 `pos_seqlabel=get_seqlabel(actmap,wlabel)`, `neg_seqlabel=get_seqlabel(actmap,1-wlabel)`, `pos/neg_dist=softDTW(...)/split_size`, `loss=relu(beta+pos_dist-neg_dist)`(`model.py` dtw_loss); (5) total `= bce_loss + dtw_loss.mean(0)`(`train_classifier.py:129`), Adam(lr), SoftDTW(gamma, normalize=False). point label은 학습을 직접 supervise하지 않으며 — weak label을 max-coarsen하는 **원천**으로만 쓰인다.
- **우리 wrapper의 활용.** pipeline이 train-split point label `train_y`를 `model.fit(train_X, train_y=..., ...)`로 forward(`baseline_common.py:1325-1328`). wrapper가 weak label을 직접 coarsen(train split only, leak-free): `wlabels = stack([train_y[s:s+split_size].max() for s in starts])`(`wrapper.py:262-264`) — `timeseries.py:53`과 동일한 max 공식. loss 항별 동일: `bce(out['wscore'],batch_w)` + `model.dtw_loss(out['output'],batch_w).mean(0)`(`wrapper.py:221`, `model.py:153`), SoftDTW(gamma, normalize=False), BCELoss('mean'), Adam(lr). `get_seqlabel`/`dtw_loss`/`get_alignment`은 official `model.py:157-214`에서 **verbatim vendored**(device edit만; byte-equivalent 확인). 정규화는 wrapper가 raw 데이터를 받아 **source file(entity)별 StandardScaler를 그 file의 TRAIN slice에만 fit**하고(`wrapper.py:186-201`, 공유 `_per_file_norm` 커널), test는 같은 scaler로 `.transform`만 한다(이전의 transductive fit-on-test leak 제거 = 이번 fidelity fix). `timeseries.py:38-39`의 per-recording z-score와 일치, multi-entity는 per-entity / single-file은 whole-array NO-OP.
- **fidelity.** **faithful=true** (term-by-term 동일, alignment 코드 byte-equivalent). 추론 시 emit하는 연속 `dscore`(`model.py:153`)는 upstream의 `dauc`/`dauprc` ranking-score 입력과 정확히 동일하므로, **본 실험의 ranking metric(`pak_auc_f1`/ROC/PRC)에 대해 faithful**하다. WETAS 논문 headline인 DTW-aligned point-F1/IoU(binary `get_dpred` 기반 segmentation metric)는 **본 실험 metric suite에 없는 별개의 metric family**이므로 **의도적으로 산출하지 않는다** — 결함이 아니라 scope-out(필요 fix 아님). frozen metric layer에는 raw `dscore` 그대로 들어간다.

### 2.3 TreeMIL — pyramid MIL → window BCE (faithful=true)

- **원본 label 프로토콜.** weakly(instance-level) supervised. point label을 non-overlapping `split_size` window의 MAX로 coarsen: `wlabel = torch.max(label, dim=1)[0]`(`timeseries.py:71`), per-file StandardScaler z-score(`timeseries.py:53-55`) 이후. 모델은 N-ary-tree pyramid(CSCM + PAM attention)이고, 공유 scorenet이 bag score `wscore = sigmoid(scorenet(max-over-tree-nodes))`(`agg_type='max'`, `pooling_type='max'`)를 낸다. **유일한 active 학습 목적은 window-level BCE on `wscore`**: `loss = pyra.last_loss(out['wscore'], wlabel)`(`train.py:134-144`), `last_loss = nn.BCELoss('mean')`(`utils.py:30-32`). DTW/alignment loss는 유일 호출부에서 commented out(`train.py:31`) → default `python train.py` recipe는 **BCE-only**.
- **우리 wrapper의 활용.** 동일 프로토콜. `_make_train_windows`(`wrapper.py:157-180`): per-segment z-score(`wrapper.py:170`), non-overlapping `split_size` window(`wrapper.py:171-174`), `wlabel = float(y[s+a:s+b].max())`(`wrapper.py:175`) — `timeseries.py:71`과 동일 max-coarsen, train split only. loss: `loss = loss_fn.last_loss(out['wscore'], wlabel)`(`wrapper.py:221`), `last_loss=nn.BCELoss('mean')`, `wscore=sigmoid(scorenet(max-over-nodes))`(`agg_type/pooling_type='max'`), Adam lr=1e-4(`wrapper.py:210`). 정규화는 train에서 source file(entity)별 StandardScaler를 그 file의 TRAIN slice에만 fit해 캐시하고(`wrapper.py:211` `fit_transform_train_per_file`), test는 같은 per-file scaler로 `.transform`만 한다(`wrapper.py:331` `transform_test_per_file`) — 단일 파일 데이터셋에서도 test-side fit-on-test leak이 제거된 leak-free 경로(이번 fix). multi-entity는 per-entity.
- **fidelity.** **faithful=true.** **deviation(severity=none):** weak label을 train split에만 유도(원본은 valid/test에도 유도해 내부 threshold-F1 metric/early stopping에 사용) — valid/test wlabel은 원본 자체 metric에만 쓰였고 우리는 harness metric layer로 대체하므로 reported 수치에 영향 없음. 동일한 train-time label semantics.

### 2.4 NRdetector — PU learning + reveal (faithful=true)

- **원본 label 프로토콜.** WEAK SEGMENT label에 대한 PU(Positive-Unlabeled) learning — point label은 학습에 직접 쓰이지 않는다. (1) weak segment label = 각 non-overlapping `L=100` window의 max point-label(`data_loader.py:56`). (2) **REVEAL**: positive segment 중 **index 순서로 앞에서 `int(noisy_rate*#pos)`개**만 labeled-positive(`train_DP`)로 유지, 나머지는 unlabeled(`train_DU`)로 demote(`selector.py:31-38`, **positional first-N**, random 아님). `noisy_rate=0.4` = paper §5.1 *"40% of anomalous segment-level labels"*. (3) **PU-LP**: `calc_pu`(m=4, lmbda=0.32)가 `W=pinv(I-arg_a*A)-I` 유사도로 unlabeled를 ranking해 reliable-positive 확장, `calc_lp`(kNN k=5 label propagation)가 reliable negative `lp_n` 반환(`selector.py:91-240`). (4) `train_()`는 `(P, lp_n)` 반환 → classifier는 `RP = train_DP`(신뢰 labeled-P **만**) + `RN = lp_n`으로 학습하고, calc_pu가 확장한 RP는 train path에서 **버린다**(`selector.py:62-75`). (5) PU classifier(6-MLP, hidden=128) loss = `lamda_1*pu_loss + lamda*const_loss`(둘 다 1; `solver.py:115-116`), `pu_loss=LabelDistributionLoss(prior)`. class prior는 `create_loss(args.prior)`로 진입 — **released code는 argparse default `prior=0.25`(EMG-tuned)** 고정(추정 로직 없음). 단 PAPER 본문(§5.2, Fig.3a)은 *"employs class prior estimation"*이라 기술.
- **우리 wrapper의 활용.** 충실한 in-memory re-host. (1) `wlabel = torch.max(y,dim=1)[0]`(`wrapper.py:158`) — `data_loader.py:56` 정확 일치. (2) REVEAL을 TRAIN split에만, windowing 전 적용: `known=int(noisy_rate*#pos); DP=pos_idx[:known]; 나머지+모든 neg→unlabeled`(`wrapper.py:226-240`) — positional first-N(`selector.py:31-38`), leak-free. (3) `_pulp_select`(`wrapper.py:203-301`)가 directed-A + `W=pinv(I-a*A)-I` + `calc_pu(m=4,λ=0.32)` + `calc_lp(k=5)` re-host. (4) train 입력 = `RP=DP`(labeled-P만) + `RN=lp_n`, calc_pu RP는 버림(`train_()`의 `return P, lp_n` 일치). (5) PU loss = `pu_loss + const_loss`(둘 다 λ=1), `LabelDistributionLoss(prior)`(`model.py:225-284`). (정규화: source file별 StandardScaler를 TRAIN에만 fit, test는 transform — `wrapper.py:469,586`, leak-free)
- **우리 wrapper의 anomaly-score(추론) 경로.** test-time score = **연속(continuous) per-window min-max ACTMAP을 그대로 방출**(= upstream `get_dpred`의 `interested_instance`, `(h−min)/max`, `h=fc(out)`). 이것이 upstream의 **DEFAULT** 경로가 ranking하는 정확한 object다: `python main.py`(README)는 `--mode` argparse **default=`'train'`** → dispatch가 `solver.rank_test()`를 실행하고, `rank_test()`는 이 연속 ACTMAP을 `np.argsort`로 RANK한 뒤 상위 `anomaly_ratio`(default 0.65) 비율 + HOC training-free auto-rate re-threshold로 단일 label을 만든다. 우리 ranking harness(ROC/PRC/pak_auc_f1)는 자체 operating-point 선택을 하므로 연속 ACTMAP을 직접 받는 것이 faithful하며, upstream의 `anomaly_ratio=0.65`+HOC 단일-operating-point selector는 in-code 주석으로 남겨둔다. **(commit 5cff9da, 2026-06-03)** — binary mean-gate **제거**됨. ~~이전 `× BINARY segment gate`(`seg ≥ mean+anomaly_thre·(max−min)`) "2026-06-01 fidelity fix" 기술은 SUPERSEDED/WRONG~~: 그 mean-gate는 upstream의 **commented-out, NON-default `solver.test()`(`--mode test`) 분기**일 뿐이며 DEFAULT `rank_test()`가 아니다 — 비-flagged window를 0으로 하드-binarize해 일부 데이터셋에서 >80% exact-zero score를 만들어 연속 ranking을 붕괴시켰다. per-window min-max actmap 자체는 upstream-faithful이라 그대로 유지한다.
- **fidelity.** **faithful=true** (label-usage pipeline: weak segment label, positional first-40% reveal, RP=labeled-P/RN=lp_n train set 모두 정확). 추론 score는 **classifier-gated 연속 per-window min-max ACTMAP** = `actmap × [seg_prob ≥ mean(seg_prob)]`을 방출한다(2026-06-03 PM 복원; `NRDETECTOR_CORRECTION_v2.md`). upstream DEFAULT `python main.py`는 `solver.rank_test()`만 실행하고, 이것이 ranking하는 `interested_instance`(solver.py:219)는 `save_instance_files`가 **classifier가 flag한 window(`instance_label[i]>0`, solver.py:198-203)의 actmap만** 모은 것이므로 **classifier gate가 ranked pool을 결정 = faithful의 핵심**(우리 harness가 `anomaly_ratio=0.65`+HOC selector를 자체 thresholding으로 대체). ⚠ gate를 제거했던 commit `5cff9da`(2026-06-03 AM, no-gate continuous)는 **DOUBLE 회귀로 revert됨**: encoder가 Stage 0 후 frozen이라 gate 없는 actmap은 epoch마다 불변(bit-identical) + ranking 붕괴(SWaT pak 0.86→0.44, roc 0.88→0.37). 이전 "binary mean-gate=NON-default/WRONG" 기술은 **오판**. prior no-gate run score는 STALE → re-score 필요. **명시적 deviation:** `prior=None` → train wlabel rate로 추정 후 `clip[0.05,0.5]`(`wrapper.py:420-422`), code의 고정 0.25 대신. 이는 **PAPER가 기술한 "class prior estimation"에 부합**하며, multi-dataset 사용에 필수(0.25는 EMG 전용). soft-DTW alignment loss는 2026-06-01 **복원**됨 — `loss = bce + dtw_loss`(equal-weight, upstream `extractor.py:127-129`)이며 CPU numba `@jit` 경로(`use_cuda=False` 강제, `wrapper.py:26-32,113`)로 실행되어 `@cuda.jit` 의존 없이 동작(kernel은 `comparison.baselines.wetas.softdtw_cuda`와 byte-identical). (이전 "`@cuda.jit` 의존이라 DROPPED" 기술은 stale — GUIDE.md §8 nrdetector row와 일치.)

---

## 3. 실험 세팅 파라미터 — 무엇이고 / 무엇을 의미하며 / 무엇을 결정해야 하는가 (핵심)

label-사용 학습 코드는 이미 §2처럼 원본 충실하게 구현돼 있다. **사용자가 실험 세팅에서 다루는 것은 파라미터**다.
아래 표의 ground-truth는 default preset `_get_default_model_params()`(`baseline_common.py:324-369`)다.

**결정주체(Decision owner) 범례:**

| 태그 | 의미 |
|------|------|
| **USER** | 사용자가 실험 설계로 결정 — 실제로 정해야 하는 값(epoch, window, reveal fraction 등) |
| **ORIGINAL** | 원본 논문/official config 고정값 — 바꾸지 않는 것이 기본(분석 목적이 분명할 때만) |
| **DATA** | 데이터에서 런타임 추정 — 사용자가 직접 정하지 않음 |
| **IMPL** | impl-invented(official recipe 부재로 본 프로젝트가 정함) — confound로 명시 |

### 3.1 DeepMIL (`baseline_common.py:362-369`)

| 파라미터 | 의미 | 결정주체 | default | 원본값 |
|----------|------|----------|---------|--------|
| `seq_len` | bag(window) 길이 = MIL instance 수(dense per-timestep) | ORIGINAL | 128 | DiCNN RF=2^7=128 |
| `encoder_dim` | DiCNN hidden d | ORIGINAL | 128 | WETAS d=128 |
| `ranking_margin` | MIL hinge margin | ORIGINAL | 1.0 | Sultani Eq.4 margin 1 |
| `lambda_smooth` / `lambda_sparse` | smoothness/sparsity 정규화 계수 | ORIGINAL | 8e-5 / 8e-5 | Sultani 0.00008 |
| `l2_reg` | head L2 | ORIGINAL | 0.001 | Sultani Keras l2(0.001) |
| `bags_per_batch` | batch당 bag 수(=30 pos + 30 neg) | ORIGINAL | 60 | Sultani batchsize=60 |
| `dropout` | head dropout | ORIGINAL | 0.6 | Sultani head |
| `optimizer` / `lr` | optimizer / 학습률 | ORIGINAL(encoder-sourced) | adam / 1e-4 | WETAS encoder optimizer(§2.1) |
| `aggregation` / `test_stride` | test overlap window 집계 / stride | ORIGINAL | mean / 1 | TS 추론 집계 |
| `epochs` | 학습 epoch | **USER** | 10 | TS epoch 파이프라인(`--sota-epochs`로 override) |
| `iters_per_epoch` | epoch당 MIL iteration | **USER** | 50 | — |
| `train_stride` | train window stride | **USER** | 1 | — |
| `n_segments` | (vestigial) config back-compat | (무시) | 32 | 실제는 dense per-timestep |

### 3.2 WETAS (`baseline_common.py:324-329`)

| 파라미터 | 의미 | 결정주체 | default | 원본값 |
|----------|------|----------|---------|--------|
| `split_size` | weak label/segment chunk 길이(window) | **USER** | 500 | upstream split_size |
| `train_stride` | train chunk stride | **USER** | 500 | non-overlapping(==split_size) |
| `hidden_size` / `output_size` | DiCNN hidden / output(==hidden, fc quirk) | ORIGINAL | 128 / 128 | 128 |
| `kernel_size` / `n_layers` | dilated conv kernel / layer 수(RF=2^7) | ORIGINAL | 2 / 7 | 2 / 7 |
| `pooling_type` | actmap pooling | ORIGINAL | avg | — |
| `local_threshold` / `granularity` | get_seqlabel 이진화 threshold / coarsen | ORIGINAL | 0.3 / 4 | — |
| `beta` | DTW-alignment hinge margin | ORIGINAL | 0.1 | — |
| `gamma` | soft-DTW smoothing | ORIGINAL | 0.1 | SoftDTW gamma |
| `batch_size` / `lr` | batch / 학습률 | ORIGINAL | 32 / 1e-4 | — |
| `epochs` | 학습 epoch | **USER** | 200 | upstream=200(`--sota-epochs`로 override) |

### 3.3 TreeMIL (`baseline_common.py:331-336`)

| 파라미터 | 의미 | 결정주체 | default | 원본값 |
|----------|------|----------|---------|--------|
| `split_size` | weak label window 길이 | **USER** | 500 | upstream split_size |
| `train_stride` | train window stride | **USER** | 1 | — |
| `epochs` | 학습 epoch | **USER** | 200 | upstream(`--sota-epochs`로 override) |
| `batch_size` / `lr` | batch / 학습률 | ORIGINAL | 32 / 1e-4 | — |
| `ary_size` / `inner_size` | N-ary tree arity / inner size | ORIGINAL | 2 / 3 | pyramid |
| `d_model` / `d_k` / `d_v` / `d_inner_hid` | attention 차원 | ORIGINAL | 128/128/128/32 | — |
| `n_head` / `n_layer` / `dropout` | PAM head/layer/dropout | ORIGINAL | 5 / 2 / 0.5 | — |
| `agg_type` / `pooling_type` | tree-node 집계 / pooling | ORIGINAL | max / max | scorenet max |

### 3.4 NRdetector (`baseline_common.py:347-353`)

| 파라미터 | 의미 | 결정주체 | default | 원본값 |
|----------|------|----------|---------|--------|
| `win_size` | weak segment window 길이 | **USER** | 100 | L=100(pipeline 500 아님) |
| `noisy_rate` | **REVEAL fraction** — positive segment 중 앞에서 몇 %만 labeled-P로 공개할지(나머지 demote) | **USER** | 0.4 | paper §5.1 40% |
| `prior` | PU class prior(=anomaly 비율). None이면 train wlabel rate로 추정 후 `clip[0.05,0.5]` | **DATA** | None | code 고정 0.25(EMG); PAPER는 estimation |
| `knn_k` | calc_lp label propagation kNN k | ORIGINAL | 5 | k=5 |
| `classifier_hidden` | PU classifier MLP hidden | ORIGINAL | 128 | hidden=128 |
| `hidden_size`/`output_size`/`d_model` | encoder 차원 | ORIGINAL | 64/64/64 | — |
| `kernel_size` / `n_layers` | encoder dilated conv | ORIGINAL | 2 / 7 | — |
| `batch_size` / `lr` | classifier batch / 학습률 | ORIGINAL | 32 / 1e-5 | — |
| `epochs` | PU classifier epoch | **USER** | 200 | upstream(`--sota-epochs`로 override) |
| `seed` | random seed | ORIGINAL | 0 | — |
| `encoder_epochs` / `encoder_lr` | feature encoder 학습 schedule | **IMPL** | 50 / 1e-3 | official은 pretrained `.pth` 로드(학습 recipe 부재) — confound |

### 3.5 사용자가 실제로 결정해야 하는 값 (요약)

세팅 시 **명시적으로 정해야 하는** 것은 결정주체=**USER**(+ DATA/IMPL 인지) 항목이다:

- **공통 — `epochs`**: 4개 모두 best-epoch=`pak_auc_f1`로 per-epoch 평가하므로(§5), 충분히 길게 두고 best epoch을 자동 선택하는 것이 안전. CLI `--sota-epochs <N>`로 override(§4).
- **공통 — window 길이**: deepmil `seq_len`(default 128, DiCNN RF에 묶임 → 변경 비권장), wetas/treemil `split_size`(default 500), nrdetector `win_size`(default 100, NOT 500). weak label은 이 window의 `max`로 만들어지므로 window가 커질수록 weak label이 더 자주 1이 된다.
- **공통 — `train_stride`**: train window 겹침. deepmil/treemil/nrdetector=1(조밀), wetas=500(non-overlapping).
- **nrdetector `noisy_rate` (= REVEAL fraction)**: positive label 중 앞에서 40%만 labeled-P로 쓰고 나머지는 unlabeled로 demote하는 **실험 설계값**. 데이터 속성이 아니므로 추정 대상이 아니고, paper 세팅(0.4)을 따른다. 의도적으로 label noise 강건성을 보려면 사용자가 조정한다.
- **nrdetector `prior` (DATA)**: 사용자가 직접 정하지 않음 — None이면 데이터에서 추정. 특정 값으로 고정하고 싶을 때만 명시.
- **nrdetector `encoder_epochs`/`encoder_lr` (IMPL)**: official recipe가 없어 본 프로젝트가 정한 confound. 분석 목적이 분명할 때만 건드린다.

### 3.6 파라미터 override 방법

- **CLI:** epoch만 `--sota-epochs <N>` (weak는 `create_model`에서 `sota_epochs`를 읽어 `params['epochs']`를 덮어씀). 그 외 preset 키는 CLI 플래그가 없다.
- **preset 직접 수정:** `_get_default_model_params()`(`baseline_common.py:324-369`)의 해당 dict를 수정. wrapper가 모든 키를 `hparams.get(key, default)`로 읽으므로 preset 변경이 그대로 반영된다.
- **개별 인스턴스화:** 분석용으로 `WETASBaseline(**hparams)` 식 직접 생성도 가능.
- **`--normalize-mode`는 weak에 무의미**(무시됨) — weak는 self-normalizing(§5.3). 줘도 `[override] ... passing raw data (normalize_mode=none)` 로그만 남는다.

---

## 4. 실행

### 4.1 환경

```bash
conda activate dc_vis        # 필수(CLAUDE.md). pip install 금지.
```

- **GPU 필요.** 4개 모두 딥러닝 모델(DiCNN encoder 학습 등). wrapper는 device-agnostic이지만 CPU 학습은 비현실적으로 느리다.

### 4.2 단일 모델 실행

dispatch는 `run_baseline.py:446-465`(weak 분기 → `run_weak_sota_baseline_with_epoch_eval`). **반드시 labeled-train(anomaly 포함) experiment를 쓴다** — `*_normalonly`(Q3)가 아니다.

```bash
conda activate dc_vis
python comparison/run_baseline.py \
    --experiment <labeled_train_exp> \
    --model <deepmil|wetas|treemil|nrdetector> \
    --output-base comparison/results/experiments/<N>_<desc> \
    --eval-interval 1
```

구체 예(SWaT, deepmil):

```bash
python comparison/run_baseline.py \
    --experiment swat_a1a2 \
    --model deepmil \
    --output-base comparison/results/experiments/7_20260530_weak_ssl \
    --eval-interval 1
```

인자 의미:

- `--experiment`: **labeled-train(anomaly 포함) experiment**. `*_normalonly`(Q3, normal-only)는 weak 적용 대상이 아니다(§1). 사용 가능한 experiment 목록: `python comparison/run_baseline.py --list-experiments`.
- `--model`: weak key 하나. `all`에는 weak가 포함되지 않으므로(레지스트리 격리) 한 번에 하나씩 명시.
- `--output-base`: 결과는 `<output-base>/<results_dir_name>/<model>/`에 저장.
- `--eval-interval 1`: 매 epoch test 추론+평가(권장). best epoch 자동 선택에 유리.
- epoch override는 `--sota-epochs <N>` (weak는 `sota_epochs`를 읽음). `--neural-epochs`는 weak에 영향 없음.

이미 결과가 있으면 자동 SKIP(`--force`로 재실행).

### 4.3 weak queue 템플릿

weak 전용 queue config는 아직 없다. 큐로 돌리려면 아래 템플릿으로 `comparison/configs/`에 새 config를 작성한다. 한 entry당 **weak 모델 하나**, `experiment`는 **labeled-train**(normalonly 아님).

```json
{
  "description": "Weakly-supervised SSL baselines (labeled-train)",
  "experiments": [
    { "name": "swat_deepmil",    "experiment": "swat_a1a2",      "model": "deepmil",    "eval_interval": 1 },
    { "name": "swat_wetas",      "experiment": "swat_a1a2",      "model": "wetas",      "eval_interval": 1 },
    { "name": "swat_treemil",    "experiment": "swat_a1a2",      "model": "treemil",    "eval_interval": 1 },
    { "name": "swat_nrdetector", "experiment": "swat_a1a2",      "model": "nrdetector", "eval_interval": 1 },
    { "name": "psm_wetas",       "experiment": "psm",            "model": "wetas",      "eval_interval": 1 }
  ]
}
```

```bash
conda activate dc_vis
python comparison/run_baseline_queue.py \
    --queue comparison/configs/baseline_queue_weak.json \
    --desc weak_ssl
# 명령만 먼저 확인:
python comparison/run_baseline_queue.py --queue <config> --dry-run
```

queue runner는 experiment마다 `run_baseline.py`를 **fresh subprocess**로 호출하므로, 한 entry가 실패해도 다음으로 격리 진행된다.

### 4.4 Resume — 중단된 학습 이어서 진행 (2026-06-01 추가)

weak SSL 4개 모델 모두 **per-epoch 체크포인트 저장 + 재개 학습**을 지원한다. 사용자 의도: "50 epoch까지 학습 → 나중에 그 weight로 51-100 epoch 이어서 학습하고 싶을 때, 1 epoch부터 다시 기록되거나 별도 디렉토리에 저장되면 절대 안 된다 — 같은 결과 파일에 51 epoch부터 제대로 append 되어야 한다." 구현은 [B option (epoch-level dynamics 일관성, byte-level 비결정론은 허용)](#설계-결정-옵션-b-2026-06-01)으로 결정됨.

#### 4.4.1 체크포인트 위치 + 포맷

```
<output-base>/<dataset>/<model>/
└── checkpoints/
    ├── last.pt     # 매 epoch 끝나면 덮어쓰기 (atomic write via .tmp + os.replace)
    └── best.pt     # pak_auc_f1 갱신 시 last.pt에서 mirror (shutil.copy2)
```

체크포인트 내용 (4개 wrapper 공통):
- `epoch` — 마지막 완료 epoch 번호 (1-indexed)
- `target_epochs` — 그 run의 `--sota-epochs` 값
- `model_state_dict` + `optimizer_state_dict` — torch 표준 체크포인트
- `train_loss_history` — epoch 별 평균 loss 누적
- `wrapper_hparams` — 아키텍처 + 학습 dynamics knobs (resume 시 mismatch 검출용)
- `rng_state` — `{torch_cpu, torch_cuda_all, numpy_global, python}` 전역 RNG 캡처 (eval **이전** 시점에서 캡처되므로 결정론 보장 없음, B option 의도된 trade-off)
- `scaler_state` — StandardScaler `mean_/scale_/var_/n_samples_seen_` (deepmil/nrdetector — wetas는 fit()에서 deterministic하게 재생성)
- 모델별 추가:
  - **deepmil**: `rng_wrapper_specific.bag_rng_state` (numpy bag-sampling Generator state)
  - **nrdetector**: `encoder_state_dict` + `h_scores` + `RP` + `RN` + `prior` (Stage 0+1을 skip 가능하게 함)

코드: `comparison/baselines/_checkpoint.py` (atomic_save / load_checkpoint / capture_rng_state / restore_rng_state / standard_scaler_to_dict).

#### 4.4.2 Resume 호출 방법

**CLI flag**: `--resume`. 다른 flag와의 우선순위는 **`--force` > `--resume`** (둘 다 주면 `--force` win, 처음부터 재시작).

기본 사용 예 (deepmil PSM, 5 epoch까지 학습된 상태에서 10 epoch까지 이어서 학습):
```bash
conda activate dc_vis
python comparison/run_baseline.py \
    --experiment psm \
    --model deepmil \
    --sota-epochs 10 \           # 이어서 학습할 최종 target
    --output-base comparison/results/experiments/7_20260601_weak_ssl_50ep \
    --resume                     # ← last.pt 자동 detect
```

queue runner도 동일 — queue config 안의 `sota_epochs`가 target. runner에 `--resume` flag를 forwarding하면 모든 entry가 자동 detect:
```bash
python comparison/run_baseline_queue.py \
    --queue comparison/configs/baseline_queue_weak_ssl.json \
    --output-base comparison/results/experiments/7_20260601_weak_ssl_50ep \
    --resume
```

**Resume 의미 매트릭스**:
| 상황 | 동작 |
|---|---|
| `epoch_metrics.json` 없음 | from-scratch (fresh run) |
| `epoch_metrics.json` 있음 + `last.pt` 없음 | `[SKIP]` — 완료된 run으로 간주 |
| `epoch_metrics.json` 있음 + `last.pt`.epoch ≥ target_epochs | `[SKIP]` — 이미 target 도달 |
| `epoch_metrics.json` 있음 + `last.pt`.epoch < target_epochs | `[RESUME]` from `last.pt.epoch + 1` → target |
| non-SSL 모델 + `--resume` | `[SKIP]` — 명시적 메시지 "resume not applicable: only weak SSL models support resume" |
| hparam mismatch (wrapper_hparams의 어떤 키든 다름) | `RuntimeError` — `--force`로 재시작하라는 에러 메시지 |

#### 4.4.3 Resume 후 출력 파일 일관성 보장

| 파일 | 동작 |
|---|---|
| `epoch_metrics.json` | **append** (매 epoch atomic save). resume 시 기존 epoch entries 보존 + 새 epoch append. 동일 epoch 중복 방지 (last.pt.epoch 기준으로 ghost entries truncate). |
| `epoch_scores/epoch_NNN_scores.npz` | resume 시 기존 npz 유지, 새 epoch부터 추가만 |
| `scores.npz` | 학습 종료 시 best epoch의 npz로부터 자동 복사 (resume 후 새 best가 나오면 갱신됨) |
| `model/{model.pt, config.json}` | 학습 종료 시 최종 state로 덮어쓰기 (= last epoch의 모델, NOT best — best.pt는 별도 checkpoints/에) |
| `visualization/epoch_metrics/*.png` | 학습 종료 시 epoch_metrics.json 전체 데이터로 재생성 → 자동 누적 (1..target_epochs 전부 plot) |
| `visualization/best_model/*.png` | 학습 종료 시 scores.npz + best epoch threshold 기반 재생성 |

#### 4.4.4 안전성 (검증된 시나리오, 2026-06-01)

다음 6개 시나리오를 dry-run으로 검증함:
1. **deepmil PSM 5+5ep** → `epoch_metrics.json` 1..10 연속, `epoch_scores/` 10 파일, `last.pt.epoch=10` ✓
2. **nrdetector PSM 5+5ep** → 3-stage 중 Stage 0+1 skip + Stage 2 ep6-10 정상 진행 ✓
3. **dropout mismatch** (수동 mutation) → `RuntimeError: DeepMIL resume hparam mismatch on 'dropout'` 명시적 fail ✓
4. **non-SSL (catch) + `--resume`** → 명시적 SKIP 메시지 + `checkpoints/` 디렉토리 생성 X ✓
5. **시각화 재생성** → resume 후 dashboard 파일 크기 변경 (5ep 180KB → 10ep 183KB) ✓
6. **epoch_metrics.json atomic write** → `.tmp` + `os.replace` (`baseline_common.save_epoch_metrics`) ✓

#### 4.4.5 설계 결정: 옵션 B (2026-06-01)

| 항목 | 옵션 A (byte-level) | **옵션 B (선택됨)** |
|---|---|---|
| RNG 캡처/복원 | 모든 RNG + `cudnn.deterministic=True` + `use_deterministic_algorithms` | 모든 RNG만 (deterministic ops 강제 안 함) |
| 학습 속도 | cuDNN deterministic backward 1.2~2× 느림 | 영향 없음 |
| 검증 가능성 | `np.array_equal(scores_fresh, scores_resume)` | epoch-level dynamics 비교만 |
| Risk | wetas SoftDTW custom CUDA kernel 호환 불확실 | risk 0 |
| 결정 사유 | — | 사용자 결정 ("byte-level 불필요, 학습 시간 1.5× 부담") |

코드 인용: `comparison/baselines/_checkpoint.py:capture_rng_state` docstring 참조 (pre-eval snapshot semantic 설명).

---

## 5. 출력 / 평가

### 5.1 결과 디렉토리

`<output-base>/<results_dir_name>/<model>/` — 기존 SOTA와 동일 레이아웃:

```text
<model>/
├── scores.npz          # key 'anomaly_score' = BEST epoch의 per-timestep 점수 (float32, len==len(test_X))
├── epoch_metrics.json  # { "eval_interval": N, "epochs": [ {epoch별 전체 metric}, ... ] }
├── epoch_scores/       # epoch_{NNN}_scores.npz (평가한 epoch마다)
├── metadata.json       # model_name, experiment, timing(train/inference)
├── model/              # 저장된 weight (wrapper.save 지원 시)
└── visualization/      # epoch_metrics plot + best_model PRC curve + threshold timeline
```

- `scores.npz`는 **best epoch**(pak_auc_f1 최대)의 점수와 일치(`baseline_common.py:1333-1345`).
- 점수는 연속값, 높을수록 더 이상. `np.nan_to_num`으로 NaN→0.

### 5.2 metric / best-epoch

- 모든 metric은 **frozen pipeline metric layer**(`mae_anomaly/evaluator.py` 경유 `compute_all_metrics`)로 계산 → unsupervised 22개와 동일 metric set이라 **직접 비교 가능**.
- per-epoch에 ROC/PRC-AUC, F1, F1_T, PA%K(0..100), PAK_AUC, VUS, affiliation/range-F1, AR-threshold variant 전부 기록.
- **best epoch 기준 = `pak_auc_f1`**(`run_weak_sota_baseline_with_epoch_eval` best_idx 선택, `baseline_common.py:1335-1338`; 프로젝트 전역 규칙). PRC-AUC는 보조 로깅.
- MAE 전용 키(teacher_*, disc_snr 등)는 baseline이므로 `null`.

### 5.3 정규화(참고)

weak 4개는 `SELF_NORMALIZING_WEAK`(`run_baseline.py:247`)로 분류되어 run_baseline가 **raw 데이터를 전달**(`normalize_mode='none'`)하고, 각 wrapper가 **원논문 정규화를 스스로 적용**한다(deepmil/wetas/treemil: per-recording StandardScaler; nrdetector: z-score). 따라서 `--normalize-mode`는 무시된다.

**per-entity 정규화(중요).** 4개 모두 공유 leak-free 커널 `comparison/baselines/_per_file_norm.py`를 쓴다: harness가 `loader.get_file_norm_segments()`로 per-entity `(train_len, test_len)` 구간을 넘기면(`run_baseline.py:487,495-496,513,521-522`), wrapper는 **source file(entity)마다 별도 scaler를 그 entity의 TRAIN slice에만 fit**(`fit_transform_train_per_file`)하고 자기 TEST slice를 `.transform`만 한다(`transform_test_per_file`, **fit-on-test 금지**). 이로써 wetas/treemil/deepmil/nrdetector test 경로에 있던 4건의 transductive fit-on-test leak이 제거됐다.

**THE PRINCIPLE.** per-entity 정규화는 **데이터셋 설정 방식과 무관하게 일관 적용**된다: multi-entity(SMD·MSL·SMAP·Exathlon concat)는 entity별로, 단일 entity(PSM/SWaT/WaDi/simulation, smd_simple 등)는 per-entity ≡ whole-array NO-OP라 이전과 bit-identical. (대조: 6개 untouchable SOTA — timesnet/tfmae/memto/moderntcn/dcdetector/catch — 는 upstream 자체가 whole-array single-scaler라 per-entity를 적용하지 않는다. weak 4개는 그 그룹이 아니다.)

---

## 6. 충실도 검증 결과 요약

원본 repo file:line / 논문 절 대조(증거표준 G-A, docstring/추측 배제). **4개 모두 label-usage 학습 pipeline이 faithful=true.**

| 모델 | faithful | 주요 deviation / consequence |
|------|----------|------------------------------|
| **deepmil** | **true** | MIL reduction이 all-pairs sum→paired hinge(동일 semantics, 영향 없음); encoder=WETAS DiCNN(DERIVATIVE_CITED, official 아님 — Sultani 원저에 TS encoder 부재); optimizer=Adam 1e-4(Adagrad 0.01은 deep DiCNN 발산). head/loss는 Sultani FAITHFUL. |
| **wetas** | **true** | term-by-term 동일, `get_seqlabel`/`dtw_loss`/`get_alignment` byte-equivalent vendored. 추론 시 binary `get_dpred` 대신 연속 `dscore` 사용(metric layer raw signal). |
| **treemil** | **true** | weak label을 train split에만 유도(원본은 valid/test에도 — 그 내부 metric에만 쓰여 영향 없음, severity=none). DTW 항은 원본도 dead code(BCE-only active). model selection은 valid early stopping 대신 best-epoch by pak_auc_f1. |
| **nrdetector** | **true** | PU label-usage(weak segment label, positional first-40% reveal, RP=labeled-P/RN=lp_n) 정확 일치. `prior` 추정(None→clip[0.05,0.5])은 code의 고정 0.25 대신 — PAPER의 "class prior estimation"에 부합, multi-dataset에 필요. soft-DTW alignment loss는 2026-06-01 복원(CPU numba `@jit`, `use_cuda=False`; `loss = bce + dtw_loss`, upstream `extractor.py:127-129`) — `@cuda.jit` 미의존, GUIDE.md §8 row와 일치. |

provenance gate: 이번 porting은 G1–G5 provenance gate(provenance label 정합 / stale label 제거 / source-chain documented / 외부 의존 분기 DROPPED 명시 / gitignore 권고)를 통과했고, label-usage pipeline 충실도는 file:line 대조로 재검증됐다. 상세 provenance는 phase2 source-correctness 리포트 참조.

---

> 변경 범위 요약: 4개 wrapper(deepmil/wetas/treemil/nrdetector) + deepmil model.py + 공유 2개
> (`run_baseline.py` SELF_NORMALIZING_WEAK routing + weak dispatch, `baseline_common.py` weak preset + `run_weak_sota_baseline_with_epoch_eval`).
> fit/predict 시그니처·(N_test,) float32 output·registry(WEAK=4)는 불변. **GPU 미실행 / 성능 수치 없음.**
