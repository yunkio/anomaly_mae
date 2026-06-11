---
phase: 8
agent: spec-reviewer
directives: [R3]
last_modified: 2026-06-11
review_target: paper/08_final_audit/NOTION_PLACEHOLDER_SPECS.md
canon_basis: |
  paper/05_manuscript/PLACEHOLDER_REGISTRY.md (v3-r1),
  paper/07_latex/sections/*.tex + main.tex (캡션·stub 원문),
  paper/01_research_understanding/EXPERIMENT_PROTOCOL_TRUTH.md (r4),
  paper/01_research_understanding/271_CONFIG_TRUTH.md (r4),
  paper/00_admin/DECISION_LOG.md (D-014),
  코드(read-only): mae_anomaly/{scoring,loss,trainer,config}.py, datasets/{noisy,loaders}.py,
  scripts/run_base_experiments.py, comparison/{baseline_common,experiment_configs,run_baseline_queue}.py,
  configs/queue_dedup_renumbered_v5.json, results/experiments/ + comparison/results/experiments/ 실측 (2026-06-11)
verdict: 조건부 통과 (REVISE) — BLOCKER 1건(D-014(b) 미등재) + MAJOR 1건 정정 후 발행 가능
---

# P8 NOTION_PLACEHOLDER_SPECS 검수 r1 — 독립 리뷰

마스터 규정 "한국어 품질, 실행 가능성, REGISTRY 전수 일치"에 대한 적대적 검수.
모든 판정은 정본 문서·tex 원문·코드·실험 폴더 **직접 실측**에 근거한다 (추정 0건).

---

## 1. REGISTRY 전수 일치 — **PASS** (ID 단위 기계 대조)

| 분류 | REGISTRY v3-r1 | 명세 존재 여부 | 판정 |
|---|---|---|---|
| FIG | FIG-1/2/3/4 + FIG-B1 = 5 | §1.1–1.5 전부 존재 | **5/5 ✓** |
| body TAB | TAB-1/2/3 + TAB-4(흡수) = 4 | §2.1–2.4 (TAB-4는 흡수 audit 절로 존재) | **4/4 ✓** |
| appendix TAB | TAB-A3/A6/A7/A8 + TAB-B1/B2/B3/B4 = 8 | §3 전부 존재 | **8/8 ✓** |
| 부분 placeholder | Table A.4 SMD 셀 (registry §7.2) | §3 "Table A.4" 절 존재 (TAB-1과 동일 소스 명시) | **✓** |
| ALG | ALG-C1 = 1 | §4 존재 | **1/1 ✓** |
| NUM | 001–031 = 31 | §5 그룹 8개에 전수 배정 | **31/31 ✓** (아래 검산) |
| TXT | TXT-001 ×1 + TXT-002 ×3 = 2종 4개소 | §6 (위치 file:line까지 명시) | **✓** |

- **NUM 그룹 합산 검산**: N-A{001,003,004,029}=4 + N-B{002,005,030}=3 + N-C{006–013}=8 + N-D{014–019}=6 + N-E{020–025}=6 + N-F{026,027}=2 + N-G{028}=1 + N-H{031}=1 = **31 ✓**. 합집합 001–031 연속·중복 없음 — 누락 0, 이중 배정 0.
- **tex 측 교차 검증**: `paper/07_latex/` 전체 `PH:NUM-*` 마커 grep — **고유 31종** 실측 (NUM-001/002/003 ×3개소, NUM-014/028 ×2개소 포함). REGISTRY·명세·tex 삼자 일치.
- **TXT 위치 실측**: `main.tex:110` ([URL]) · `appendix_A.tex:80` ([GPU model]) · `appendix_A.tex:81` ([URL]) · `sec5_conclusion.tex:31` ([URL]) — 명세의 file:line 포인터 전부 정확.
- TAB 총계 = body 3 + 흡수 1 + appendix 8 = **12 (흡수 포함) ✓**.

**누락 BLOCKER 없음.**

---

## 2. 실행 가능성 — 재사용 판정 3건 + 신규 실행 11건 + 무작위 6건

### 2.1 재사용 판정 3건 — **전부 PASS** (results/experiments 직접 대조)

폴더 실재 + PSM entity `experiment_metadata.json.config`를 271canon과 **전 키 diff** 수행:

| 실험 | 폴더 | 271canon 대비 config diff (전수) | 단독 diff 판정 |
|---|---|---|---|
| exp287_unmask | `287_20260603_132835_unmask` (37 entity meta) | `force_mask_anomaly: True→False` **만** | ✓ (TAB-3 행3 적합) |
| exp285_no_fm | `285_20260602_212000_no_fm` (37 entity meta) | `use_feature_matching: True→False` **만** | ✓ (TAB-B4 w/o FM 적합) |
| exp298 / exp299 | `298_20260610_234021_ep300_warm150` (36) / `299_20260611_052843_ep200_warm100` (37) | `num_epochs 500→300/200` + `teacher_only_warmup_epochs 250→150/100` **만** | ✓ (TAB-B2 축소 budget 적합; warmup 비례 1/2 보존 — 명세 ⑤의 "비례 축소" 요건 충족) |

- 명세의 "tex stub 열 라벨 100 epochs ↔ exp299 200ep" 결정 필요 지적도 실측 일치: `appendix_B.tex:65` CSMAD 행이 정확히 "100 epochs" 열에 [X.XX]를 둠.
- 참고 (OBS-2): exp287의 큐 `config_override`에 `force_mask_anomaly=True … force_mask_anomaly=False` **키 중복** (last-wins로 net False). metadata 실측으로 단독 diff는 확정이나, 큐 항목 재사용 시 혼동 소지 — 신규 큐 작성 시 중복 키 제거 권장.

### 2.2 신규 실행 11건 (§7.3) — 10건 PASS / 1건 MAJOR

| # | 실험 | 검증 결과 |
|---|---|---|
| 1 | baseline SMD/SMAP/MSL 재실행 (normalonly) | PASS — `comparison/run_baseline_queue.py` 실재(606줄), `experiment_configs.py`에 normalonly variant 등록(smd_concat/smap/msl 포함) 확인. 단 F-4 (MINOR) 참조: [CMP-Q3] 폴더 서술 부정확 |
| 2 | weak 4종 GPU (Q1) | PASS — `experiment_configs.py:36-40` "weak = Q1-ONLY, Q3에선 RuntimeError" 명시; `baseline_common.py` weak preset epochs=50 (":333,337,355,367,384"), eval_interval=1 기본(:943) — 명세의 "50 epochs, eval 매 epoch"와 일치 |
| 3 | standard clean-train split | PASS — loaders.py에 `standard` variant 부재 실측(grep 0건, 신규 loader 필요 판단 정확); tex stub 행(CSMAD (clean)/Baseline A/B, `sec4_experiments.tex:277-279`)·채움 열(SWaT excl22·WaDi A1·PSM만) 일치; **use_grl=True 유지 지침의 자가 비활성 근거 실증** — `loss.py:294` `if _pos_count == 0:` → GRL 손실 skip |
| 4 | ablation 행2 w/o GRL | **PASS — dead-component 함정 회피 지침 코드 검증 완료**. 함정 실재: `config.py:123` `grl_disable_anomaly_loss=True` + `loss.py:259-261` `if self.use_grl and self.grl_disable_anomaly_loss: anomaly_loss=0` → **use_grl=False 단독이면 게이트가 풀려 dynamic-margin anomaly loss(maximize 방향) 재활성**. 처방 유효: `anomaly_loss_weight=0.0`이 loss.py:265/272/404에서 곱해져 하드 제로. exp290은 nofm+nogrl 복합(큐 실측 `use_feature_matching=False use_grl=False`) — 행2 정의 불일치 판단도 정확 |
| 5 | symmetric decoder | PASS — `num_teacher_decoder_layers` config 키 실재; 271canon=3 (metadata 실측) → 2로 변경은 단독 diff; depth-2 행과 run 공유 설계 합리적 |
| 6 | ablation 행4 w/o OD | **MAJOR — F-2 참조.** `use_output_discrepancy=False` 키는 실재하나, 명세의 전제("OD 학습 제거 후에도 추론 score는 disc 성분을 포함")가 **코드와 반대** |
| 7 | label sparsity sweep | **PASS — `label_keep_ratio` 신설 제안이 기존 메커니즘과 정합.** ① 전용 파라미터 부재 실측 확인(`label_ratio/label_keep_ratio/sparsity` grep — mae_anomaly/·run_base_experiments.py 0건; PROTOCOL_TRUTH §⑦ r4와 일치) ② `noisy.py:52` `use_noisy_labels=(split=='train')` 실재 — 학습 한정 주입 구조 정확 ③ `run_base_experiments.py:397` `apply_normal50_noise` — region 단위 무작위 선택(seed=123)·미선택 region 라벨만 0(데이터 보존) 실측, p-일반화 자연스러움 ④ 라벨 경로 일괄 제어 주장은 PROTOCOL_TRUTH §⑦과 동일 ⑤ p=1.0 = [271c] 재사용은 비트 동일 조건이므로 타당 ⑥ "배치 positive 부재 시 GRL 미계산" 인용(loss.py:293-302)은 :294 실측 일치, §4.4 tex 문단(`sec4_experiments.tex:402-408`)도 실재 |
| 8 | contaminated 22종 (Q1) | PASS — experiment_configs.py에 non-normalonly(Q1) 항목 실재(:358, :404 등); Δ가 TAB-2 완성 의존이라는 순서 제약 명시도 정확 |
| 9 | w/o warmup / depth 1 | PASS — `teacher_only_warmup_epochs=0` 시 λ_rev ramp 분모 `num_epochs−warmup` (trainer.py:1205 `_student_total = max(num_epochs − _student_start, 1)`) → epoch 0부터 ramp 개시 — 명세의 "의도된 변형" 서술이 코드와 정확히 일치 |
| 10 | epoch-budget 50/100 | PASS — baseline epochs는 MODEL_CONFIGS 단위로 override 가능(unsup 10 통일 주석 ":272,279,286…" 실측); best-epoch 구조 동일 요건은 PROTOCOL_TRUTH §④-③ⓐⓑ와 정합 |
| 11 | masking ratio sweep | PASS — 큐 v5 전 32항목 `masking_ratio` override **0건** 실측(신규 등재 필요 판단 정확); \|M\|=round(50×ρ) (model.py:986; 0.05→2, 0.30→15) 일치. 단 F-5 (MINOR): "295–303은 window/patch sweep" 범위 서술 부정확 |

### 2.3 무작위 spot 6건 (재사용/측정 계열) — **전부 PASS**

| 항목 | 검증 결과 |
|---|---|
| FIG-4 | [271c] PSM `metrics.anomaly_ratio_threshold = 0.0017441…` — 명세 예시값 0.001744 일치. 산식 `score = recon + scaled_disc/4.0` = scoring.py:241-256 실측 일치. "excl22 후 13개 사건" = PROTOCOL_TRUTH §⑥ "test 14개 region 중 첫째 제외" → 13 ✓ |
| FIG-B1 좌 (c sweep) | c(`score_recon_disc_ratio`)는 scoring.py:247에서만 소비 — 추론 전용 주장 정확, 재학습 불필요 타당. "main best epoch 고정" 지침은 c별 재선정의 test-set selection 오염을 막는 올바른 설계 |
| TAB-1 / Table A.4 | 6 family 실값 전부 PROTOCOL_TRUTH §① 표와 숫자 단위 일치 (SWaT 719,959/224,960/45/1.63/19.05·3.68 등 전 셀 대조). SMD 분할 산식 인용 `loaders.py:1152-1157` — `test_split = len(test_data)//2` 실측 위치 정확 (정식 경로 `mae_anomaly/datasets/loaders.py`) |
| TAB-A3 | `baseline_common.py` MODEL_CONFIGS 실재; batch 실측 집합 {32,50,64,100,128,256,512} — "32–512 범위" 일치; 26 = 22+4 정합 |
| TAB-A7 | [271c] metadata에 `pak_auc_prc_auc`/`vus_roc`/`affiliation_f1_ar`/`pa_0_f1` 전부 실재(153키), **`pa_0_f1_ar` 부재 실측** — 키 혼동 경고 정확; `compute_full_metric_set` 실재(evaluator.py:864) |
| TAB-B3 / NUM-031 | `timing.inference_time` metadata 실재 — wall-clock 교차 검증 지침 실행 가능; `gpu_total`은 시간 합계 필드(모델명 아님) → TXT-001의 "metadata에 GPU 모델 필드 없음" 주장도 실측 일치 |

추가 교차 실측: 271canon 완주 entity = SMD 22/SMAP 5/MSL 5 (+SWaT 2, WaDi 2, PSM 1 = 37 meta) → 잔여 **SMD 6/SMAP 49/MSL 22** — 명세 [완주 대기] 수치와 정확히 일치. SWaT excl22 `timing.best_epoch_metric='excl22_pak_auc_f1'`(best_epoch 315) 실측 — TAB-A6 독립 best-epoch 서술 일치. full 0.944 vs excl22 0.629 인용도 PROTOCOL_TRUTH §⑥ 실측값과 일치.

---

## 3. 캡션 정합 (.tex 확정본 대조) — spot 15건 중 14건 일치, 1건 불일치

대조: FIG-1(sec1_intro:48) ✓ / FIG-2(sec3_method:69) ✓ / FIG-3(sec4:444) ✓ / FIG-4(sec4:489 — "regardless of event type", tex 우선 채택 정확) ✓ / TAB-1(sec4:33, dagger·(\%) 포함) ✓ / TAB-2(sec4:208) ✓ / TAB-3(sec4:344) ✓ / TAB-A3(appendix_A:107) ✓ / TAB-A6(:320) ✓ / TAB-A7(:347) ✓ / TAB-A8(:375) ✓ / TAB-B1(appendix_B:23) ✓ / TAB-B2(:57) ✓ / TAB-B3(:89) ✓ / FIG-B1(:124) ✓ — 전부 글자 단위 일치.

**불일치 1건**: TAB-B4 (appendix_B:155) — F-3 참조.

부수 확인: TAB-B1 명세의 "캡션은 F1+VUS-PR 약속 vs 표 stub은 F1/Δ만" 지적 — tex 실측(stub 열 {SWaT excl22, PSM, SMD avg}×2)으로 **사실 확인됨**, 정합화 지침 타당. ALG-C1 현 캡션 "(pseudocode placeholder)" (appendix_C:119) 실측 일치.

---

## 4. 한국어 품질 — **PASS**

- 모호 지시어("적절히/적당히/필요에 따라/알아서/등등") grep **0건**. 모든 실행 지침이 config 키=값·file:line·폴더명 수준으로 구체적.
- 번역투 어색 문장 미발견 — 라벨 체계(`[재사용]`/`[완주 대기]` 등)와 5요소 구조가 일관 적용되어 가독성 높음.
- 사소한 표기 일관성: §0 약칭([271c]/[CMP-Q3]) 정의 후 본문 전체에서 일관 사용 ✓.

## 5. R3 톤 — **PASS**

- "실험 데이터 부족 = 한계" 식 서술 **0건** ("한계/부족/limitation" grep 0건). 미실행 실험은 전부 "실행 지침 + 채워질 placeholder"로 서술 — D-014 ①의 R18/R3 정책과 일치.
- 부분 게재 금지·fallback 규칙(sync 그룹 B, TAB-2 ⑤ⓒ)도 한계 인정이 아닌 절차로 기술됨.

## 6. D-014 (b) 보강 확인 — **FAIL (BLOCKER, F-1)**

DECISION_LOG D-014 ②(b): "GRL 기계적 증거(probing classifier 분석)를 **권고 실험**으로 NOTION_PLACEHOLDER_SPECS에 추가 (원고 무변경 — rebuttal 대비)".
실측: 명세 전문에서 `probing/probe/권고 실험` grep **0건** — 미등재. (참고: D-014 (a) epoch 비대칭 공개는 Appendix B.2 보강 건으로 원고 측 사항 — 본 명세 범위 밖, 미적용이 결격 아님.)

**추가 권고안 (검수자 제안 — 명세 작성자가 반영할 것; 본 리뷰는 명세를 수정하지 않음):**

> ### 권고 실험 R-PROBE — GRL 억제의 기계적 증거 (probing classifier) `[신규 측정]` (D-014 (b))
> - **목적**: rebuttal 대비 — GRL이 Student 표현에서 anomaly-identity 정보를 실제로 억제했다는 직접 증거. 원고 무변경, Notion 명세에만 등재.
> - **절차**: [271c] 대표 entity(권장: TAB-3 대표 데이터셋과 동일) best checkpoint를 동결하고, test 윈도에 대해 ① Student decoder **final-layer hidden (output projection 직전 — GRL 부착 지점과 동일, FIG-2 ③ⓒ)**과 ② Teacher 동일 위치 hidden을 추출. 각 표현 위에 소형 probe(LayerNorm + Linear 1층, GRL head와 유사 용량)를 anomaly window 분류로 학습(표현은 frozen, probe만 학습) → probe AUC 비교. 기대: Student probe AUC ≪ Teacher probe AUC (억제 성공의 정량 증거).
> - **확장(선택)**: TAB-3 행2(w/o GRL) run 완료 후 동일 probing을 적용해 GRL 유/무 Student probe AUC 차이를 병기 — "GRL이 없으면 Student에 anomaly 정보가 잔존"의 대조군. (exp290은 no_fm 복합이므로 대조군으로 쓸 경우 각주 필수.)
> - **분류**: 학습 불필요 probe만 학습 → `[신규 측정]` 등급; §7.4 표에 1행 추가. 산출물은 본문 placeholder와 무관(원고 무변경) — Notion 페이지 '권고 실험' 하위 절로 발행.

---

## 발견 종합 (severity)

| ID | Severity | 위치 | 내용 |
|---|---|---|---|
| **F-1** | **BLOCKER** | 문서 전체 (§7 포함) | **D-014 (b) GRL probing classifier 권고 실험 미등재** — 마스터 결정 위반. 위 §6 권고안으로 등재 후 발행할 것 |
| **F-2** | **MAJOR** | §2 TAB-3 행4 / §7.3 #6 | **코드 사실 오류**: "OD 학습 제거 후에도 추론 score는 disc 성분을 포함하므로(adaptive 식)" — 실제로는 `scoring.py:106-107` `resolve_score_weights`가 `use_output_discrepancy=False`면 **w_disc를 0으로 강제** → `scoring.py:250-253`에서 `student_error=0`, **score는 자동으로 recon-only**. "disc 포함 유지 vs recon-only 사전 확정·각주 명시" 결론은 유지하되, 전제를 "코드 기본 동작 = 자동 recon-only (resolve_score_weights가 w_disc=0 강제); disc 포함을 원하면 별도 채점 경로 필요"로 정정할 것 — 현 서술대로면 실행자가 기본 동작을 반대로 알고 잘못된 각주를 달 위험 |
| **F-3** | MINOR | §3 TAB-B4 ④ | 캡션 전사 불일치: 명세 "(Teacher 2L\,/\,2L)" vs tex 원문(appendix_B:157) "(Teacher 2L\,/\,Student 2L)" — 명세 자체 규칙("캡션은 tex 원문 그대로") 위반. registry v2-r2 구문을 복사한 흔적; tex 기준으로 교체 |
| **F-4** | MINOR | §0 [CMP-Q3] / §2 TAB-2 ② 2 | "[CMP-Q3]에서 SMD/SMAP/MSL이 STALE" 서술 부정확 — 실측: `6_20260526_*` 폴더에는 **PSM/SWaT/WaDi만 존재**, SMD normalonly는 구버전 `3_20260312_*`뿐, **SMAP/MSL normalonly baseline은 어느 폴더에도 부재** (재사용 불가가 아니라 미존재). 실행 결론(전 entity 재실행)은 불변이나, "STALE 재실행"이 아닌 "미실행분 신규 실행 + SMD 구버전 폐기"가 정확한 서술 |
| **F-5** | MINOR | §1 FIG-B1 ⑤ⓒ | "기존 큐 295–303은 window/patch 크기 sweep" — 범위에 297(dyn_dmodel)·298/299(epoch budget) 포함되어 부정확. 핵심 결론(큐 32항목 전수에 masking_ratio override 0건 — 신규 등재 필요)은 실측으로 유효 |
| OBS-1 | 관찰 | §4 ALG-C1 ④ | τ=clip((e−250)/250) 식은 의사코드의 **1-based e 규약에서만** 코드(trainer.py:1207, 0-based `(epoch−250+1)/250`)와 일치 — item 3의 규약 명시 지침이 item 4의 식에도 적용됨을 한 줄 연동 표기하면 off-by-one 재발 위험 제거 |
| OBS-2 | 관찰 | 재사용 exp287 | 큐 config_override에 `force_mask_anomaly` 키 중복(True→False, last-wins) — net 단독 diff는 metadata로 확정; 신규 큐 항목 작성 시 중복 키 패턴 답습 금지 |

**통계**: BLOCKER 1 / MAJOR 1 / MINOR 3 / 관찰 2. 검증 항목: REGISTRY ID 대조 52건(FIG 5, TAB 12, ALG 1, NUM 31, TXT 2종+개소 검증), 캡션 spot 15건(14 일치), 재사용 판정 3건(3 PASS, config 전 키 diff), 신규 실행 11건(10 PASS / 1 MAJOR), 무작위 spot 6건(6 PASS), 코드 사실 인용 20여 건 file:line 실측.

---

## 판정

**조건부 통과 (REVISE)** — 구조·전수성·실행 지침의 품질은 발행 수준이다. 단:

1. **F-1 (BLOCKER)**: D-014 (b) probing classifier 권고 실험을 등재하기 전에는 발행 불가 (마스터 결정 사항).
2. **F-2 (MAJOR)**: w/o OD 행의 코드-사실 전제를 정정할 것 — 실행 지침 문서에서 기본 동작을 반대로 기술한 유일한 사실 오류.
3. F-3/F-4/F-5 (MINOR)는 같은 pass에서 일괄 수정 권장.

위 5건 반영 후 재검수 없이 발행 가능 (r2 fixlog로 갈음 가능한 수준).
