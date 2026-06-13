---
phase: 4
agent: excerpt-curator-2
directives: [T4]
last_modified: 2026-06-11
---

# Manifest — excerpt-curator-2 (짝수 위치 FULL cards)

## 담당 배정 근거

SCOUT_CANDIDATE_LIST §A FULL+FULL-cond 22편의 목록 등장 순서 기준 짝수 위치(2,4,6,8,10,12,14,16,18,20,22).

---

## 작성 완료 카드 목록 (11편)

| 위치 | Key | 파일 | EXCERPT_UNVERIFIED | 비고 |
|-----|-----|------|-------------------|------|
| 2 | ristea2024sdmae | library/ristea2024sdmae.md | 없음 (HTML 직접 접근 성공) | CVPR 2024; publisher DOI verifier-TODO |
| 4 | zhang2022selfdistill | library/zhang2022selfdistill.md | 2건 (TPAMI abstract + 본문 미접근) | IEEE Xplore 페이페월; verifier 본문 발췌 필수 |
| 6 | lin2017focal | library/lin2017focal.md | 1건 (p_t 공식 — abstract에 없음) | arXiv abs 접근; verifier PDF §3 발췌 필수 |
| 8 | kim2022rigorous | library/kim2022rigorous.md | 1건 (PA%K 정의 — abstract에 없음) | AAAI ojs abs 접근; verifier PDF §3/§4 발췌 필수 |
| 10 | huet2022affiliation | library/huet2022affiliation.md | 없음 (abstract 충분) | arXiv abs 접근; affiliation 공식은 PDF |
| 12 | liu2024elephant | library/liu2024elephant.md | 없음 (abstract 충분) | NeurIPS 2024 D&B; clean-train 발췌는 verifier |
| 14 | sultani2018deepmil | library/sultani2018deepmil.md | 없음 (abstract 충분) | CVPR 2018; 비디오 도메인 |
| 16 | liu2024treemil | library/liu2024treemil.md | 없음 (abstract 충분) | ICASSP 2024 (venue 정정 확인) |
| 18 | ruff2020deepsad | library/ruff2020deepsad.md | 1건 (SAD 목적함수 공식) | ICLR 2020; verifier PDF §3 필수 |
| 20 | xue2022fewpositive | library/xue2022fewpositive.md | 없음 (abstract 충분) | IJCNN 2022; **C-011/C-025 반증 후보 ①** |
| 22 | darban2024dacad | library/darban2024dacad.md | 없음 (abstract 충분) | TKDE 2025 (venue 확정 — verifier 재확인); FULL-cond |

---

## EXCERPT_UNVERIFIED 목록

총 4건:

1. **zhang2022selfdistill** — IEEE Xplore 페이페월로 abstract verbatim 및 본문 미접근. verifier가 기관 접속으로 abstract + self-distillation 정의 발췌 확보 필요.
2. **lin2017focal** — abstract 접근 완료. p_t 정의 공식 및 FL(p_t) = -(1-p_t)^γ log(p_t) 수식이 abstract에 없음 (PDF §3). verifier 발췌 필수 (C-037 차별화 1문장 작성에 필요).
3. **kim2022rigorous** — abstract 접근 완료. PA%K 정확한 정의(K 파라미터 의미, 공식)가 abstract에 없음 (본문 §3/§4). verifier 발췌 필수 (C-047 지표 소개에 필요).
4. **ruff2020deepsad** — abstract 접근 완료. SAD 목적함수 공식(hypersphere center c, 라벨 이상 항)이 abstract에 없음 (PDF §3). arXiv HTML 404 반환. verifier PDF 접근 필요.

---

## "지지 발췌 없음" 발생 claim 목록

아래 claim들은 배정된 카드들이 직접 지지하나, 핵심 발췌가 abstract 수준에서만 확보됨:

- **C-053** (AR threshold 관행): Anomaly Transformer 본문 발췌 미확보 (CLAIM_CITATION_MAP R30 보류 유지). 이 claim은 xu2022anomalytransformer 카드 담당 (curator 1 배정). curator 2 배정 카드에서는 관련 없음.
- **C-028 지지 발췌**: zhang2022selfdistill의 "self-distillation" 정의 문장 미확보 (EXCERPT_UNVERIFIED). 용어 원류로서의 인용은 가능하나 verbatim 발췌 지지 없음.
- **C-037 지지 발췌**: lin2017focal의 p_t 공식 및 FL 수식 미확보 (EXCERPT_UNVERIFIED). abstract 수준 발췌로 차별화 서술 방향은 확인되었으나 수식 기반 1문장 차별화는 verifier 발췌 후 가능.

---

## verifier 작업 지시사항

| Key | 필요 작업 |
|-----|---------|
| zhang2022selfdistill | IEEE Xplore 기관 접속 → abstract verbatim + §1/§3 self-distillation 정의 발췌 |
| lin2017focal | arXiv PDF §3 → p_t 정의 + FL(p_t) 공식 발췌 (C-037 차별화 1문장용) |
| kim2022rigorous | AAAI PDF §3/§4 → PA%K 정식 정의 + K 파라미터 설명 발췌 |
| ruff2020deepsad | ICLR 2020 PDF §3 → SAD 목적함수 공식 (center c, labeled anomaly 항) |
| ristea2024sdmae | CVPR 2024 publisher DOI 확인 (arXiv 2306.12041 외) |
| darban2024dacad | TKDE 2025 vol.37 no.8 pp.4485-4496 최종 확인 |
| liu2024elephant | 본문(§2/§3) → clean-train 가정 비판 명시 발췌 (C-008/C-045 필수) |
| xue2022fewpositive | IJCNN 2022 IEEE DOI 확인 + PDF 전문 정독 (반증 성립 여부 판정 필수) |

---

## 특이사항

- **xue2022fewpositive**: C-011/C-025 최강 반증 후보 ①. verifier의 PDF 정독 없이 우리 논문의 novelty 주장 최종 서술 불가.
- **darban2024dacad**: venue가 arXiv → TKDE 2025로 갱신됨. FULL-cond 조건 해소 — verifier 재확인 후 인용 표기 업데이트 필요.
- **ristea2024sdmae**: HTML 직접 접근으로 핵심 발췌(student decoder 분기 문장 verbatim) 확보 완료 — 이 카드의 C-030 지지 발췌는 가장 신뢰도 높음.
