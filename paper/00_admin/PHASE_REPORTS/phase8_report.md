---
phase: 8
agent: orchestrator
directives: [M9, R18, R3, M10, M13]
last_modified: 2026-06-11
---

# Phase 8 보고 — 최종 감사, Notion Placeholder 페이지, 핸드오프

## ① 수행 내용 요약

1. **최종 감사 (절대 엄격 구역)**: 신규 리뷰어 2인 (applied AI 저널 시니어 / TSAD 도메인 전문가) — 실제 학회 양식 (점수·강약점·판정·reject 사유). 결과: 양인 모두 "설계 문서로서 이례적으로 엄밀, Clarity 4/4" + Major Revision (사유의 대부분 = 수치 placeholder 부재·미실행 실험). **D-014 triage**: placeholder-본질 지적은 R18 "(placeholder는 허용)"·R3 정책상 기각, 실행 가능 2건 채택 — (a) 선택-기회 비대칭(100/50/10회) 명시 공개를 Appendix B.2에 보강 (본문 분량 좌표 단위 무영향 검증 + 미니 감사 3종 PASS) (b) GRL probing 분석을 권고 실험으로 Notion 명세에 등재.
2. **Notion placeholder 명세 (2단계 발행)**: 한국어 명세 작성 (placeholder 전수 + 신규 실행 11건 실행 지침 + 재사용 3건 실측 판정) → 독립 검수 (BLOCKER 1·MAJOR 1 적발: R-PROBE 미등재, w/o OD 전제 반전 — 코드 재확인 후 정정) → r2 → **발행**: 비교 실험 페이지 하위 단일 페이지, 단일 create-pages 호출, re-fetch 렌더링 검증 (헤딩 46개·표 11개 보존·절단 없음). URL: https://www.notion.so/37c87856b207810e83e3d1b5f14766fc
3. **Coverage 마감**: 57행 전부 DONE (T7+R37+M13) — 최종 전수 감사로 근거 유효성 재검증 (별도 보고).
4. **워크스페이스 마감**: INDEX 최종 갱신 (최종 인도물 요약 절 포함), git commit.

## ② 산출물

`08_final_audit/` 3종 + Notion 발행 페이지 + `99_reviews/p8_*.md` 3건 + appendix_B.tex 보강 + zip 재패키징 (12/12 일치, 단독 컴파일 3회째 PASS).

## ③ 게이트 결과

최종 감사 (D-014 triage 후 actionable reject급 0) + coverage 57/57 + Notion 발행·렌더링 확인 — 최종 전수 감사 PASS 시 종료.

## ④ 사용자 액션 필요 항목 (핸드오프 — 상세는 채팅 최종 보고)

1. **실험 수행 → placeholder 채우기**: Notion 명세의 신규 실행 11건 (우선순위: baseline 재실행 → weak 4종 → standard-split → ablation w/o GRL·symmetric decoder → sparsity sweep …) + 271canon 잔여 완주 (SMD 6/SMAP 49/MSL 22) + 재사용 3건 집계.
2. **저자 정보·저널명·코드 URL 기입** (main.tex [AUTHOR NAMES]·[JOURNAL NAME]·TXT-002).
3. **실수치·실figure 투입 후 분량 재측정** (본문 상한 여유 0.003p — 5p 판형 기준 재확인 필수).
4. SWaT 재현성 플래그 (RF-005: 모델 입력 45 feat vs 현 로더 51) — 코드 공개 전 정리 권장.
5. R-PROBE 권고 실험 (rebuttal 대비, 선택).

---

## 부록 — 재진입 round 1 (2026-06-13): KBS 정합화 + Notion 단일 페이지 재구축

### A. KBS (Knowledge-Based Systems) 포맷 정합화
- 공식 GFA 조사(`07_latex/KBS_FORMAT_REQUIREMENTS.md`) → 적용: Highlights 5개 전부 ≤85자 재작성(KBS 규정; 기존 125자 기준은 초과였음) + 제출용 `highlights.txt` / Keywords 7→6 / 선언 5종(CRediT·이해관계·**GenAI 고지**·Data availability·Funding — 저자 확정 placeholder) / `\journal{Knowledge-Based Systems}` / **flat 구조**(sections/*.tex → 루트) / zip flat 13파일 단독 컴파일 PASS.
- **GenAI 고지(사용자 지시)**: Elsevier 허용 최소 범위(가독성·문법 교정만)의 완성형 단문으로 축소 — 도구명·사유 템플릿 제거 ("문장 교정 수준"). 3 main*.tex 적용.
- 본문 8.997p 유지(선언 섹션은 references 동급 — R6 측정 외, 좌표 동일). 미니 감사 4/4 PASS(`99_reviews/p7_kbs_miniaudit_r1.md`).

### B. Notion placeholder 명세 — 단일 페이지 재구축 (사용자 지시 D-016/D-017)
- 위치 정정: 비교실험 페이지 하위 → **MAE for Anomaly Detection 하위**.
- 내용 격상: 모든 placeholder에 🎯목적·의도(논증 역할·방어 공격) + 🏁목표·기대 결과 차원 추가 — "Notion만 보고 실험 설계·figure 작성 가능" 수준.
- 구성: placeholder별 하위 페이지 폐기 → **단일 페이지 1장**(toggle heading 21개 + TOC + callout + 대시보드 + colored table). 한국어 전수 정제.
- 발행: https://www.notion.so/37e87856b20781fc92f6d8580c4b72a0 — 독립 검수 PASS + orchestrator 구조 무결 재확인(H2 8·토글 21·NUM 31·블록 파싱 확인·절단 0). 구 페이지(37c87856…) 중립화(제목 [이관됨] + 안내).

### C. 저자 액션 추가 항목 (핸드오프 보강)
1. 선언 5종 확정: CRediT 역할 배분 / 이해관계 / **GenAI 고지(현 최소 문안 검토)** / Data availability(SWaT·WaDi는 iTrust 신청 기반 — "publicly available" 문구 정밀화 권고) / Funding.
2. Highlights는 KBS 제출 시 **별도 파일**(`highlights.txt`)로도 업로드.
3. NUM-003([N] datasets) 확정 시 본문·highlights.txt·Notion 동기화.
4. (KBS) 제출 시스템에서 ORCID·declaration tool .docx 별도 요구 확인.
