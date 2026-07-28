---
name: lasad-revise
description: Run the LASAD manuscript top-venue feedback & revision orchestrator for Knowledge-Based Systems (KBS). Reads SPEC.md and drives a multi-agent Workflow that revises ./paper_writing/paper into ./paper_writing/paper_feedback/result/ (Overleaf zip + PDF) and loops until an explicit reviewer CONSENSUS. Use when the user asks to feedback/revise/polish the LASAD (semi-supervised time-series anomaly detection) paper to top-venue quality.
---

# LASAD Revision Orchestrator

너는 LASAD 논문을 top-venue(KBS) 수준으로 끌어올리는 **ORCHESTRATOR**다.

## 단일 진실원
무엇을 하기 전에 먼저 **`.Codex/skills/lasad-revise/SPEC.md`를 끝까지 정독**하라. 그 파일이 mission,
15개 prime directives(§2), 평가기준(§3 §A–§G), 절차(§4 P1–P3), 종료 조건(§5 CONSENSUS GATE), 작업공간(§6),
sub-agent roster(§7), phase pipeline(§8), 산출물(§9), RED LINES(§10)을 정의한다. **SPEC가 유일한 권위이며
그대로 따른다 — 임의로 우회·변형하지 마라.**

## 실행 방식
- **Workflow tool**로 SPEC §8 phase pipeline(P0→P8)을 실행한다. SPEC가 요구하는 quality pattern을 쓴다:
  blind 독립 리뷰 → 양방향 adversarial 검증 → adjudication(자동 채택 금지) → loop-until-dry →
  dual-build(LaTeX 정적 + PDF 시각) → CONSENSUS.
- **subagent는 fresh context로 시작한다.** 네가 띄우는 **모든** subagent prompt에 반드시 (a) SPEC.md의
  경로(`.Codex/skills/lasad-revise/SPEC.md`)와 (b) 그 에이전트가 따라야 할 정확한 섹션을 적어, 각자 직접
  Read하게 하라. 예: "Read SPEC.md §A-2 and P1; you are `kbs-fit-reviewer` …". subagent가 SPEC를 안다고
  가정하지 마라.
- **agent별 모델 배정(§7).** 각 subagent를 띄울 때 작업 성격에 맞춰 모델을 지정한다(`agent()`의 `model`):
  **문장 품질·추론·해석·판단**이 필요하면 `fable`, **단순 확인·기계적·반복**이면 `opus`. 애매하면 실제 해석·판단이 필요한지로 가른다.
- 결정·라운드는 SPEC §6가 지정한 작업공간 파일(`DECISION_LOG.md`, `FINDINGS_LEDGER.md`,
  `CONVERGENCE_LOG.md` 등)에 기록한다.

## 하드 불변식 (전체는 SPEC §2/§10)
- **오직 `./paper_writing/paper/`(예: `./paper_writing/paper/07_latex/`)에서만** 작업. **`./paper_writing/paper_legacy/`·`./paper_writing/paper_gpt/`·`./paper_writing/paper-gpt/`
  등 다른 `paper*`는 읽지도 참고하지도 마라.** `./paper_writing/paper/`는 불변 — `./paper_writing/paper_feedback/` 안 사본을 수정.
- 중립성/anti-anchoring: 기존 논문·이전 노트·사용자 의견을 *검증할 가설*로 취급(D1).
- 실측·확정(D4): 모든 정량 결과는 원고의 **실측값**, 모든 **Table·Figure는 확정 최종본(내용 불변)** — 날조 금지;
  캡션 교정과 PDF 시각 QA만 수행.
- 제목·모델명 고정: **LASAD: Label-Guided Adversarial Suppression for Semi-Supervised Time Series Anomaly
  Detection** (D9).
- KBS scope-fit은 명시 topic으로만 — "uncertain
  information processing"(IJUFKS) 어휘 금지(§A-2).
- SDMAE는 정직한 right-sizing, teacher-student distillation gap을 헤드라인화 금지(§G).
- 과잉 방어·변명조·선제 반론 차단 금지 — 의도·기여·목표를 긍정적으로 주장(D14). 수정 시 뺀 자기 내용을
  "A가 아니라 B" 식으로 되살리지 말 것 — 최종본은 처음부터 B로 쓰인 듯이(D15).
- 형식은 official elsarticle/KBS 출처로만 검증; LaTeX 정적 감사 **그리고** PDF 시각 QA 둘 다 필수(§A-3·P3).
- **스크래치 규약(§6-SCRATCH):** 네가 띄우는 모든 워크플로/Bash에서 `pdftoppm`·`latexmk` 등 *전이적* 렌더/빌드
  중간물(페이지 PNG·stdout 로그)은 반드시 **`/tmp/Codex-$(id -u)/lasad/`** 아래에 써라 — `paper_writing/.../temp/`나
  임의 `/tmp/<name>`은 bypass 모드에서도 매번 권한 프롬프트를 유발한다. 보존할 최종물만 `BUILD/`·`result/`로.

## 종료
**§5 CONSENSUS GATE 충족 전까지 멈추지 마라** — 다양한 sub-agent가 원고가 top-venue 상위 0.0001% 기준을
통과했다는 데 명시적으로 합의(미해결 Blocker/Critical/Major 0, 2연속 dry round, dual-build·reference·coherence
통과)하고, **§9 deliverable 3종이 `./paper_writing/paper_feedback/result/`에 모두 존재**할 때만. 그 후 최종 요약 보고를 올려라.
