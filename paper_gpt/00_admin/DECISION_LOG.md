# Decision Log

Created: 2026-06-13 KST

| ID | Date | Decision | Rationale | Impact |
|---|---|---|---|---|
| D-0001 | 2026-06-13 | Execute `paper_gpt/orchestrator_master_prompt.md` as the authoritative workflow. | User requested end-to-end execution of the master prompt. | All artifacts and phase gates follow the prompt. |
| D-0002 | 2026-06-13 | Keep all generated work under `paper_gpt/`. | Required by user and master prompt. | Avoids contaminating existing `paper/` and `paper_legacy/` work. |
| D-0003 | 2026-06-13 | Treat `paper_legacy/**` and most of `paper/**` as forbidden sources. | User explicitly forbade previous paper artifacts; allowed only the Korean PDF and Elsevier template text. | Source inventory must filter these paths. |
| D-0004 | 2026-06-13 | Run Phase 0 before any substantive research or writing. | Master prompt requires governance before research. | Prevents requirement loss. |
| D-0005 | 2026-06-13 | Add KBS fit, contribution reframing, anomaly-priority masking de-emphasis, time-series interaction, and Notion readability to governance requirements. | User supplied feedback from a prior final-output review and requested it be pre-integrated. | These checks become formal requirements and final audit items. |
