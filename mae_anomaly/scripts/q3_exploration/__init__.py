"""
Q3 Exploration — Follow-up experiments on 2분기 findings.

2분기 보고서 (report_2분기/)에서 도출된 실험 방향을 본 폴더에서 실행:
- Phase A: Unsupervised E9 sigma estimation
- Phase B: Hybrid methods (E9 + NLM-T2, Conditional)
- F2: Cross-channel interaction
- F5: Dataset clustering
- F9: Multi-metric ensemble
- F10: Severity-weighted F1
- E9 sigma multiplier sweep

본 폴더는 self-contained: 자체 core/ 모듈 사용. 기존 mae_anomaly modules는 saved scores
loader와 evaluation function 등 minimum만 import.
"""
