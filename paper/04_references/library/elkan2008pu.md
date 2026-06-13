---
phase: 4
agent: excerpt-curator-3
directives: [T4, R19]
last_modified: 2026-06-11
key: elkan2008pu
verification_status: VERIFIED_A
verified_by_A: 2026-06-11
corrected_fields: [title_case]
card_grade: LIGHT
abstract_source: author-camera-ready-PDF (cseweb.ucsd.edu/~elkan/posonly.pdf — 공식 dl.acm.org 403 차단; PDF 1면에서 verbatim 전사)
---
# Learning Classifiers from Only Positive and Unlabeled Data
**경고: abstract verbatim은 검증/표절 대조 전용 — 본문 복사 금지 (A2)**

## 서지 (scout 예비 — 검증 전)
- 저자 (PDF/Crossref): Charles Elkan, Keith Noto (Computer Science and Engineering, University of California, San Diego)
- Venue: KDD 2008 — Proceedings of the 14th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (Las Vegas, Nevada, USA, 2008-08-24), pp.213–220
- DOI: 10.1145/1401890.1401920
- DBLP: conf/kdd/ElkanN08
- fetch한 페이지: 저자 camera-ready PDF https://cseweb.ucsd.edu/~elkan/posonly.pdf (1면 직접 판독) + api.crossref.org (서지, 2026-06-11); 공식 ACM 페이지는 403

## Abstract 전문 (verbatim — 저자 PDF 1면 ABSTRACT 절 전사)
The input to an algorithm that learns a binary classifier normally consists of two sets of examples, where one set consists of positive examples of the concept to be learned, and the other set consists of negative examples. However, it is often the case that the available training data are an incomplete set of positive examples, and a set of unlabeled examples, some of which are positive and some of which are negative. The problem solved in this paper is how to learn a standard binary classifier given a nontraditional training set of this nature.

Under the assumption that the labeled examples are selected randomly from the positive examples, we show that a classifier trained on positive and unlabeled examples predicts probabilities that differ by only a constant factor from the true conditional probabilities of being positive. We show how to use this result in two different ways to learn a classifier from a nontraditional training set. We then apply these two new methods to solve a real-world problem: identifying protein records that should be included in an incomplete specialized molecular biology database. Our experiments in this domain show that models trained using the new methods perform better than the current state-of-the-art biased SVM method for learning from positive and unlabeled examples.

## 역할 (커버 claim)
- C-020: §2.2 단락 1 — PU Learning 샘플선별형(reliable-negative extraction류) 계열 원류 인용.

## 비고
- 통칭: Elkan & Noto (PU learning의 "selected completely at random" 가정 고전). [A1 확정] 공식 제목(Crossref/DBLP): "Learning classifiers from only positive and unlabeled data" (소문자; PDF 표제 "Learning Classifiers from Only Positive and Unlabeled Data"와 대소문자 차이 있음 — Crossref 공식 표기 우선).
