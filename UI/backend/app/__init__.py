"""TSMAE experiment-dashboard backend package.

Implements the data-access / compute / GIF layer per
``temp/dashboard_build_20260601/03_architecture/backend-design.md``.

HARD SAFETY (PROTOCOL §1, enforced in code):
  * results/ and .trash/ are opened READ-ONLY ('rb'); never written/moved/deleted.
  * all writes land under ./UI/ (the cache root); never under results/ or .trash/.
  * CPU-only; this package never imports torch, never touches the GPU.
  * *.pt / checkpoints/ are stat-only — there is NO code path that opens a .pt body.
"""

__version__ = "0.1.0"
