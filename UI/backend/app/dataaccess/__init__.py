"""Read-only data-access layer (discovery, normalize, classify, cache).

This is the ONLY package that touches the filesystem under results/ / .trash/.
Every endpoint goes through it. All file opens are read-only.
"""
