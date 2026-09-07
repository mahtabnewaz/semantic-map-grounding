"""Docs -> draft capability descriptor agent (Phase 6 study).

Reads a platform's public artefacts (URDF, package.xml, message
definitions, SDK documentation from data/platforms) and produces a draft
descriptor that a human then corrects.

Invariant (CLAUDE.md #8): no generated descriptor is used unreviewed.
Human correction is not a fallback — it is the measured quantity
(correction effort vs. authoring from scratch). A descriptor reaching
simulation without review is a bug.

The draft is validated by descriptor.py before a human sees it.

TODO: artefact ingestion, field-filling agent, null-with-reason handling.
"""
