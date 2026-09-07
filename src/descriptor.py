"""Capability descriptor schema + validator.

A fixed-schema record declaring a robot platform's footprint, kinematic
class, sensor complement, frame conventions, and supported command
interfaces.

Invariant (CLAUDE.md #7): the schema is hand-defined and fixed. The agent
fills fields; it never invents them. Unknown/unfindable values are
explicit nulls with a reason, never guessed. The validator is a non-model
checker that rejects anything malformed.

Written early (before the Phase-6 onboarding study) because the platform
abstraction layer consumes the same schema. One schema, two studies.

TODO: schema definition + strict validator.
"""
