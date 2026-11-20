"""Stage 4b, signals: signal-quality features beyond the main extraction (plan B-tier).

embed         text -> unit vectors: hashing (offline), sentence-transformers (CPU/GPU), or an endpoint
novelty       T5: time-safe retrieval of the firm's prior news -> novelty features + RAG context
disagreement  T4: dispersion of direction scores across several extraction methods

All outputs are keyed by event_id (SIGNALS contract) and join the PANEL as features.
"""
