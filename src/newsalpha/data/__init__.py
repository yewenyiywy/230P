"""Stage 1, data: fetch raw tables, store them immutably, clean them.

sources/   one adapter per vendor; all return the contracts in newsalpha.schemas
storage    raw / interim / features / processed / runs layout (mirrors the Drive folder)
clean      revision filter, de-duplication
"""
