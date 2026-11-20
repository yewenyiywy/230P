"""Stage 4, features: events x entry delays x horizons -> the modelling PANEL.

panel   builds one row per (event, delay, horizon) with features known at entry
        and labels realised after entry; the only place prices meet text
gap     the per-delay reaction gap, fitted on training folds only
"""
