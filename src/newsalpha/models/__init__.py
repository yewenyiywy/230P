"""Stage 5, models: PANEL -> out-of-sample PREDICTIONS via walk-forward.

base         ReturnModel interface + Ridge and LightGBM implementations
walkforward  monthly expanding-window training with purging; fits the gap model per fold
"""
