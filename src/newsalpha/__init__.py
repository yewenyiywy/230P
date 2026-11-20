"""newsalpha: news-driven mispricing research pipeline for 230P.

The package is organised as one module per pipeline stage. Data flows strictly
left to right; each stage consumes and produces the DataFrames defined in
`newsalpha.schemas`:

    data -> align -> extraction -> features -> models -> trading -> evaluation

See docs/ARCHITECTURE.md for the full map.
"""

__version__ = "0.1.0"
