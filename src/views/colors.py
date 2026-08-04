"""
Shared archetype color palette and labels — one archetype means one color
everywhere in the app that encodes archetype as a categorical color.

Not used for magnitude encodings: V2's full-matrix heatmap colors cells by
count/percentage (a sequential blue scale), a different kind of encoding
that would conflict with categorical per-archetype coloring.

A1-A4 are a cohesive but visually distinct family, not a strict light-to-
dark gradient — A2 and A3 are siblings in the migration graph (both lead
to A4, neither leads to the other), so a straight gradient would visually
imply an ordering that isn't accurate. A4 is the most saturated as the
single convergence point both branches lead to.
"""

ARCHETYPE_COLORS = {
    "A1_TINKERER": "#90caf9",
    "A2_RESEARCHER": "#42a5f5",
    "A3_BATCH": "#26a69a",
    "A4_PRODUCTION": "#1565c0",
    "A0_UNCLASSIFIED": "#b0b0b0",
    "CHURNED": "#c0392b",
}

ARCHETYPE_LABELS = {
    "A1_TINKERER": "A1 Tinkerer",
    "A2_RESEARCHER": "A2 Researcher",
    "A3_BATCH": "A3 Batch",
    "A4_PRODUCTION": "A4 Production",
    "A0_UNCLASSIFIED": "A0 Unclassified",
    "CHURNED": "Churned",
}
