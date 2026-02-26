import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")


@app.cell
def __(mo):
    mo.md(
        r"""
        # Benchmark Legacy Mode Mappings Archive

        This notebook archives the old `current-*` mode identifiers that were
        previously used in benchmark artifacts.

        Active benchmark notebooks now use concise IDs (`d0-i1`, `d0-i3`, `d1-i3`, `d6-i1`).
        """
    )
    return


@app.cell
def __():
    import pandas as pd
    import marimo as mo

    return mo, pd


@app.cell
def __(pd):
    LEGACY_MODE_MAPPINGS = [
        {
            "legacy_mode": "current-default",
            "display_name": "Default (d0, i1)",
            "modern_id": "d0-i1",
            "notes": "Single pass, depth 0 baseline profile.",
        },
        {
            "legacy_mode": "current-no-recursion-best-of-3",
            "display_name": "No Recursion Best-of-3 (d0, i3)",
            "modern_id": "d0-i3",
            "notes": "No recursion, three iterations for retry-like quality lift.",
        },
        {
            "legacy_mode": "current-depth1-iter3",
            "display_name": "Depth-1 Best-of-3 (d1, i3)",
            "modern_id": "d1-i3",
            "notes": "Shallow recursion with multiple iterations.",
        },
        {
            "legacy_mode": "current-depth6-single-pass",
            "display_name": "Deep Recursion Single-Pass (d6, i1)",
            "modern_id": "d6-i1",
            "notes": "Deep recursion profile for hard decomposition tasks.",
        },
    ]
    return LEGACY_MODE_MAPPINGS


@app.cell
def __(LEGACY_MODE_MAPPINGS, mo, pd):
    df = pd.DataFrame(LEGACY_MODE_MAPPINGS)
    mo.md("## Archived Mapping Table")
    mo.ui.table(df)
    return df


if __name__ == "__main__":
    app.run()
