# chart.py
# Generates a professional Seaborn heatmap of customer engagement patterns
# Output: chart.png (exactly 512x512 px)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def make_synthetic_engagement(seed: int = 42) -> pd.DataFrame:
    """
    Create realistic synthetic engagement data:
    - Rows: Days of week
    - Cols: Hours of day (0..23)
    Pattern: higher midday + evening use, weekday/daypart effects, weekend shift.
    """
    rng = np.random.default_rng(seed)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    hours = np.arange(24)

    # Day effects (Fri peak, weekend evening bias)
    day_effect = {
        "Mon": -3, "Tue": 0, "Wed": 2, "Thu": 4,
        "Fri": 8, "Sat": 5, "Sun": 1
    }

    # Hour-of-day curves: midday + evening peaks
    def gaussian(x, mu, sigma):
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    midday = gaussian(hours, mu=13, sigma=3.8)    # lunch/early afternoon
    evening = gaussian(hours, mu=20, sigma=2.8)   # prime evening

    base_curve = 20 + 55 * midday + 40 * evening  # baseline intensity

    data = []
    for d in days:
        # Weekday vs weekend adjustments
        is_weekend = d in {"Sat", "Sun"}
        # Weekend: relatively stronger evening, slightly lower midday
        curve = base_curve * (1.05 if d == "Fri" else (0.98 if is_weekend else 1.00))
        if is_weekend:
            curve = curve * (1.08 * (0.9 + evening)) / (1.0 + midday*0.15)

        # Add day effect and noise
        noise = rng.normal(0, 3.5, size=hours.size)
        row = np.clip(curve + day_effect[d] + noise, 0, None)
        data.append(row)

    df = pd.DataFrame(data, index=days, columns=hours)
    return df.round(1)

def main():
    # --- Seaborn styling (best practices) ---
    sns.set_style("white")           # clean background
    sns.set_context("talk")          # presentation-friendly text sizes

    # --- Data ---
    df = make_synthetic_engagement()

    # --- Figure: 8x8 inches @ 64 dpi => 512x512 px ---
    plt.figure(figsize=(8, 8))

    # Professional colormap (seaborn palette as cmap)
    cmap = sns.color_palette("rocket_r", as_cmap=True)

    ax = sns.heatmap(
        df,
        cmap=cmap,
        square=True,              # equal aspect ratio -> clean grid
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Engagement Index"},
    )

    # Titles & labels
    ax.set_title("Customer Engagement Heatmap\n(7 days × 24 hours)", pad=14)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")

    # Ticks: show every 2 hours for readability
    ax.set_xticks(np.arange(0.5, 24.5, 2))
    ax.set_xticklabels([str(h) for h in range(0, 24, 2)], rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Save exactly 512×512 px (8 in × 64 dpi) as required
    plt.savefig("chart.png", dpi=64, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()
