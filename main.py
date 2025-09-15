import pandas as pd
import matplotlib.pyplot as plt
from utils import perplexity, get_per_token_log_probs, plot_log_probs
from data import data

results = []

for i, (category, gp_sent, bl_sent) in enumerate(data):
    gp_ppl = perplexity(gp_sent)
    bl_ppl = perplexity(bl_sent)
    delta = gp_ppl - bl_ppl
    relative_delta = (gp_ppl - bl_ppl) / bl_ppl

    print(f"\nPair {i + 1} ({category})")
    print(f"Garden-path sentence: {gp_sent}")
    print(f"Baseline sentence: {bl_sent}")
    print(f"Garden-path sentence perplexity: {gp_ppl:.2f}")
    print(f"Baseline sentence perplexity: {bl_ppl:.2f}")
    print(f"Surprisal delta: {delta:.2f}")

    results.append({
        "pair": i + 1,
        "category": category,
        "gp_sent": gp_sent,
        "bl_sent": bl_sent,
        "gp_ppl": gp_ppl,
        "bl_ppl": bl_ppl,
        "delta": delta,
        "relative delta": relative_delta
    })

    if i < 30:
        tokens_gp, log_probs_gp = get_per_token_log_probs(gp_sent)
        tokens_bl, log_probs_bl = get_per_token_log_probs(bl_sent)

        plot_log_probs(tokens_gp, log_probs_gp, title="Garden-Path Sentence")
        plot_log_probs(tokens_bl, log_probs_bl, title="Baseline Sentence")

# Make DataFrame
df = pd.DataFrame(results)

#Compute mean delta per category
mean_deltas = df.groupby("category")["delta"].mean().sort_values(ascending=False)
print(mean_deltas)
mean_deltas.plot(
    kind="bar",
    title="Average Surprisal Delta by Category",
    ylabel="Δ Perplexity",
    xlabel="Category",
    color="salmon"
)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Convert relative delta to percentage
df["relative_delta_percent"] = df["relative delta"] * 100

# Compute mean relative delta per category
mean_relative_deltas = df.groupby("category")["relative_delta_percent"].mean().sort_values(ascending=False)
print(mean_relative_deltas)
mean_relative_deltas.plot(
    kind="bar",
    title="Average Relative Perplexity Increase by Category",
    ylabel="Relative Δ Perplexity (%)",
    xlabel="Category",
    color="lightgreen"
)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
