# -*- coding: utf-8 -*-
"""
Reproduces the Thuram / Pavard case study using the corrected similarity
pipeline (see fix_pipeline.py). Run fix_pipeline.py first to generate
players_norm_fixed.csv and similarity_euclidean_fixed.npy.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

RATE_STATS = [
    'Goals', 'Assists', 'Shots', 'Shots on Target', 'Yellow Cards', 'Touches',
    'Tackles', 'Interceptions', 'Blocks', 'xG', 'npxG', 'xA', 'SCA', 'GCA',
    'Passes Completed', 'Passes Attempted', 'Key Passes', 'Carries',
    'Carries Progressively', 'Dribbles Attempted', 'Dribble Success'
]

df = pd.read_csv('players_norm_fixed.csv', encoding='utf-8')
sim = np.load('similarity_euclidean_fixed.npy')


def top_n_similar(player_name, n=10, max_market_value=None):
    idx = df.index[df['Player'] == player_name]
    if len(idx) == 0:
        print(f'{player_name} not found', file=sys.stderr)
        return []
    idx = idx[0]
    scores = sorted(list(enumerate(sim[idx])), key=lambda x: x[1])
    results = []
    for i, score in scores[1:]:
        if max_market_value is not None:
            mv = df.iloc[i]['Market Value']
            if pd.notna(mv) and mv > max_market_value:
                continue
        results.append((i, df.iloc[i]['Player'], df.iloc[i]['Position Label'], score))
        if len(results) == n:
            break
    print(f'Top {n} similar to {player_name} (per-90 stats, corrected):')
    for _, name, pos, score in results:
        print(f'  {name} ({pos}) - distance {score:.3f}')
    return results


def players_distance(p1, p2):
    i1 = df.index[df['Player'] == p1][0]
    i2 = df.index[df['Player'] == p2][0]
    print(f'Distance {p1} <-> {p2}: {sim[i1][i2]:.3f}')


def plot_radar(player_name, results, out_path):
    idx = df.index[df['Player'] == player_name][0]
    angles = np.linspace(0, 2 * np.pi, len(RATE_STATS), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True})
    values = df.iloc[idx][RATE_STATS].values.tolist()
    values += values[:1]
    ax.plot(angles, values, label=player_name, linewidth=2, linestyle='solid')
    ax.fill(angles, values, alpha=0.25)

    for i, name, _, _ in results:
        v = df.iloc[i][RATE_STATS].values.tolist()
        v += v[:1]
        ax.plot(angles, v, label=name, linewidth=1, linestyle='dashed')
        ax.fill(angles, v, alpha=0.1)

    ax.set_yticks([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(RATE_STATS, fontsize=9)
    ax.set_title(f"Player and Top {len(results)} Similar Players (per-90 stats, corrected)", size=13)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    print(f'saved {out_path}')


if __name__ == '__main__':
    results = top_n_similar('Marcus Thuram', n=10)
    print()
    players_distance('Marcus Thuram', 'Romelu Lukaku')
    print()
    top_n_similar('Benjamin Pavard', n=10)
    print()
    # Stress test: Cancelo is famous for playing all over the pitch (full-back,
    # wing-back, central midfield depending on the system) -- a good check that
    # Position is genuinely not driving the distance.
    cancelo_results = top_n_similar('João Cancelo', n=10)
    print()
    plot_radar('Marcus Thuram', results, 'assets/similar_players_corrected.png')
    plot_radar('João Cancelo', cancelo_results, 'assets/cancelo_similarity.png')
