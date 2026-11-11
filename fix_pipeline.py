# -*- coding: utf-8 -*-
"""
Corrected aggregation / normalization / similarity pipeline.
Fixes (vs. original thesis notebook):
  1. Per-90-minute rates instead of raw season totals (removes playing-time bias).
  2. Position via mode (most frequent role) instead of median of an ordinal code,
     kept as metadata/optional filter only -- NOT fed into the distance metric
     (see note below on why one-hot encoding it was tried and rejected).
  3. Players without a scraped market value are kept in the similarity pool
     (only excluded when a market-value cap is actually requested).

Note on Position: an earlier version of this fix one-hot-encoded Position into
the distance features. That turned out to overcorrect the original bug: it made
the minimum distance to ANY player of a different position (1.44) higher than
the AVERAGE distance between same-position players (0.50), i.e. position became
a hard partition instead of a soft signal, and it's what let the original
(flawed) thesis surface cross-position matches like Thuram-Dzeko. Position is
therefore left out of the distance computation entirely here, and only used
as metadata / through the existing `same_position` filter.

No new scraping is performed: only the already-saved per-match CSVs and the
already-saved market value CSV are used as input.
"""
import pandas as pd
import numpy as np
import re
import sys

RATE_STATS = [
    'Goals', 'Assists', 'Shots', 'Shots on Target', 'Yellow Cards', 'Touches',
    'Tackles', 'Interceptions', 'Blocks', 'xG', 'npxG', 'xA', 'SCA', 'GCA',
    'Passes Completed', 'Passes Attempted', 'Key Passes', 'Carries',
    'Carries Progressively', 'Dribbles Attempted', 'Dribble Success'
]

POSITION_MAP_INV = {
    1: 'CB', 2: 'LB', 3: 'RB', 4: 'WB', 5: 'DM', 6: 'CM', 7: 'LM',
    8: 'RM', 9: 'AM', 10: 'LW', 11: 'RW', 12: 'FW'
}


def load_season(path):
    return pd.read_csv(path, encoding='utf-8')


def per90_season_aggregate(df):
    """Per player, per season: sum raw counts, sum minutes, derive per-90 rates,
    and take the mode of Position (most frequently played role)."""
    grouped_sum = df.groupby('Player')[RATE_STATS + ['Minutes']].sum()
    matches = df.groupby('Player').size().rename('Match Played')
    started = df.groupby('Player')['Start'].sum().rename('Match Started')
    position_mode = df.groupby('Player')['Position'].agg(
        lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
    ).rename('Position')

    minutes = grouped_sum['Minutes'].replace(0, np.nan)
    per90 = grouped_sum[RATE_STATS].div(minutes, axis=0) * 90

    out = per90.join(matches).join(started).join(position_mode)
    out['Minutes'] = grouped_sum['Minutes']
    out = out.reset_index()
    return out


def combine_seasons(season_dfs):
    """Median of per-90 rates across the seasons a player appears in;
    overall mode of Position across all seasons, weighted by matches played
    per season (so a one-off cameo season doesn't outvote a full season)."""
    all_rows = pd.concat(season_dfs, ignore_index=True)

    rate_median = all_rows.groupby('Player')[RATE_STATS].median()
    totals = all_rows.groupby('Player')[['Minutes', 'Match Played', 'Match Started']].sum()

    pos_rows = []
    for _, r in all_rows.iterrows():
        if pd.notna(r['Position']) and r['Match Played'] > 0:
            pos_rows.extend([(r['Player'], r['Position'])] * int(r['Match Played']))
    pos_df = pd.DataFrame(pos_rows, columns=['Player', 'Position'])
    position_mode = pos_df.groupby('Player')['Position'].agg(lambda s: s.mode().iloc[0]).rename('Position')

    out = rate_median.join(totals).join(position_mode).reset_index()
    out['Position Label'] = out['Position'].map(POSITION_MAP_INV)
    return out


def convert_market_value(value):
    if pd.isna(value):
        return np.nan
    m = re.search(r'\d.*', str(value))
    if not m:
        return np.nan
    v = m.group()
    if 'm' in v:
        return float(v.replace('m', '')) * 1_000_000
    if 'k' in v:
        return float(v.replace('k', '')) * 1_000
    try:
        return float(v)
    except ValueError:
        return np.nan


def main():
    seasons = ['20_21', '21_22', '22_23', '23_24']
    season_aggs = []
    for s in seasons:
        df = load_season(f'players_stats_{s}.csv')
        season_aggs.append(per90_season_aggregate(df))
        print(f'season {s}: {df.shape[0]} match rows, {df["Player"].nunique()} players', file=sys.stderr)

    df_players = combine_seasons(season_aggs)
    print('combined players:', df_players.shape, file=sys.stderr)

    mv = pd.read_csv('players_market_value.csv', encoding='utf-8')
    mv['Market Value'] = mv['Market Value'].apply(convert_market_value)
    df_players = df_players.merge(mv, on='Player', how='left')
    n_missing = df_players['Market Value'].isna().sum()
    print(f'players kept without a market value: {n_missing} / {len(df_players)}', file=sys.stderr)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df_norm = df_players.copy()
    df_norm[RATE_STATS] = scaler.fit_transform(df_players[RATE_STATS])

    # Position is NOT part of the distance features (see module docstring) --
    # kept only as metadata for the same_position filter / display.
    from sklearn.metrics.pairwise import euclidean_distances
    similarity = euclidean_distances(df_norm[RATE_STATS].values)

    df_norm.to_csv('players_norm_fixed.csv', index=False, encoding='utf-8')
    np.save('similarity_euclidean_fixed.npy', similarity)
    print('saved players_norm_fixed.csv, similarity_euclidean_fixed.npy', file=sys.stderr)
    print('distance feature shape:', df_norm[RATE_STATS].shape, file=sys.stderr)


if __name__ == '__main__':
    main()
