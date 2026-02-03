import pandas as pd
import numpy as np
from pathlib import Path
import zipfile, xml.etree.ElementTree as ET
BASE = Path('/Users/Kyrosah/Documents/New project')
CSV_PATH = BASE / 'whl_2025.csv'
MATCHUPS_PATH = BASE / 'WHSDSC_Rnd1_matchups.xlsx'
PLOT_PATH = BASE / 'whl_scatter.svg'

# --- XLSX reader avoiding external deps (openpyxl not available) ---

def read_xlsx_simple(path, sheet='sheet1'):
    ns = {'main': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    def read_shared_strings(z):
        try:
            xml = z.read('xl/sharedStrings.xml')
        except KeyError:
            return []
        root = ET.fromstring(xml)
        strings = []
        for si in root.findall('main:si', ns):
            texts = [t.text or '' for t in si.findall('.//main:t', ns)]
            strings.append(''.join(texts))
        return strings
    def col_to_idx(col):
        idx = 0
        for c in col:
            idx = idx*26 + (ord(c.upper()) - 64)
        return idx-1
    with zipfile.ZipFile(path) as z:
        shared = read_shared_strings(z)
        root = ET.fromstring(z.read(f'xl/worksheets/{sheet}.xml'))
        rows = []
        for row in root.findall('main:sheetData/main:row', ns):
            r = {}
            for c in row.findall('main:c', ns):
                ref = c.attrib.get('r')
                col = ''.join(ch for ch in ref if ch.isalpha())
                idx = col_to_idx(col)
                t = c.attrib.get('t')
                v = c.find('main:v', ns)
                if v is None:
                    val = ''
                else:
                    if t == 's':
                        val = shared[int(v.text)]
                    else:
                        val = v.text
                r[idx] = val
            if r:
                max_idx = max(r.keys())
                rows.append([r.get(i, '') for i in range(max_idx+1)])
    return rows

# --- Core computation wrapped for repeatability ---

def compute_metrics():
    raw = pd.read_csv(CSV_PATH)

    home = pd.DataFrame({
        'game_id': raw['game_id'],
        'team': raw['home_team'],
        'opp_team': raw['away_team'],
        'off_line': raw['home_off_line'],
        'def_pairing': raw['home_def_pairing'],
        'opp_def_pairing': raw['away_def_pairing'],
        'toi_min': raw['toi'] / 60.0,
        'xg': raw['home_xg'],
        'xga': raw['away_xg'],
        'goals': raw['home_goals'],
        'goals_against': raw['away_goals'],
        'penalties_committed': raw['home_penalties_committed'],
        'penalty_minutes': raw['home_penalty_minutes'],
        'is_home': 1,
    })
    away = pd.DataFrame({
        'game_id': raw['game_id'],
        'team': raw['away_team'],
        'opp_team': raw['home_team'],
        'off_line': raw['away_off_line'],
        'def_pairing': raw['away_def_pairing'],
        'opp_def_pairing': raw['home_def_pairing'],
        'toi_min': raw['toi'] / 60.0,
        'xg': raw['away_xg'],
        'xga': raw['home_xg'],
        'goals': raw['away_goals'],
        'goals_against': raw['home_goals'],
        'penalties_committed': raw['away_penalties_committed'],
        'penalty_minutes': raw['away_penalty_minutes'],
        'is_home': 0,
    })
    long = pd.concat([home, away], ignore_index=True)

    # Defensive pairing strength
    pair = long.groupby(['team', 'def_pairing'], as_index=False).agg(xga=('xga', 'sum'), toi=('toi_min', 'sum'))
    pair['xga60'] = pair['xga'] / pair['toi'] * 60.0
    low, high = pair['xga60'].quantile([0.05, 0.95])
    pair['xga60_w'] = pair['xga60'].clip(lower=low, upper=high)
    league_def_xga60 = pair['xga60_w'].mean()
    opp_def_map = pair.set_index(['team', 'def_pairing'])['xga60_w']

    long['opp_def_xga60'] = long.set_index(['opp_team', 'opp_def_pairing']).index.map(opp_def_map)
    long['opp_def_xga60'] = long['opp_def_xga60'].fillna(league_def_xga60)
    long['matchup_factor'] = league_def_xga60 / long['opp_def_xga60']
    f_low, f_high = long['matchup_factor'].quantile([0.05, 0.95])
    long['matchup_factor'] = long['matchup_factor'].clip(lower=f_low, upper=f_high)

    # Opponent line strength for defensive adjustment
    line_raw = long.groupby(['team', 'off_line'], as_index=False).agg(xg=('xg', 'sum'), toi=('toi_min', 'sum'))
    line_raw['xg60'] = line_raw['xg'] / line_raw['toi'] * 60.0
    lr_low, lr_high = line_raw['xg60'].quantile([0.05, 0.95])
    line_raw['xg60_w'] = line_raw['xg60'].clip(lower=lr_low, upper=lr_high)
    league_line_xg60 = line_raw['xg60_w'].mean()
    line_xg60_map = line_raw.set_index(['team', 'off_line'])['xg60_w']

    long.loc[long['is_home'] == 1, 'opp_off_line'] = raw['away_off_line'].values
    long.loc[long['is_home'] == 0, 'opp_off_line'] = raw['home_off_line'].values
    long['opp_line_xg60'] = long.set_index(['opp_team', 'opp_off_line']).index.map(line_xg60_map)
    long['opp_line_xg60'] = long['opp_line_xg60'].fillna(league_line_xg60)
    long['def_matchup_factor'] = long['opp_line_xg60'] / league_line_xg60
    d_low, d_high = long['def_matchup_factor'].quantile([0.05, 0.95])
    long['def_matchup_factor'] = long['def_matchup_factor'].clip(lower=d_low, upper=d_high)

    long['adj_xg'] = long['xg'] * long['matchup_factor']
    long['adj_xga'] = long['xga'] * long['def_matchup_factor']

    team_off = long.groupby('team', as_index=False).agg(adj_xg=('adj_xg', 'sum'), toi=('toi_min', 'sum'))
    team_off['off_xg60_adj'] = team_off['adj_xg'] / team_off['toi'] * 60.0
    team_def = long.groupby('team', as_index=False).agg(adj_xga=('adj_xga', 'sum'), toi=('toi_min', 'sum'))
    team_def['def_xga60_adj'] = team_def['adj_xga'] / team_def['toi'] * 60.0

    team_st = long.groupby('team', as_index=False).agg(pen_min=('penalty_minutes', 'sum'), pen_cnt=('penalties_committed', 'sum'), toi=('toi_min', 'sum'))
    team_st['pen_min60'] = team_st['pen_min'] / team_st['toi'] * 60.0
    team_st['pen_cnt60'] = team_st['pen_cnt'] / team_st['toi'] * 60.0

    long['xg_diff_adj'] = long['adj_xg'] - long['adj_xga']
    ha = long.groupby(['team', 'is_home'], as_index=False).agg(xg_diff=('xg_diff_adj', 'sum'), toi=('toi_min', 'sum'))
    ha['xg_diff60'] = ha['xg_diff'] / ha['toi'] * 60.0
    home_diff = ha[ha['is_home'] == 1].set_index('team')['xg_diff60']
    away_diff = ha[ha['is_home'] == 0].set_index('team')['xg_diff60']
    ha_diff = (home_diff - away_diff).rename('home_away_diff')

    teams = team_off.merge(team_def, on='team').merge(team_st, on='team')
    teams = teams.merge(ha_diff, on='team', how='left')

    for col in ['off_xg60_adj', 'def_xga60_adj', 'pen_min60', 'home_away_diff']:
        teams[col + '_z'] = (teams[col] - teams[col].mean()) / teams[col].std(ddof=0)

    teams['rating_raw'] = (
        0.4 * teams['off_xg60_adj_z']
        + 0.4 * (-teams['def_xga60_adj_z'])
        + 0.1 * (-teams['pen_min60_z'])
        + 0.1 * (teams['home_away_diff_z'])
    )

    schedule = raw[['game_id', 'home_team', 'away_team']].drop_duplicates()
    ratings_raw = teams.set_index('team')['rating_raw']

    def avg_opp_rating(team):
        games = schedule[(schedule['home_team'] == team) | (schedule['away_team'] == team)]
        opps = games.apply(lambda r: r['away_team'] if r['home_team'] == team else r['home_team'], axis=1)
        return ratings_raw.reindex(opps).mean()

    sos = {team: avg_opp_rating(team) for team in teams['team']}
    teams['sos'] = teams['team'].map(sos)
    teams['sos_z'] = (teams['sos'] - teams['sos'].mean()) / teams['sos'].std(ddof=0)

    teams['rating_adj'] = teams['rating_raw'] - 0.15 * teams['sos_z']

    scores = raw.groupby('game_id', as_index=False).agg(
        home_team=('home_team', 'first'),
        away_team=('away_team', 'first'),
        home_goals=('home_goals', 'sum'),
        away_goals=('away_goals', 'sum'),
    )
    scores['home_win'] = (scores['home_goals'] > scores['away_goals']).astype(int)
    ratings = teams.set_index('team')['rating_adj']
    scores['rating_diff'] = scores['home_team'].map(ratings) - scores['away_team'].map(ratings)

    try:
        from sklearn.linear_model import LogisticRegression
        X = scores[['rating_diff']].values
        y = scores['home_win'].values
        model = LogisticRegression(fit_intercept=True, C=1e6, solver='lbfgs')
        model.fit(X, y)
        coef = float(model.coef_[0][0])
        intercept = float(model.intercept_[0])
    except Exception:
        X = scores[['rating_diff']].values
        y = scores['home_win'].values
        coef = 0.0
        intercept = np.log(y.mean() / (1 - y.mean()))
        for _ in range(50):
            z = intercept + coef * X[:, 0]
            p = 1 / (1 + np.exp(-z))
            g0 = (y - p).sum()
            g1 = ((y - p) * X[:, 0]).sum()
            w = p * (1 - p)
            h00 = -w.sum(); h01 = -(w * X[:, 0]).sum(); h11 = -(w * (X[:, 0] ** 2)).sum()
            det = h00 * h11 - h01 * h01
            if det == 0:
                break
            d0 = (g0 * h11 - g1 * h01) / det
            d1 = (g1 * h00 - g0 * h01) / det
            intercept -= d0
            coef -= d1

    league_avg_rating = teams['rating_adj'].mean()
    teams['composite_score'] = 1 / (1 + np.exp(-(coef * (teams['rating_adj'] - league_avg_rating))))

    ranked = teams.sort_values('composite_score', ascending=False).reset_index(drop=True)
    ranked['rank'] = np.arange(1, len(ranked) + 1)

    # line disparity
    line_adj = long.groupby(['team', 'off_line'], as_index=False).agg(adj_xg=('adj_xg', 'sum'), toi=('toi_min', 'sum'))
    line_adj['adj_xg60'] = line_adj['adj_xg'] / line_adj['toi'] * 60.0
    first = line_adj[line_adj['off_line'] == 'first_off'].set_index('team')['adj_xg60']
    second = line_adj[line_adj['off_line'] == 'second_off'].set_index('team')['adj_xg60']
    disparity = (first / second).replace([np.inf, -np.inf], np.nan).dropna()

    # correlation for plotting annotation
    rank_map = ranked.set_index('team')['rank']
    common = disparity.index.intersection(rank_map.index)
    corr = np.corrcoef(rank_map.loc[common], disparity.loc[common])[0, 1]

    return ranked[['team', 'rank', 'composite_score']], disparity, corr


def build_plot(ranked, disparity, corr, path=PLOT_PATH):
    df = ranked.set_index('team').join(disparity.rename('disparity_ratio')).dropna().reset_index()
    df['tier'] = pd.cut(df['rank'], bins=[0, 8, 24, 32], labels=['Top 8', 'Middle 16', 'Bottom 8'])

    width, height = 1000, 600
    margin_left, margin_right, margin_top, margin_bottom = 80, 40, 50, 80
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    x_min, x_max = df['rank'].min(), df['rank'].max()
    y_min, y_max = df['disparity_ratio'].min(), df['disparity_ratio'].max()

    def x_scale(x):
        return margin_left + (plot_width) * (x - x_min) / (x_max - x_min)

    def y_scale(y):
        return margin_top + (plot_height) * (1 - (y - y_min) / (y_max - y_min))

    m, b = np.polyfit(df['rank'], df['disparity_ratio'], 1)
    x_line = np.array([x_min, x_max])
    y_line = m * x_line + b

    tier_color = {'Top 8': '#1f77b4', 'Middle 16': '#2ca02c', 'Bottom 8': '#d62728'}

    parts = []
    parts.append(f'<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">')
    parts.append('<style>text{font-family:Arial,Helvetica,sans-serif;}</style>')

    x0, y0 = margin_left, height - margin_bottom
    x1, y1 = width - margin_right, y0
    parts.append(f'<line x1=\"{x0}\" y1=\"{y0}\" x2=\"{x1}\" y2=\"{y1}\" stroke=\"#000\" stroke-width=\"2\"/>')
    parts.append(f'<line x1=\"{x0}\" y1=\"{margin_top}\" x2=\"{x0}\" y2=\"{y0}\" stroke=\"#000\" stroke-width=\"2\"/>')

    for r in range(int(x_min), int(x_max) + 1, 4):
        x = x_scale(r)
        parts.append(f'<line x1=\"{x}\" y1=\"{y0}\" x2=\"{x}\" y2=\"{y0 + 6}\" stroke=\"#000\"/>')
        parts.append(f'<text x=\"{x}\" y=\"{y0 + 22}\" font-size=\"12\" text-anchor=\"middle\">{r}</text>')
    parts.append(f'<text x=\"{(x0 + x1) / 2}\" y=\"{height - 20}\" font-size=\"14\" text-anchor=\"middle\">Team Power Rank (1 = strongest)</text>')

    for i in range(6):
        val = y_min + (y_max - y_min) * i / 5
        y = y_scale(val)
        parts.append(f'<line x1=\"{x0 - 6}\" y1=\"{y}\" x2=\"{x0}\" y2=\"{y}\" stroke=\"#000\"/>')
        parts.append(f'<text x=\"{x0 - 10}\" y=\"{y + 4}\" font-size=\"12\" text-anchor=\"end\">{val:.2f}</text>')
        parts.append(f'<line x1=\"{x0}\" y1=\"{y}\" x2=\"{x1}\" y2=\"{y}\" stroke=\"#ccc\" stroke-width=\"1\" stroke-dasharray=\"2,2\"/>')
    parts.append(f'<text transform=\"translate({20},{(margin_top + plot_height / 2)}) rotate(-90)\" font-size=\"14\" text-anchor=\"middle\">First/Second Line Disparity Ratio</text>')

    parts.append(f'<line x1=\"{x_scale(x_line[0])}\" y1=\"{y_scale(y_line[0])}\" x2=\"{x_scale(x_line[1])}\" y2=\"{y_scale(y_line[1])}\" stroke=\"#666\" stroke-width=\"2\" stroke-dasharray=\"6,4\"/>')
    parts.append(f'<text x=\"{x_scale(x_line[0]) + 10}\" y=\"{y_scale(y_line[0]) - 10}\" font-size=\"12\" fill=\"#555\">Trend (r={corr:.2f})</text>')

    for row in df.itertuples():
        x = x_scale(row.rank)
        y = y_scale(row.disparity_ratio)
        radius = (2000 * (row.composite_score ** 2) / np.pi) ** 0.5
        color = tier_color[row.tier]
        parts.append(f'<circle cx=\"{x}\" cy=\"{y}\" r=\"{radius}\" fill=\"{color}\" fill-opacity=\"0.65\" stroke=\"#000\" stroke-width=\"0.7\"/>')
        parts.append(f'<title>{row.team}: rank {row.rank}, disparity {row.disparity_ratio:.2f}, win prob {row.composite_score:.2f}</title>')

    legend_x, legend_y = x1 - 150, margin_top + 10
    parts.append(f'<text x=\"{legend_x}\" y=\"{legend_y - 16}\" font-size=\"12\" font-weight=\"bold\">Tier</text>')
    for i, (label, color) in enumerate(tier_color.items()):
        y = legend_y + i * 22
        parts.append(f'<rect x=\"{legend_x}\" y=\"{y - 10}\" width=\"14\" height=\"14\" fill=\"{color}\" stroke=\"#000\" stroke-width=\"0.5\"/>')
        parts.append(f'<text x=\"{legend_x + 20}\" y=\"{y + 1}\" font-size=\"12\">{label}</text>')

    parts.append('</svg>')
    Path(path).write_text('\\n'.join(parts))


def verify_repeatability(runs=3):
    rankings = []
    disparities = []
    corrs = []
    for _ in range(runs):
        ranked, disparity, corr = compute_metrics()
        rankings.append(ranked[['team', 'rank', 'composite_score']].reset_index(drop=True))
        disparities.append(disparity.sort_index())
        corrs.append(corr)
    same_rank = all(rankings[0].equals(r) for r in rankings[1:])
    same_disp = all(disparities[0].equals(d) for d in disparities[1:])
    same_corr = all(np.isclose(corrs[0], c) for c in corrs[1:])
    return same_rank, same_disp, same_corr, rankings[0], disparities[0], corrs[0]


def main():
    same_rank, same_disp, same_corr, ranked, disparity, corr = verify_repeatability(runs=4)
    if not (same_rank and same_disp and same_corr):
        raise SystemExit('Repeatability check failed')
    build_plot(ranked, disparity, corr, path=PLOT_PATH)
    print('Repeatability: rank OK:', same_rank, 'disparity OK:', same_disp, 'corr OK:', same_corr)
    print('Correlation:', round(float(corr), 4))
    print('Plot saved to', PLOT_PATH)
    print('Top 5 teams:', ranked.head(5).to_dict(orient='records'))


if __name__ == '__main__':
    main()
