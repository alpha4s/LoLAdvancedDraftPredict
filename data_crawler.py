import sqlite3, time, json
from collections import deque
from riotwatcher import LolWatcher, ApiError

api_key, routing_region, platform_region = '', '', ''
watcher = LolWatcher(api_key)
min_matches, min_apps = 70000, 5
VALID_ROLES = {'TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY'}

def load_champions():
    with open('champions.json') as f:
        return json.load(f)

def get_team_by_roles(participants, team_id):
    team_p = [p for p in participants if p['teamId'] == team_id]
    if len(team_p) != 5: return None, None
    role_map, stats_map = {}, {}
    for p in team_p:
        pos = p.get('teamPosition')
        if pos not in VALID_ROLES or pos in role_map:
            return None, None  # Drop match if role is invalid or duplicated to prevent label corruption
        role_map[pos] = p['championName']
        stats_map[pos] = (int(p.get('physicalDamageDealtToChampions', 0)), int(p.get('magicDamageDealtToChampions', 0)))
    
    champs = [role_map['TOP'], role_map['JUNGLE'], role_map['MIDDLE'], role_map['BOTTOM'], role_map['UTILITY']]
    dmgs = [val for r in ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY'] for val in stats_map[r]]
    return champs, dmgs

def get_coverage(cursor, champions):
    cols = ['blue_top', 'blue_jungle', 'blue_mid', 'blue_bot', 'blue_support',
            'red_top', 'red_jungle', 'red_mid', 'red_bot', 'red_support']
    query = " UNION ALL ".join([f"SELECT {col} AS champ FROM matches WHERE {col} != ''" for col in cols])
    cursor.execute(f"SELECT champ, COUNT(*) FROM ({query}) GROUP BY champ")
    counts = dict(cursor.fetchall())
    champs = list(champions.values()) if isinstance(champions, dict) else list(champions)
    met = sum(1 for c in champs if counts.get(c, 0) >= min_apps)
    return len(champs), met

def init_db():
    conn = sqlite3.connect('league_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS matches (
        match_id TEXT PRIMARY KEY, game_version TEXT, winning_team TEXT,
        blue_top TEXT, blue_jungle TEXT, blue_mid TEXT, blue_bot TEXT, blue_support TEXT,
        red_top TEXT, red_jungle TEXT, red_mid TEXT, red_bot TEXT, red_support TEXT)''')
    c.execute('CREATE TABLE IF NOT EXISTS processed_players (puuid TEXT PRIMARY KEY)')
    
    c.execute('PRAGMA table_info(matches)')
    existing_cols = {row[1] for row in c.fetchall()}
    damage_cols = [
        'blue_top_phys', 'blue_top_magic', 'blue_jungle_phys', 'blue_jungle_magic',
        'blue_mid_phys', 'blue_mid_magic', 'blue_bot_phys', 'blue_bot_magic',
        'blue_supp_phys', 'blue_supp_magic', 'red_top_phys', 'red_top_magic',
        'red_jungle_phys', 'red_jungle_magic', 'red_mid_phys', 'red_mid_magic',
        'red_bot_phys', 'red_bot_magic', 'red_supp_phys', 'red_supp_magic'
    ]
    for col in damage_cols:
        if col not in existing_cols:
            c.execute(f'ALTER TABLE matches ADD COLUMN {col} INT DEFAULT 0')
    conn.commit()
    return conn

def get_next_players(state):
    tiers, divs = ['PLATINUM', 'EMERALD', 'DIAMOND'], ['I', 'II', 'III', 'IV']
    while state['tier'] < len(tiers):
        t, d, p = tiers[state['tier']], divs[state['div']], state['page']
        try:
            entries = watcher.league.entries(platform_region, 'RANKED_SOLO_5x5', t, d, page=p)
            if not entries:
                state['page'], state['div'] = 1, state['div'] + 1
                if state['div'] >= 4:
                    state['div'], state['tier'] = 0, state['tier'] + 1
                continue
            state['page'] += 1
            return entries
        except ApiError as e:
            if e.response.status_code in [401, 403, 429]: raise e
            state['page'] += 1
    return []

def crawl():
    conn = init_db()
    c = conn.cursor()
    champs = load_champions()
    c.execute('SELECT COUNT(*) FROM matches')
    matches = c.fetchone()[0]
    total, met = get_coverage(c, champs)
    print(f"Initial Matches: {matches} | Coverage: {met}/{total} ({met/total:.1%})")

    state = {'tier': 0, 'div': 0, 'page': 1}
    queue = deque()

    while matches < min_matches or met < total:
        if not queue:
            try:
                entries = get_next_players(state)
                if not entries:
                    print("No more player entries available.")
                    break
                queue.extend(entries)
            except ApiError as e:
                if e.response.status_code == 429:
                    time.sleep(int(e.response.headers.get('Retry-After', 60)))
                continue

        entry = queue.popleft()
        puuid = entry['puuid']
        if c.execute('SELECT 1 FROM processed_players WHERE puuid = ?', (puuid,)).fetchone():
            continue

        try:
            for m_id in watcher.match.matchlist_by_puuid(routing_region, puuid, count=20, queue=420):
                if matches >= min_matches and met >= total: break
                if c.execute('SELECT 1 FROM matches WHERE match_id = ?', (m_id,)).fetchone():
                    continue

                match = watcher.match.by_id(routing_region, m_id)
                info = match.get('info', {})
                if info.get('queueId') != 420 or info.get('gameDuration', 0) < 900:
                    continue

                b_team, b_dmgs = get_team_by_roles(info.get('participants', []), 100)
                r_team, r_dmgs = get_team_by_roles(info.get('participants', []), 200)
                if not b_team or not r_team:
                    continue  # Skip match if role assignment is ambiguous

                winner = "BLUE_WIN" if next(t for t in info.get('teams', []) if t['teamId'] == 100)['win'] else "RED_WIN"
                game_version = info.get('gameVersion', '')

                sql = '''INSERT OR IGNORE INTO matches VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'''
                c.execute(sql, (m_id, game_version, winner, *b_team, *r_team, *b_dmgs, *r_dmgs))
                matches += 1
                conn.commit()
                total, met = get_coverage(c, champs)
                print(f"[{matches}/{min_matches}] Saved {m_id} | Patch: {game_version} | Coverage: {met}/{total} ({met/total:.1%})")

            c.execute('INSERT OR IGNORE INTO processed_players VALUES (?)', (puuid,))
            conn.commit()
        except ApiError as e:
            if e.response.status_code == 429:
                queue.appendleft(entry)
                time.sleep(int(e.response.headers.get('Retry-After', 60)))

    conn.close()
    print(f"Completed. Matches: {matches} | Coverage: {met}/{total}")

if __name__ == "__main__":
    crawl()
