import sqlite3
import time
from collections import deque
import json
from riotwatcher import LolWatcher, ApiError

api_key = ''
watcher = LolWatcher(api_key)

routing_region = ''
platform_region = ''
min_matches_to_search = 70000  
min_apps_per_champ = 5

def load_champion_list():
    with open('champions.json', 'r') as f:
        return json.load(f)

def get_team_by_roles(participants, team_id):
    roles = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']
    team_participants = [p for p in participants if p['teamId'] == team_id]
    role_map = {r: '' for r in roles}
    unmapped = []
    
    for p in team_participants:
        pos = p.get('teamPosition')
        if pos in role_map and not role_map[pos]:
            role_map[pos] = p['championName']
        else:
            unmapped.append(p['championName'])
            
    res = []
    for r in roles:
        if role_map[r]:
            res.append(role_map[r])
        elif unmapped:
            res.append(unmapped.pop(0))
        else:
            res.append('')
    return res

def get_current_coverage(cursor, all_champs, target_apps):
    champ_names = list(all_champs.values()) if isinstance(all_champs, dict) else list(all_champs)
    columns = ['blue_top', 'blue_jungle', 'blue_mid', 'blue_bot', 'blue_support',
               'red_top', 'red_jungle', 'red_mid', 'red_bot', 'red_support']
    query = " UNION ALL ".join([f"SELECT {col} AS champ FROM matches WHERE {col} != ''" for col in columns])
    cursor.execute(f"SELECT champ, COUNT(*) FROM ({query}) GROUP BY champ")
    counts = {row[0]: row[1] for row in cursor.fetchall()}
    total = len(champ_names)
    seen = sum(1 for c in champ_names if c in counts)
    met = sum(1 for c in champ_names if counts.get(c, 0) >= target_apps)
    return seen, met, total, (seen / total * 100), (met / total * 100)

def init_db():
    conn = sqlite3.connect('league_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS matches (
            match_id TEXT PRIMARY KEY,
            winning_team TEXT,
            blue_top TEXT, blue_jungle TEXT, blue_mid TEXT, blue_bot TEXT, blue_support TEXT,
            red_top TEXT, red_jungle TEXT, red_mid TEXT, red_bot TEXT, red_support TEXT
        )
    ''')
    cursor.execute('CREATE TABLE IF NOT EXISTS processed_players (puuid TEXT PRIMARY KEY)')
    conn.commit()
    return conn

def get_next_league_players(watcher, platform, state):
    tiers = ['PLATINUM', 'EMERALD', 'DIAMOND']
    divisions = ['I', 'II', 'III', 'IV']
    
    while state['tier_idx'] < len(tiers):
        tier = tiers[state['tier_idx']]
        division = divisions[state['div_idx']]
        page = state['page']
        
        print(f"Fetching league page: {tier} {division} Page {page}...")
        try:
            entries = watcher.league.entries(platform, 'RANKED_SOLO_5x5', tier, division, page=page)
            if not entries:
                state['page'] = 1
                state['div_idx'] += 1
                if state['div_idx'] >= len(divisions):
                    state['div_idx'] = 0
                    state['tier_idx'] += 1
                continue
                
            state['page'] += 1
            return entries
        except ApiError as e:
            if e.response.status_code in [401, 403]:
                print(f"\n[ERROR] API Key is invalid or expired (HTTP {e.response.status_code}).")
                print("Please regenerate your key on the Riot Developer Portal and update data_crawler.py.\n")
                raise e
            if e.response.status_code == 429:
                raise e
            print(f"Error fetching league entries: {e}")
            state['page'] += 1
            continue
    return []

def crawl():
    conn = init_db()
    cursor = conn.cursor()
    champions_master = load_champion_list()

    cursor.execute('SELECT COUNT(*) FROM matches')
    processed_matches = cursor.fetchone()[0]
    seen, met, total, p_seen, p_met = get_current_coverage(cursor, champions_master, min_apps_per_champ)
    print(f"Initial State: {processed_matches} matches")
    print(f"Coverage: {seen}/{total} seen ({p_seen:.1f}%), {met}/{total} at {min_apps_per_champ}+ games ({p_met:.1f}%)")

    league_state = {'tier_idx': 0, 'div_idx': 0, 'page': 1}
    player_entries = deque()

    while True:
        if processed_matches >= min_matches_to_search and (met >= total):
            print("\n!!! All Quality Data Goals Met !!!")
            break

        if not player_entries:
            try:
                entries = get_next_league_players(watcher, platform_region, league_state)
                if not entries:
                    print("No more players found in target leagues.")
                    break
                player_entries.extend(entries)
            except ApiError as e:
                if e.response.status_code == 429:
                    retry_after = int(e.response.headers.get('Retry-After', 60))
                    print(f"Rate limited on league entries! Sleeping for {retry_after}s...")
                    time.sleep(retry_after)
                continue

        entry = player_entries.popleft()
        puuid = entry['puuid']

        cursor.execute('SELECT 1 FROM processed_players WHERE puuid = ?', (puuid,))
        if cursor.fetchone():
            continue

        try:
            match_ids = watcher.match.matchlist_by_puuid(routing_region, puuid, count=20, queue=420)
            for match_id in match_ids:
                if processed_matches >= min_matches_to_search and (met >= total):
                    break
                
                cursor.execute('SELECT 1 FROM matches WHERE match_id = ?', (match_id,))
                if cursor.fetchone():
                    continue

                match = watcher.match.by_id(routing_region, match_id)
                if match['info']['gameDuration'] < 900:
                    continue

                participants = match['info']['participants']
                winner = "BLUE_WIN" if next(t for t in match['info']['teams'] if t['teamId'] == 100)['win'] else "RED_WIN"
                blue_team = get_team_by_roles(participants, 100)
                red_team = get_team_by_roles(participants, 200)

                cursor.execute('''
                    INSERT OR IGNORE INTO matches 
                    (match_id, winning_team, blue_top, blue_jungle, blue_mid, blue_bot, blue_support,
                     red_top, red_jungle, red_mid, red_bot, red_support)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (match_id, winner, *blue_team, *red_team))
                
                processed_matches += 1
                conn.commit()
                
                seen, met, total, p_seen, p_met = get_current_coverage(cursor, champions_master, min_apps_per_champ)
                print(f"[{processed_matches}/{min_matches_to_search}+] Saved: {match_id} | Coverage: {met}/{total} met target {min_apps_per_champ} games ({p_met:.1f}%)")

            cursor.execute('INSERT OR IGNORE INTO processed_players (puuid) VALUES (?)', (puuid,))
            conn.commit()

        except ApiError as e:
            if e.response.status_code == 429:
                player_entries.appendleft(entry)
                retry_after = int(e.response.headers.get('Retry-After', 60))
                print(f"Rate limited on matches. Sleeping for {retry_after}s...")
                time.sleep(retry_after)

    conn.close()
    print(f"  - Total Matches: {processed_matches}")
    print(f"  - Champions Meeting {min_apps_per_champ} Game Minimum: {met}/{total} ({p_met:.1f}%)")

if __name__ == "__main__":
    crawl()
