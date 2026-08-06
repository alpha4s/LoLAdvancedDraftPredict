import sqlite3, time, json
from collections import deque
from riotwatcher import LolWatcher, ApiError
from config import DB_PATH, CHAMPIONS_PATH, VALID_RIOT_ROLES

api_key = 'RGAPI-your-key-here'
if 'your-key-here' in api_key:
    raise ValueError("Paste your Riot API key into data_crawler.py line 6 before running.")

routing_region = 'americas'
platform_region = 'na1'
TARGET_PATCH = ''
watcher = LolWatcher(api_key)
min_matches = 700000
min_apps = 100
VALID_ROLES = VALID_RIOT_ROLES

def load_champions():
    with open(CHAMPIONS_PATH) as file:
        return json.load(file)

def get_team_by_roles(match_players, team_id):
    team_members = []
    for member in match_players:
        if member['teamId'] == team_id:
            team_members.append(member)
    if len(team_members) != 5: return None, None
    role_map, stats_map = {}, {}
    for member in team_members:
        pos = member.get('teamPosition')
        if pos not in VALID_ROLES or pos in role_map:
            return None, None  # Drop match if role is invalid or duplicated to prevent label corruption
        role_map[pos] = member['championName']
        stats_map[pos] = (int(member.get('physicalDamageDealtToChampions', 0)), int(member.get('magicDamageDealtToChampions', 0)))
    
    champs = [role_map['TOP'], role_map['JUNGLE'], role_map['MIDDLE'], role_map['BOTTOM'], role_map['UTILITY']]
    dmgs = []
    for role in ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']:
        phys, magic = stats_map[role]
        dmgs.extend([phys, magic])
    return champs, dmgs

def get_coverage(cursor, champions):
    cols = ['blue_top',
            'blue_jungle',
            'blue_mid', 
            'blue_bot', 
            'blue_support',
            'red_top', 
            'red_jungle', 
            'red_mid', 
            'red_bot', 
            'red_support']
    query = " UNION ALL ".join([f"SELECT {col} AS champ FROM matches WHERE {col} != ''" for col in cols])
    cursor.execute(f"SELECT champ, COUNT(*) FROM ({query}) GROUP BY champ")
    counts = dict(cursor.fetchall())
    champs = list(champions.values()) if isinstance(champions, dict) else list(champions)
    met = 0
    for champ in champs:
        if counts.get(champ, 0) >= min_apps:
            met += 1
    return len(champs), met

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS matches (
        match_id TEXT PRIMARY KEY,
        game_version TEXT, 
        winning_team TEXT,
        blue_top TEXT, 
        blue_jungle TEXT, 
        blue_mid TEXT, 
        blue_bot TEXT, 
        blue_support TEXT,
        red_top TEXT, 
        red_jungle TEXT, 
        red_mid TEXT, 
        red_bot TEXT, 
        red_support TEXT)''')
    cursor.execute('CREATE TABLE IF NOT EXISTS processed_players (puuid TEXT PRIMARY KEY)')
    
    cursor.execute("PRAGMA table_info(matches)")
    existing_cols = []
    for col_info in cursor.fetchall():
        existing_cols.append(col_info[1])
    
    all_target_cols = [
        'match_id', 'game_version', 'winning_team',
        'blue_top', 'blue_jungle', 'blue_mid', 'blue_bot', 'blue_support',
        'red_top', 'red_jungle', 'red_mid', 'red_bot', 'red_support',
        'blue_top_phys', 'blue_top_magic', 'blue_jungle_phys', 'blue_jungle_magic',
        'blue_mid_phys', 'blue_mid_magic', 'blue_bot_phys', 'blue_bot_magic',
        'blue_supp_phys', 'blue_supp_magic',
        'red_top_phys', 'red_top_magic', 'red_jungle_phys', 'red_jungle_magic',
        'red_mid_phys', 'red_mid_magic', 'red_bot_phys', 'red_bot_magic',
        'red_supp_phys', 'red_supp_magic'
    ]
    
    for col in all_target_cols:
        if col not in existing_cols:
            cursor.execute(f"ALTER TABLE matches ADD COLUMN {col} INTEGER DEFAULT 0")
            
    conn.commit()
    conn.close()

def safe_api_call(func, *args, **kwargs):
    try:
        res = func(*args, **kwargs)
        time.sleep(0.05)
        return res
    except ApiError as err:
        if err.response.status_code == 429:
            print("[RATE LIMIT] Waiting 120s for Riot API rate limit reset...")
            time.sleep(120)
        return None
    except Exception as err:
        print(f"[NETWORK ERROR] {err}")
        time.sleep(2)
        return None

def main():
    init_db()
    champions = load_champions()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(*) FROM matches')
    match_count = cursor.fetchone()[0]
    total, met = get_coverage(cursor, champions)
    print(f"[START] Current DB: {match_count}/{min_matches} matches | Coverage: {met}/{total} champs met min {min_apps} games")

    cursor.execute('SELECT puuid FROM processed_players')
    seen_puuids = set(row[0] for row in cursor.fetchall())

    player_queue = deque()

    print("[SEED] Fetching fresh seed players from Diamond league...")
    entries = safe_api_call(watcher.league.entries, platform_region, 'RANKED_SOLO_5x5', 'DIAMOND', 'I', page=1) or []
    for entry in entries:
        puuid = entry.get('puuid')
        if puuid and puuid not in seen_puuids:
            player_queue.append(puuid)
            seen_puuids.add(puuid)
    print(f"[SEED] Queued {len(player_queue)} seed players.")

    if not player_queue:
        cursor.execute('SELECT puuid FROM processed_players ORDER BY RANDOM() LIMIT 200')
        for row in cursor.fetchall():
            player_queue.append(row[0])
        print(f"[RESUME] Loaded {len(player_queue)} active PUUIDs from DB.")

    global TARGET_PATCH
    versions = safe_api_call(watcher.data_dragon.versions_all)
    if versions:
        TARGET_PATCH = '.'.join(versions[0].split('.')[:2])
        print(f"[PATCH] Auto-detected current live patch: {TARGET_PATCH} (Full version: {versions[0]})")

    patch_start_time = int(time.time()) - (3 * 86400)

    while player_queue and match_count < min_matches:
        curr_puuid = player_queue.popleft()

        match_ids = safe_api_call(watcher.match.matchlist_by_puuid, routing_region, curr_puuid, count=10, type='ranked', start_time=patch_start_time)
        if not match_ids:
            continue

        for m_id in match_ids:
            if match_count >= min_matches: break

            cursor.execute('SELECT 1 FROM matches WHERE match_id = ?', (m_id,))
            if cursor.fetchone(): continue

            mdata = safe_api_call(watcher.match.by_id, routing_region, m_id)
            if not mdata:
                continue

            info = mdata.get('info', {})
            if info.get('queueId') != 420: continue

            game_version = info.get('gameVersion', '')
            if TARGET_PATCH and not game_version.startswith(TARGET_PATCH):
                continue

            participants = info.get('participants', [])
            b_champs, b_dmgs = get_team_by_roles(participants, 100)
            r_champs, r_dmgs = get_team_by_roles(participants, 200)

            if not b_champs or not r_champs: continue

            teams = info.get('teams', [])
            blue_win = teams[0].get('win', False) if teams else False
            winning_team = 'BLUE_WIN' if blue_win else 'RED_WIN'

            row_data = [m_id, game_version, winning_team] + b_champs + r_champs + b_dmgs + r_dmgs
            
            sql_insert = '''INSERT OR IGNORE INTO matches (
                match_id, game_version, winning_team,
                blue_top, blue_jungle, blue_mid, blue_bot, blue_support,
                red_top, red_jungle, red_mid, red_bot, red_support,
                blue_top_phys, blue_top_magic, blue_jungle_phys, blue_jungle_magic,
                blue_mid_phys, blue_mid_magic, blue_bot_phys, blue_bot_magic,
                blue_supp_phys, blue_supp_magic,
                red_top_phys, red_top_magic, red_jungle_phys, red_jungle_magic,
                red_mid_phys, red_mid_magic, red_bot_phys, red_bot_magic,
                red_supp_phys, red_supp_magic
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'''
            
            cursor.execute(sql_insert, row_data)
            conn.commit()

            match_count += 1
            print(f"[{match_count}/{min_matches}] Saved {m_id} | Patch: {game_version}")

            for p in participants:
                p_puuid = p.get('puuid')
                if p_puuid and p_puuid not in seen_puuids:
                    seen_puuids.add(p_puuid)
                    player_queue.append(p_puuid)
                    cursor.execute('INSERT OR IGNORE INTO processed_players (puuid) VALUES (?)', (p_puuid,))
            conn.commit()

            if match_count % 50 == 0:
                total, met = get_coverage(cursor, champions)
                print(f"[PROGRESS] Matches: {match_count}/{min_matches} | Coverage: {met}/{total} champs met min {min_apps} games")

    conn.close()
    print("[COMPLETE] Data crawler finished successfully.")

if __name__ == '__main__':
    main()
