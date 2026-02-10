import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from tabulate import tabulate



# --- ⚙️ RATING CONFIGURATION ---
# Συντελεστές δυσκολίας πρωταθλημάτων (Top 5 Leagues)
league_weights = {
    'Premier League': 1.0,
    'La Liga': 0.97,
    'Serie A': 0.95,
    'Bundesliga': 0.92,
    'Ligue 1': 0.89
}



# --- 🧠 ROLE LOGIC ---

def classify_player_role(row):
    
    """
    Κατηγοριοποιεί τον παίκτη με βάση τα στατιστικά του.
    Βασισμένο στο Project Trident logic.
    """
    
    pos = str(row['Pos'])
    shots = row['Sh/90']
    efficiency = row['G/Sh']
    shot_accuracy = row['SoT%']
    assists = row['Ast_per_90']
    goals = row['Gls']
    
     # ΚΑΤΗΓΟΡΙΑ 1: ΕΞΤΡΕΜ + ΔΗΜΙΟΥΡΓΟΙ (FW + MF)
    
    if 'MF' in pos:
        # Shadow Striker
        if assists >= 0.19 and shot_accuracy >= 30:
            return '👻 Shadow Striker / Creator'
        
        elif shots >= 2.8 and shot_accuracy >= 35:
            return '🚀 Winger / Inside Forward'
        
        elif shots > 2.5:
            return '⚡ Winger (Attacking)'
        
        else:
            return '🏹 Supporting Winger'
        
    # 2. ΚΑΤΗΓΟΡΙΑ: FORWARDS
    else:
        if efficiency >= 0.15 and shot_accuracy > 35:
            return '💀 Killer Striker'
        elif shots >= 3.0 and goals >= 5:
            return '🎯 Elite Striker'
        elif shots >= 2.2:
            return '⚽ Striker'
        else:
            return '🔗 Support Striker'        
        


# --- ⚖️ FEATURE WEIGHTS (Η ΚΑΡΔΙΑ ΤΟΥ ΣΥΣΤΗΜΑΤΟΣ) ---

def get_weights_by_role(role):

    """
    Επιστρέφει τα βάρη (weights) ανάλογα με τον ρόλο.
    Προσαρμοσμένο στις στήλες που έχουμε (χωρίς xG/xA).
    """

    # Βασικά βάρη που ισχύουν για όλους
    weights = {
        'Team_Goal_Share': 1.5, # Πόσο σημαντικός είναι για την ομάδα του
        'League_Factor': 1.0
    }

    # 💀 KILLER STRIKER
    if 'Killer' in role:
        weights.update({
            'Gls_Adj': 2.0,
            'G/Sh': 1.9,
            'SoT%': 1.5,
            'G/SoT': 1.8,
            'Sh/90': 1.0,
            'Ast_Adj': 0.3
        })

    # 🎯 ELITE STRIKER
    elif 'Elite' in role:
        weights.update({
            'Gls_Adj': 1.9,
            'Sh/90': 1.4,
            'SoT%': 1.3,
            'G/Sh': 1.6,
            'G/SoT': 1.5,
            'Ast_Adj': 0.5
        })

    # ⚽ STRIKER
    elif role == '⚽ Striker':
        weights.update({
            'Gls_Adj': 1.7,
            'Sh/90': 1.3,
            'SoT%': 1.2,
            'G/Sh': 1.4,
            'G/SoT': 1.3,
            'Ast_Adj': 0.6
        })

     # 🔗 SUPPORT STRIKER
    elif 'Support Striker' in role:
        weights.update({
            'Gls_Adj': 1.2,
            'Ast_Adj': 1.4,
            'Sh/90': 0.9,
            'SoT%': 1.0,
            'G/Sh': 1.1,
            'G/SoT': 1.0
        })

    # 👻 SHADOW STRIKER / CREATOR
    elif 'Shadow Striker' in role or 'Creator' in role:
        weights.update({
            'Gls_Adj': 1.4,
            'Ast_Adj': 1.8,
            'SoT%': 1.3,
            'G/Sh': 1.2,
            'Sh/90': 1.1,
            'G/SoT': 1.1
        })

    # 🚀 WINGER / INSIDE FORWARD
    elif 'Inside Forward' in role:
        weights.update({
            'Gls_Adj': 1.6,
            'Sh/90': 1.5,
            'SoT%': 1.4,
            'G/Sh': 1.5,
            'Ast_Adj': 1.0,
            'G/SoT': 1.3
        })

    # ⚡ WINGER (ATTACKING)
    elif 'Winger (Attacking)' in role:
        weights.update({
            'Gls_Adj': 1.3,
            'Ast_Adj': 1.3,
            'Sh/90': 1.4,
            'SoT%': 1.2,
            'G/Sh': 1.2,
            'G/SoT': 1.1
        })

     # 🏹 SUPPORTING WINGER
    elif 'Supporting Winger' in role:
        weights.update({
            'Gls_Adj': 0.8,
            'Ast_Adj': 1.6,
            'Sh/90': 0.9,
            'SoT%': 1.0,
            'G/Sh': 0.9,
            'G/SoT': 0.9
        })

    # Default (fallback)
    else:
        weights.update({
            'Gls_Adj': 1.0,
            'Ast_Adj': 1.0,
            'Sh/90': 1.0,
            'SoT%': 1.0,
            'G/Sh': 1.0,
            'G/SoT': 1.0
        })
    
    return weights



# --- 🧹 DATA PREPARATION & FEATURE ENGINEERING ---

def load_and_prep_data(df):

    print("⏳ Loading Database...")
    try:
        df = pd.read_csv(df)
    except FileNotFoundError:
        print("❌ Error: Το αρχείο csv δεν βρέθηκε.")
        return None
    
    print(f"📦 Loaded {len(df)} players")

    # 1️⃣ ΦΙΛΤΡΑΡΙΣΜΑ ΘΕΣΕΩΝ
    print("🎯 Filtering positions (FW/MF only)...")
    df = df[df['Pos'].str.contains('FW|MF', na=False)].copy()
    
    # 2️⃣ ΚΑΘΑΡΙΣΜΟΣ & ΜΕΤΑΤΡΟΠΗ ΣΕ ΑΡΙΘΜΟΥΣ
    cols_to_fix = ['Gls', 'Ast', 'Sh', 'SoT', 'SoT%', 'Sh/90', 'G/Sh', 
                   'G/SoT', 'G-PK', 'PK', 'PKatt', 'Min']
    
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 3️⃣ ΥΠΟΛΟΓΙΣΜΟΣ Ast_per_90
    df['Ast_per_90'] = ((df['Ast'] / df['Min']) * 90).fillna(0)
    
    # 4️⃣ ΑΠΟΚΛΕΙΣΜΟΣ ΑΜΥΝΤΙΚΩΝ ΜΕΣΩΝ
    print("🚫 Excluding defensive midfielders(DF)...")
    df = df[(df['Pos'].str.contains('FW', na=False)) | 
            ((df['Pos'].str.contains('MF', na=False)) & (df['Sh/90'] >= 1))].copy()
    
    # 5️⃣ TEAM GOAL SHARE
    team_goals = df.groupby('Squad')['Gls'].transform('sum')
    df['Team_Goal_Share'] = (df['Gls'] / team_goals).fillna(0)
    df['Team_Goal_Share_NoPK'] = ((df['Gls'] - df['PK']) / team_goals).fillna(0)
    
    # 6️⃣ LEAGUE ADJUSTMENT
    print("🌍 Applying league difficulty factors...")
    if 'Comp' in df.columns:
        df['League_Clean'] = df['Comp'].str.replace(r'^[a-z]{2,3}\s+', '', regex=True)
        df['League_Factor'] = df['League_Clean'].map(league_weights).fillna(0.75)
    else:
        df['League_Factor'] = 1.0
    
    # 7️⃣ ADJUSTED STATS
    df['Gls_Adj'] = df['Gls'] * df['League_Factor']
    df['Ast_Adj'] = df['Ast'] * df['League_Factor']
    df['G+A_Adj'] = (df['Gls'] + df['Ast']) * df['League_Factor']
    df['Gls_NoPK_Adj'] = (df['Gls'] - df['PK']) * df['League_Factor']
    
    # 8️⃣ ROLE CLASSIFICATION
    df['Role'] = df.apply(classify_player_role, axis=1)
    
    # 9️⃣ MINUTES FILTERING
    df_final = df[df['Min'] > 450].copy()
    df_final = df_final.reset_index(drop=True)
    
    # 🔟 QUALITY CHECK
    print("\n" + "="*50)
    print(f"✅ Data Ready: {len(df_final)} players")
    print(f"📊 Roles Distribution:")
    print(df_final['Role'].value_counts())
    if 'League_Clean' in df_final.columns:
        print(f"🌍 Leagues: {df_final['League_Clean'].nunique()}")
    print("="*50 + "\n")
    
    return df_final



# --- 🔍 SIMILARITY SEARCH & ALGORITHM ---

def find_similar_players(df, n_neighbors=10):
    """
    Διαδραστική αναζήτηση + εύρεση παρόμοιων παικτών (CLI VERSION).
    Χρησιμοποιεί input() για command line interface.
    """

    # ========== ΕΠΙΛΟΓΗ ΑΛΓΟΡΙΘΜΟΥ ==========
    print("\n" + "="*80)
    print("🔬 ΕΠΙΛΟΓΗ ΑΛΓΟΡΙΘΜΟΥ ΣΥΓΚΡΙΣΗΣ")
    print("="*80)
    print("\n📊 Διαθέσιμοι αλγόριθμοι:\n")
    print("🅲  COSINE SIMILARITY")
    print("   → Συγκρίνει το playing style (αναλογίες, όχι απόλυτα νούμερα)")
    print("   → Καλύτερος για εύρεση παικτών με παρόμοιο προφίλ")
    print("   → Παράδειγμα: 20G/10A στη Bundesliga ≈ 18G/9A στη Premier League\n")
    
    print("🅴  EUCLIDEAN DISTANCE")
    print("   → Συγκρίνει απόλυτες αποστάσεις (raw numbers)")
    print("   → Καλύτερος για εύρεση παικτών με παρόμοια στατιστικά")
    print("   → Παράδειγμα: 20G/10A βρίσκει 19-21G / 9-11A\n")
    
    print("="*80)
    
    while True:
        choice = input("👉 Επίλεξε αλγόριθμο (C για Cosine, E για Euclidean): ").strip().upper()
        
        if choice == 'C':
            metric = 'cosine'
            print("✅ Επιλέχθηκε: Cosine Similarity (Playing Style Matching)\n")
            break
        elif choice == 'E':
            metric = 'euclidean'
            print("✅ Επιλέχθηκε: Euclidean Distance (Stats Matching)\n")
            break
        else:
            print("❌ Μη έγκυρη επιλογή. Πάτησε C ή E.")


    # ========== ΜΕΡΟΣ 1: SEARCH ENGINE ==========
    while True:
        query = input("🔎 Ποιον παίκτη ψάχνεις; (Γράψε 'exit' για έξοδο): ").strip()
        
        if query.lower() == 'exit':
            print("👋 Έξοδος από το πρόγραμμα.")
            return None, None
        
        matches = df[df['Player'].str.contains(query, case=False, na=False)]
        count_matches = len(matches)
        
        # ❌ ΠΕΡΙΠΤΩΣΗ 1: ΔΕΝ ΒΡΕΘΗΚΕ ΚΑΝΕΙΣ
        if count_matches == 0:
            print(f"❌ Δεν βρέθηκε κανένας παίκτης με το όνομα '{query}'.")
            print("💡 Tip: Δοκίμασε μόνο το επίθετο (π.χ. 'Haaland')")
            continue
        
        # 👥 ΠΕΡΙΠΤΩΣΗ 2: ΒΡΕΘΗΚΑΝ ΠΟΛΛΟΙ
        elif count_matches > 1:
            print(f"\n👥 Βρέθηκαν {count_matches} παίκτες με το όνομα '{query}':")
            print("=" * 80)
            
            matches = matches.reset_index(drop=True)
            
            for idx, row in matches.iterrows():
                league = row.get('League_Clean', row.get('Comp', 'N/A'))
                role = row.get('Role', 'N/A')
                goals = row.get('Gls', 0)
                assists = row.get('Ast', 0)
                
                print(f"{idx}. {row['Player']:<25} | {row['Squad']:<20} | {league:<15}")
                print(f"   {role:<30} | ⚽ {goals}G + {assists}A | 🎂 {row['Age']} ετών")
                print("-" * 80)
            
            try:
                selection = int(input(f"\n👉 Διάλεξε αριθμό (0-{count_matches-1}): "))
                
                if 0 <= selection < count_matches:
                    target = matches.iloc[selection]
                    print(f"\n✅ Επέλεξες: {target['Player']} ({target['Squad']}) - {target.get('Role', 'N/A')}")
                    break
                else:
                    print("❌ Μη έγκυρος αριθμός. Δοκίμασε ξανά.")
                    continue
                    
            except ValueError:
                print("❌ Πρέπει να δώσεις αριθμό!")
                continue
        
        # ✅ ΠΕΡΙΠΤΩΣΗ 3: ΒΡΕΘΗΚΕ ΕΝΑΣ
        else:
            target = matches.iloc[0]
            league = target.get('League_Clean', target.get('Comp', 'N/A'))
            role = target.get('Role', 'N/A')
            
            print(f"\n✅ Βρέθηκε: {target['Player']}")
            print(f"   🏟️  Ομάδα: {target['Squad']}")
            print(f"   🌍 Πρωτάθλημα: {league}")
            print(f"   🎯 Ρόλος: {role}")
            print(f"   ⚽ Stats: {target.get('Gls', 0)}G + {target.get('Ast', 0)}A")
            
            confirm = input("\n✔️  Είναι ο σωστός; (y/n): ").strip().lower()
            if confirm == 'y':
                break
            else:
                print("🔄 Δοκίμασε ξανά...")
                continue

    # ========== ΜΕΡΟΣ 2: SIMILARITY ALGORITHM ==========
    metric_name = "Cosine Similarity" if metric == 'cosine' else "Euclidean Distance"
    print(f"\n⚙️ Calculating similarities using {metric_name}...")
    
    role = target['Role']
    weights = get_weights_by_role(role)
    print(f"📊 Using weights for role: {role}")
    
    # Προετοιμασία Feature Data
    feature_data = pd.DataFrame()
    final_weights = {}
    
    for feat, w in weights.items():
        if feat in df.columns:
            feature_data[feat] = df[feat]
            final_weights[feat] = w
    
    if feature_data.empty:
        print("❌ Error: Δεν βρέθηκαν κοινές στήλες μεταξύ weights και data.")
        return None, None
    
    print(f"✅ Using {len(final_weights)} features: {list(final_weights.keys())}")
    
    # Normalization
    scaler = MinMaxScaler()
    scaled_data = pd.DataFrame(
        scaler.fit_transform(feature_data), 
        columns=feature_data.columns,
        index=df.index
    )
    
    # Apply Weights
    for col, w in final_weights.items():
        scaled_data[col] = scaled_data[col] * w
    
    # ✅ K-Nearest Neighbors με επιλεγμένο metric
    model = NearestNeighbors(n_neighbors=n_neighbors+1, metric=metric)
    model.fit(scaled_data)
    
    # Βρίσκουμε το index του target
    target_matches = df[(df['Player'] == target['Player']) & 
                        (df['Squad'] == target['Squad'])]
    
    if len(target_matches) == 0:
        target_matches = df[df['Player'] == target['Player']]
    
    target_idx = target_matches.index[0]
    
    # Εύρεση παρόμοιων
    distances, indices = model.kneighbors(scaled_data.loc[[target_idx]])
    
    # Αποτελέσματα (χωρίς τον ίδιο)
    similar_indices = indices[0][1:]
    similar_distances = distances[0][1:]
    
    results = df.iloc[similar_indices].copy()
    
    # ✅ Διαφορετικός υπολογισμός score ανά αλγόριθμο
    if metric == 'cosine':
        # Cosine: 0 = ίδιοι, 1 = αντίθετοι → μετατροπή σε %
         results['Similarity_Score'] = (1 - similar_distances) ** 0.5 * 100
    else:  # euclidean
        # Euclidean: 0 = ίδιοι, μεγάλη τιμή = διαφορετικοί
        # Χρήση median αντί για mean (πιο robust σε outliers)
        median_distance = np.median(similar_distances)
        
        if median_distance > 0:
            results['Similarity_Score'] = 100 * np.exp(-similar_distances / (median_distance * 1.5))
        else:
            results['Similarity_Score'] = 100
    
    results['Similarity_Score'] = results['Similarity_Score'].clip(0, 100)
    
    return target, results



# --- 🔍 SIMILARITY SEARCH (API VERSION για Streamlit) ---

def find_similar_players_gui(df, target_player, algorithm='cosine', n_neighbors=10):
    
    """
    Βρίσκει παρόμοιους παίκτες (χωρίς input() - για Streamlit/API).
    
    Args:
        df (DataFrame): Το prepared dataframe με παίκτες
        target_player (Series): Η γραμμή του παίκτη που ψάχνουμε (df.iloc[x])
        algorithm (str): 'cosine' ή 'euclidean'
        n_neighbors (int): Πόσους παρόμοιους να βρει
    
    Returns:
        DataFrame: Παρόμοιοι παίκτες με Similarity_Score στήλη
    """
    metric = algorithm
    role = target_player['Role']
    weights = get_weights_by_role(role)
    
    # Προετοιμασία Feature Data
    feature_data = pd.DataFrame()
    final_weights = {}
    
    for feat, w in weights.items():
        if feat in df.columns:
            feature_data[feat] = df[feat]
            final_weights[feat] = w
            
    if feature_data.empty:
        return None
    
    # Normalization
    scaler = MinMaxScaler()
    scaled_data = pd.DataFrame(
        scaler.fit_transform(feature_data), 
        columns=feature_data.columns,
        index=df.index
    )
    
    # Apply Weights
    for col, w in final_weights.items():
        scaled_data[col] = scaled_data[col] * w
        
    # K-Nearest Neighbors
    model = NearestNeighbors(n_neighbors=n_neighbors+1, metric=metric)
    model.fit(scaled_data)
    
    # Βρίσκουμε το index του target
    target_matches = df[
        (df['Player'] == target_player['Player']) &
        (df['Squad'] == target_player['Squad'])
    ]
    
    if len(target_matches) == 0:
        target_matches = df[df['Player'] == target_player['Player']]
        
    if len(target_matches) == 0:
        return None
    
    target_idx = target_matches.index[0]
    
    # Εύρεση παρόμοιων
    distances, indices = model.kneighbors(scaled_data.loc[[target_idx]])
    
    # Αποτελέσματα (χωρίς τον ίδιο)
    similar_indices = indices[0][1:]
    similar_distances = distances[0][1:]
    
    results = df.iloc[similar_indices].copy()
    
    # Διαφορετικός υπολογισμός score ανά αλγόριθμο
    if metric == 'cosine':
        results['Similarity_Score'] = (1 - similar_distances) ** 0.5 * 100
    else:  # euclidean
        median_distance = np.median(similar_distances)
        
        if median_distance > 0:
            results['Similarity_Score'] = 100 * np.exp(-similar_distances / (median_distance * 1.5))
        else:
            results['Similarity_Score'] = 100
            
    results['Similarity_Score'] = results['Similarity_Score'].clip(0, 100)
    
    return results



# --- 🚀 MAIN APP ---

if __name__ == "__main__":

    # 1️⃣ Φόρτωση δεδομένων
    df_final = load_and_prep_data('perfect_merge.csv')
    
    if df_final is not None:
        # 2️⃣ Εύρεση παρόμοιων παικτών (CLI VERSION)
        target_player, similar_players = find_similar_players(df_final, n_neighbors=10)
        
        if target_player is not None and similar_players is not None:
            
            # 3️⃣ Δημιουργία πίνακα αποτελεσμάτων
            print("\n" + "="*100)
            print(f"🎯 TARGET PLAYER: {target_player['Player']}")
            print(f"   📍 Squad: {target_player['Squad']} | 🌍 League: {target_player.get('League_Clean', 'N/A')}")
            print(f"   🎭 Role: {target_player['Role']}")
            print(f"   ⚽ Stats: {target_player['Gls']:.0f} G + {target_player['Ast']:.0f} A | 🎂 Age: {target_player['Age']}")
            print("="*100)
            
            # Προετοιμασία δεδομένων για πίνακα
            table_data = []
            
            for idx, (_, row) in enumerate(similar_players.iterrows(), 1):
                score = row['Similarity_Score']
                league = row.get('League_Clean', 'N/A')
                
                # Emojis για το score
                if score >= 85:
                    match_emoji = "🔥"
                elif score >= 70:
                    match_emoji = "✅"
                elif score >= 60:
                    match_emoji = "👍"
                else:
                    match_emoji = "⚪"
                
                table_data.append([
                    idx,
                    row['Player'][:22],
                    row['Squad'][:18],
                    league[:12],
                    row['Role'][:25],
                    f"{match_emoji} {score:.1f}%",
                    f"{int(round(row['Gls']))}",
                    f"{int(round(row['Ast']))}",
                    f"{int(row['Age'])}"
                ])
            
            # Κεφαλίδες πίνακα
            headers = [
                "#",
                "Player",
                "Squad",
                "League",
                "Role",
                "Match",
                "Goals",
                "Assists",
                "Age"
            ]
            
            # Εμφάνιση με tabulate
            print("\n📊 SIMILAR PLAYERS:\n")
            print(tabulate(
                table_data,
                headers=headers,
                tablefmt="fancy_grid",
                numalign="center",
                stralign="left"
            ))
            
            print("\n" + "="*100)
            print("💡 Legend: 🔥 Excellent (85%+) | ✅ Good (70-85%) | 👍 Decent (60-70%) | ⚪ Fair (<60%)")
            print("="*100)