import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv('perfect_merge.csv')

# Φιλτράρισμα: Μόνο επιθετικοί (FW/MF)
df = df[df['Pos'].str.contains('FW|MF', na=False)].copy()

# Αποκλεισμός αμυντικών μέσων
df['Sh/90'] = pd.to_numeric(df['Sh/90'], errors='coerce').fillna(0)
df = df[(df['Pos'].str.contains('FW', na=False)) | 
        ((df['Pos'].str.contains('MF', na=False)) & (df['Sh/90'] >= 1))].copy()

# Υπολογισμός Ast_per_90
df['Ast_per_90'] = (df['Ast'] / df['Min']) * 90
df['Ast_per_90'] = df['Ast_per_90'].fillna(0)

# Καθαρισμός αριθμητικών στηλών
metrics = ['Sh/90', 'G/Sh', 'SoT%', 'Ast_per_90', 'G-PK']
for col in metrics:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        
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
        
# Εφαρμογή κατηγοριοποίησης
df['Role'] = df.apply(classify_player_role, axis=1)

# Φιλτράρισμα λεπτών
df_final = df[df['Min'] > 450].copy()
df_final = df_final.reset_index(drop=True)

print(f"✅ Δεδομένα έτοιμα: {len(df_final)} παίκτες")

