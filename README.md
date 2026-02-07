# ⚽ Project Trident: Football Scouting Tool

Το **Project Trident** είναι ένα εργαλείο Data Scouting γραμμένο σε Python. Σκοπός του είναι να ξεπεράσει τον απλό διαχωρισμό θέσεων (FW, MF) και να αναγνωρίσει το **στυλ παιχνιδιού** (Archetype) κάθε επιθετικού, χρησιμοποιώντας προηγμένα στατιστικά (Advanced Metrics) και αλγορίθμους ομοιότητας (Similarity Search).

## 🚀 Δυνατότητες (Features)

* **📊 Smart Role Classification:** Κατηγοριοποιεί τους παίκτες όχι με βάση τη θέση τους στο χαρτί, αλλά με βάση τα δεδομένα τους (Γκολ, Σουτ, Ασίστ, Ευστοχία).
* **🔍 Dual Similarity Engine:** Διαθέτει δύο αλγορίθμους (Cosine & Euclidean) για διαφορετικά use cases - βρες παίκτες με παρόμοιο style ή παρόμοια νούμερα.
* **📈 Advanced Metrics:** Υπολογίζει και αναλύει metrics όπως `G/Sh` (Efficiency), `Sh/90` (Volume), `SoT%` (Accuracy) και `Ast/90` (Creativity).
* **⚖️ Weighted Scoring System:** Διαφορετικά βάρη ανά archetype - ένας Killer Striker αξιολογείται διαφορετικά από έναν Shadow Striker.
* **🌍 League Adjustment:** Αυτόματη προσαρμογή στατιστικών με league difficulty coefficients (Top 5 Leagues).
* **⚡ CLI Tool (sonar.py):** Production-ready command-line interface με professional output formatting.

---

## 🧠 Η Λογική των Ρόλων (The Logic)

Ο αλγόριθμος αναλύει τα στατιστικά κάθε παίκτη και του αναθέτει έναν από τους παρακάτω ρόλους:

| Ρόλος (Archetype) | Emoji | Περιγραφή & Κριτήρια |
| :--- | :---: | :--- |
| **Killer Striker** | 💀 | Το κλασικό "9άρι". Υψηλή ευστοχία (`G/Sh > 0.15`) και καλά τελειώματα, χωρίς απαραίτητα πολλές επαφές. |
| **Elite Striker** | 🎯 | Ο ολοκληρωμένος σκόρερ. Συνδυάζει μεγάλο όγκο σουτ (`Sh/90 > 3.0`) με πολλά γκολ (`> 5`). |
| **Shadow Striker** | 👻 | Δημιουργικός επιθετικός/10άρι. Υψηλές ασίστ (`Ast/90 > 0.19`) και καλή τεχνική. |
| **Winger / Inside Forward** | 🚀 | Εξτρέμ που συγκλίνει και εκτελεί. Υψηλός αριθμός σουτ (`> 2.8`) και καλή ακρίβεια. |
| **Attacking Winger** | ⚡ | Ταχύς εξτρέμ με έμφαση στην επίθεση και την απειλή (`Sh/90 > 2.5`). |
| **Supporting Winger** | 🏹 | Εξτρέμ που παίζει για την ομάδα, με πιο ισορροπημένα στατιστικά. |
| **Support Striker** | 🔗 | Ο συνδετικός κρίκος. Επιθετικός που βοηθάει στην ανάπτυξη του παιχνιδιού. |

---

## 🛠️ Πώς λειτουργεί (Technical Overview)

1.  **Data Ingestion:** Το σύστημα διαβάζει και ενώνει (Merge) αρχεία `Standard Stats` και `Shooting Stats` (από FBref).
2.  **Preprocessing:** Καθαρισμός δεδομένων, υπολογισμός Per 90 metrics και handling ελλιπών τιμών (NaN).
3.  **Role Definition:** Ένας `Rule-Based Classifier` εξετάζει τα metrics και αναθέτει το κατάλληλο "Label" σε κάθε παίκτη.
4.  **Similarity Search:**
    * Τα δεδομένα κανονικοποιούνται (MinMaxScaling) στο εύρος 0-1.
    * Εφαρμόζεται αλγόριθμος `Euclidean Distance` για να μετρηθεί η "απόσταση" μεταξύ των παικτών.
    * Το σύστημα επιστρέφει λίστα με τους παίκτες που έχουν τη μικρότερη απόσταση (μεγαλύτερη ομοιότητα %).

---

## 🎮 Τρόποι Χρήσης (Usage Modes)

Το Project Trident διαθέτει **δύο διεπαφές** για διαφορετικές ανάγκες:

### 📓 **Jupyter Notebook** (`trident_project.ipynb`)
Η **διαδραστική έκδοση** για exploratory analysis και visualization:
* Ideal για ανάλυση δεδομένων και πειραματισμό
* Βλέπεις όλα τα βήματα της επεξεργασίας
* Step-by-step execution

### ⚡ **Sonar.py** (Command-Line Tool)
Το **production-ready CLI tool** με προηγμένες δυνατότητες:

#### 🔬 Dual Algorithm Engine
* **Cosine Similarity:** Συγκρίνει playing style (αναλογίες performance)
  * *Παράδειγμα:* Βρίσκει παίκτες με παρόμοιο προφίλ (20G/10A ≈ 18G/9A)
  * Best for: Scouting παικτών με ίδιο DNA
  
* **Euclidean Distance:** Συγκρίνει raw statistics
  * *Παράδειγμα:* Βρίσκει παίκτες με παρόμοια νούμερα (20G/10A → 19-21G / 9-11A)
  * Best for: Direct replacements

#### 🎯 Advanced Features
* **Weighted Scoring System:** Διαφορετικά βάρη ανά archetype (Killer Striker ≠ Shadow Striker)
* **League Difficulty Adjustment:** Αυτόματη προσαρμογή στατιστικών με league coefficients (Premier League: 1.0, Ligue 1: 0.89)
* **Team Goal Share Analysis:** Μετράει τη σημασία του παίκτη για την ομάδα του
* **Smart Search Engine:** Διαχείριση ομωνύμων και partial name matching
* **Beautiful Output:** Professional formatting με `tabulate` (emojis, colors, scores)

#### 📊 Sample Output
```
🔥 Excellent Match (85%+)
✅ Good Match (70-85%)
👍 Decent Match (60-70%)
⚪ Fair Match (<60%)
```

---

## 💻 Εγκατάσταση & Χρήση

### 📦 Απαιτούμενες Βιβλιοθήκες
```bash
pip install pandas scikit-learn numpy tabulate
```

### 🚀 Εκτέλεση

**Option 1: Jupyter Notebook** (Recommended για exploration)
```bash
jupyter notebook trident_project.ipynb
```

**Option 2: Command-Line Tool** (Recommended για production use)
```bash
python sonar.py
```

Το σύστημα θα σας ρωτήσει:
1. Ποιον αλγόριθμο θέλετε (Cosine / Euclidean)
2. Ποιον παίκτη ψάχνετε
3. Θα εμφανίσει τους 10 πιο παρόμοιους παίκτες με similarity scores

---

## 📂 Δομή Αρχείων

* **`trident_project.ipynb`:** Jupyter Notebook για interactive analysis
* **`sonar.py`:** Production-ready CLI tool με dual algorithm engine
* **`perfect_merge.csv`:** Η κεντρική βάση δεδομένων (FBref stats)
* **`README.md`:** Αυτό το αρχείο

---

## 🧬 Technical Deep Dive (Sonar.py Architecture)

### ⚖️ Weighted Scoring Logic
Κάθε archetype έχει **διαφορετικά βάρη** στα metrics:

| Metric | 💀 Killer | 🎯 Elite | 👻 Shadow | 🚀 Inside FW | 🏹 Support Wing |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Gls_Adj** | 2.0 | 1.9 | 1.4 | 1.6 | 0.8 |
| **G/Sh** | 1.9 | 1.6 | 1.2 | 1.5 | 0.9 |
| **Ast_Adj** | 0.3 | 0.5 | 1.8 | 1.0 | 1.6 |
| **SoT%** | 1.5 | 1.3 | 1.3 | 1.4 | 1.0 |

*Παράδειγμα:* Ένας Shadow Striker (10άρι) αξιολογείται περισσότερο για τις ασίστ (1.8x) παρά για τα γκολ (1.4x).

### 🌍 League Difficulty Coefficients
```python
Premier League: 1.00 (Baseline)
La Liga:        0.97
Serie A:        0.95
Bundesliga:     0.92
Ligue 1:        0.89
```
*Αποτέλεσμα:* 20 γκολ στη Ligue 1 = **17.8 adjusted goals** (20 × 0.89)

### 🎯 Use Cases

**🔍 Scouting Scenario:**
> *"Ψάχνω έναν επιθετικό που παίζει σαν τον **Haaland** αλλά είναι πιο φθηνός."*
* Επιλέγεις **Cosine Similarity** (playing style)
* Φιλτράρεις τα αποτελέσματα για ηλικία < 24 και λιγότερο competitive league

**🔄 Transfer Replacement:**
> *"Ο **Victor Osimhen** φεύγει. Ποιος έχει παρόμοια απόδοση;"*
* Επιλέγεις **Euclidean Distance** (raw stats)
* Ψάχνεις για Elite Strikers με 80%+ match score

---

## 🎓 Lessons Learned & Future Improvements

### ✅ What Works
* Rule-based classification είναι πιο explicable από ML clustering
* Weighted features >> Uniform features για role-based matching
* Dual algorithm approach δίνει flexibility στον user

### 🔧 Potential Upgrades
* [ ] Προσθήκη **xG/xA metrics** (Expected Goals/Assists)
* [ ] **Age-adjusted projections** (peak performance prediction)
* [ ] **Market Value integration** (Transfermarkt API)
* [ ] **GUI Interface** (Streamlit/Dash)
* [ ] **Radar Charts** για visual comparison

---

*Created by **Epameinondas11** for Football Analytics Project.*
