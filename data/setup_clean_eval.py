"""
Setup script for clean, partitioned train and held-out test benchmark datasets.
Includes:
- data/fever/train_claims.json (60 claims: 20 SUPPORTED, 20 REFUTED, 20 NEI)
- data/fever/test_claims.json (30 claims: 10 SUPPORTED, 10 REFUTED, 10 NEI) [Held-out]
- data/liar/political_test_claims.json (6 claims) [Political stress test]
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent

# =====================================================================
# 1. TRAINING CLAIMS POOL (60 claims: 20 SUP, 20 REF, 20 NEI)
# Used exclusively to extract empirical features and train the MLP head / LR.
# =====================================================================
TRAIN_CLAIMS = [
    # --- SUPPORTED (20) ---
    {"id": "tr_sup_01", "claim": "Marie Curie was the first woman to win a Nobel Prize and the first person to win Nobel Prizes in two different scientific fields.", "label": "SUPPORTED", "category": "Science"},
    {"id": "tr_sup_02", "claim": "Neil Armstrong and Buzz Aldrin landed the Apollo 11 lunar module Eagle on the Moon in July 1969.", "label": "SUPPORTED", "category": "Space"},
    {"id": "tr_sup_03", "claim": "Alexander Fleming discovered penicillin in September 1928 at St. Mary's Hospital in London.", "label": "SUPPORTED", "category": "Medicine"},
    {"id": "tr_sup_04", "claim": "Canberra was chosen as the capital city of Australia in 1908 as a compromise between Sydney and Melbourne.", "label": "SUPPORTED", "category": "Geography"},
    {"id": "tr_sup_05", "claim": "Albert Einstein received the 1921 Nobel Prize in Physics for his discovery of the law of the photoelectric effect.", "label": "SUPPORTED", "category": "Physics"},
    {"id": "tr_sup_06", "claim": "Tim Berners-Lee invented the World Wide Web in 1989 while employed at CERN in Switzerland.", "label": "SUPPORTED", "category": "Technology"},
    {"id": "tr_sup_07", "claim": "Nelson Mandela served as the first president of post-apartheid South Africa from 1994 to 1999.", "label": "SUPPORTED", "category": "Politics"},
    {"id": "tr_sup_08", "claim": "The United States Constitution was drafted and signed at Independence Hall in Philadelphia in September 1787.", "label": "SUPPORTED", "category": "History"},
    {"id": "tr_sup_09", "claim": "Max Planck won the 1918 Nobel Prize in Physics in recognition of his discovery of energy quanta.", "label": "SUPPORTED", "category": "Physics"},
    {"id": "tr_sup_10", "claim": "The Treaty of Rome was signed in 1957, establishing the European Economic Community among six founding nations.", "label": "SUPPORTED", "category": "History"},
    {"id": "tr_sup_11", "claim": "Apollo 12 achieved a precision landing in the Ocean of Storms in November 1969, guided by Pete Conrad and Alan Bean.", "label": "SUPPORTED", "category": "Space"},
    {"id": "tr_sup_12", "claim": "Alan Turing designed the electromechanical Bombe machine at Bletchley Park to decipher German Enigma ciphers during World War II.", "label": "SUPPORTED", "category": "Computing"},
    {"id": "tr_sup_13", "claim": "The United Nations Charter was signed in San Francisco in June 1945 following the conclusion of World War II.", "label": "SUPPORTED", "category": "Politics"},
    {"id": "tr_sup_14", "claim": "James Watson and Francis Crick proposed the double-helix structure of DNA in 1953 at the University of Cambridge.", "label": "SUPPORTED", "category": "Biology"},
    {"id": "tr_sup_15", "claim": "Howard Florey and Ernst Boris Chain developed methods for the mass production of penicillin in Oxford in 1940.", "label": "SUPPORTED", "category": "Medicine"},
    {"id": "tr_sup_16", "claim": "Guido van Rossum released the first version of the Python programming language in 1991.", "label": "SUPPORTED", "category": "Computing"},
    {"id": "tr_sup_17", "claim": "The Federal Reserve System was created in 1913 when President Woodrow Wilson signed the Federal Reserve Act.", "label": "SUPPORTED", "category": "Finance"},
    {"id": "tr_sup_18", "claim": "Linus Torvalds released the initial Linux kernel in 1991 as a free and open-source operating system core.", "label": "SUPPORTED", "category": "Computing"},
    {"id": "tr_sup_19", "claim": "The United Kingdom officially entered the European Communities on January 1, 1973 alongside Denmark and Ireland.", "label": "SUPPORTED", "category": "History"},
    {"id": "tr_sup_20", "claim": "Pierre Curie shared the 1903 Nobel Prize in Physics with Marie Curie and Henri Becquerel.", "label": "SUPPORTED", "category": "Physics"},

    # --- REFUTED (20) ---
    {"id": "tr_ref_01", "claim": "Marie Curie won the 1911 Nobel Prize in Literature for her scientific autobiographies.", "label": "REFUTED", "category": "Science"},
    {"id": "tr_ref_02", "claim": "Michael Collins walked on the Moon alongside Neil Armstrong during the Apollo 11 lunar EVA.", "label": "REFUTED", "category": "Space"},
    {"id": "tr_ref_03", "claim": "Sydney is the constitutionally designated federal capital city of the Commonwealth of Australia.", "label": "REFUTED", "category": "Geography"},
    {"id": "tr_ref_04", "claim": "Albert Einstein was awarded the 1921 Nobel Prize in Physics for his formulation of the general theory of relativity.", "label": "REFUTED", "category": "Physics"},
    {"id": "tr_ref_05", "claim": "Penicillin is an antiviral drug discovered by Louis Pasteur in Paris to cure influenza outbreaks.", "label": "REFUTED", "category": "Medicine"},
    {"id": "tr_ref_06", "claim": "The United Kingdom remained a full member of the European Union throughout the entirety of 2022.", "label": "REFUTED", "category": "Politics"},
    {"id": "tr_ref_07", "claim": "The United States Constitution was drafted and signed at the Capitol building in Washington, D.C.", "label": "REFUTED", "category": "History"},
    {"id": "tr_ref_08", "claim": "Apollo 12 suffered an in-flight oxygen tank explosion and failed to achieve a lunar surface landing.", "label": "REFUTED", "category": "Space"},
    {"id": "tr_ref_09", "claim": "The Treaty of Rome was signed in 1992 in the Netherlands to formally create the European Union single currency.", "label": "REFUTED", "category": "History"},
    {"id": "tr_ref_10", "claim": "Alexander Fleming received the 1945 Nobel Prize in Medicine entirely as a sole laureate without co-recipients.", "label": "REFUTED", "category": "Medicine"},
    {"id": "tr_ref_11", "claim": "Tim Berners-Lee developed the World Wide Web while employed as a software engineer at Bell Labs in New Jersey.", "label": "REFUTED", "category": "Technology"},
    {"id": "tr_ref_12", "claim": "Alan Turing built the ACE computer and deployed it operationally before the end of 1944.", "label": "REFUTED", "category": "Computing"},
    {"id": "tr_ref_13", "claim": "The League of Nations was founded in San Francisco in 1945 immediately after World War II.", "label": "REFUTED", "category": "Politics"},
    {"id": "tr_ref_14", "claim": "The double-helix molecular structure of DNA was discovered in 1920 by Gregor Mendel.", "label": "REFUTED", "category": "Biology"},
    {"id": "tr_ref_15", "claim": "The Federal Reserve was established in 1865 during the American Civil War by Abraham Lincoln.", "label": "REFUTED", "category": "Finance"},
    {"id": "tr_ref_16", "claim": "Max Planck won the Nobel Prize in Chemistry for synthesizing artificial radioactive isotopes.", "label": "REFUTED", "category": "Physics"},
    {"id": "tr_ref_17", "claim": "The first ARPANET message transmission occurred between Stanford University and Harvard University in 1985.", "label": "REFUTED", "category": "Technology"},
    {"id": "tr_ref_18", "claim": "Nelson Mandela was awarded the Nobel Peace Prize in 1980 for establishing the African National Congress.", "label": "REFUTED", "category": "Politics"},
    {"id": "tr_ref_19", "claim": "Linus Torvalds created the Linux kernel while working as a computer science professor at MIT.", "label": "REFUTED", "category": "Computing"},
    {"id": "tr_ref_20", "claim": "The United Kingdom joined the European Economic Community in 1957 as one of its six original founding signatories.", "label": "REFUTED", "category": "History"},

    # --- NOT ENOUGH INFO (20) [Believable details absent from corpus] ---
    {"id": "tr_nei_01", "claim": "Marie Curie was initially rejected by the University of Paris due to insufficient tuition payment records.", "label": "NOT ENOUGH INFO", "category": "Biography"},
    {"id": "tr_nei_02", "claim": "Neil Armstrong listened to cassette recordings of classical orchestral music during the translunar coast of Apollo 11.", "label": "NOT ENOUGH INFO", "category": "Space"},
    {"id": "tr_nei_03", "claim": "Alexander Fleming initially attempted to extract antibacterial compounds using specialized cold-pressed citrus oils.", "label": "NOT ENOUGH INFO", "category": "Medicine"},
    {"id": "tr_nei_04", "claim": "The architects designing Canberra based its arterial road grid on the historical street plans of Florence, Italy.", "label": "NOT ENOUGH INFO", "category": "Geography"},
    {"id": "tr_nei_05", "claim": "Albert Einstein used his early patent office salary to finance his private study of theoretical electrodynamics.", "label": "NOT ENOUGH INFO", "category": "Physics"},
    {"id": "tr_nei_06", "claim": "Tim Berners-Lee wrote the first hyper-text markup specification using a second-hand British typewriter in Geneva.", "label": "NOT ENOUGH INFO", "category": "Technology"},
    {"id": "tr_nei_07", "claim": "Nelson Mandela completed his correspondence degree in law while working in the prison quarry at Robben Island.", "label": "NOT ENOUGH INFO", "category": "Politics"},
    {"id": "tr_nei_08", "claim": "The delegates signing the US Constitution in 1787 held their celebratory dinner at the City Tavern in Philadelphia.", "label": "NOT ENOUGH INFO", "category": "History"},
    {"id": "tr_nei_09", "claim": "Max Planck was an avid mountaineer who frequently scaled peaks in the Bavarian Alps during summer breaks.", "label": "NOT ENOUGH INFO", "category": "Personal"},
    {"id": "tr_nei_10", "claim": "The delegates at the Treaty of Rome negotiations drafted the final treaty text inside the Palazzo dei Conservatori.", "label": "NOT ENOUGH INFO", "category": "History"},
    {"id": "tr_nei_11", "claim": "Pete Conrad carried a silver four-leaf clover medallion inside his spacesuit pocket during Apollo 12.", "label": "NOT ENOUGH INFO", "category": "Space"},
    {"id": "tr_nei_12", "claim": "Alan Turing competed in cross-country running trials for the British Olympic marathon squad in 1948.", "label": "NOT ENOUGH INFO", "category": "Biography"},
    {"id": "tr_nei_13", "claim": "The preliminary delegates at the San Francisco Conference met in secret rooms inside the Fairmont Hotel.", "label": "NOT ENOUGH INFO", "category": "Politics"},
    {"id": "tr_nei_14", "claim": "Rosalind Franklin stored the original diffraction Photo 51 in an iron safe at King's College London.", "label": "NOT ENOUGH INFO", "category": "Biology"},
    {"id": "tr_nei_15", "claim": "Howard Florey obtained laboratory funding for penicillin purification through an emergency grant from the Rockefeller Foundation.", "label": "NOT ENOUGH INFO", "category": "Medicine"},
    {"id": "tr_nei_16", "claim": "Guido van Rossum chose the name Python while reading television comedy scripts in a computer laboratory in Amsterdam.", "label": "NOT ENOUGH INFO", "category": "Computing"},
    {"id": "tr_nei_17", "claim": "The Panic of 1907 began after two Wall Street financiers attempted to corner the market on United Copper stock.", "label": "NOT ENOUGH INFO", "category": "Finance"},
    {"id": "tr_nei_18", "claim": "Linus Torvalds purchased his first 386 computer using personal loans co-signed by his parents in Helsinki.", "label": "NOT ENOUGH INFO", "category": "Computing"},
    {"id": "tr_nei_19", "claim": "British Prime Minister Edward Heath celebrated the UK's EEC entry by hosting an orchestral gala at Covent Garden.", "label": "NOT ENOUGH INFO", "category": "History"},
    {"id": "tr_nei_20", "claim": "Pierre Curie constructed his first sensitive quartz electrometer in a converted wooden shed behind the physics school.", "label": "NOT ENOUGH INFO", "category": "Physics"}
]

# =====================================================================
# 2. HELD-OUT TEST BENCHMARK CLAIMS (30 claims: 10 SUP, 10 REF, 10 NEI)
# Strictly held-out: never seen during feature extraction or MLP training.
# =====================================================================
HELD_OUT_TEST_CLAIMS = [
    # --- SUPPORTED (10) ---
    {"id": "test_sup_01", "claim": "Marie Curie won the 1903 Nobel Prize in Physics along with Pierre Curie and Henri Becquerel.", "label": "SUPPORTED", "category": "Science"},
    {"id": "test_sup_02", "claim": "Michael Collins served as the command module pilot during Apollo 11 while Armstrong and Aldrin walked on the Moon.", "label": "SUPPORTED", "category": "Space"},
    {"id": "test_sup_03", "claim": "Alexander Fleming, Howard Florey, and Ernst Boris Chain were jointly awarded the 1945 Nobel Prize in Medicine for penicillin.", "label": "SUPPORTED", "category": "Medicine"},
    {"id": "test_sup_04", "claim": "Canberra is the federal capital of Australia, situated between Sydney and Melbourne in the Australian Capital Territory.", "label": "SUPPORTED", "category": "Geography"},
    {"id": "test_sup_05", "claim": "Albert Einstein formulated the theory of Special Relativity in 1905 and General Relativity in 1915.", "label": "SUPPORTED", "category": "Physics"},
    {"id": "test_sup_06", "claim": "Tim Berners-Lee developed the World Wide Web and created the first web browser at CERN.", "label": "SUPPORTED", "category": "Technology"},
    {"id": "test_sup_07", "claim": "Nelson Mandela and F. W. de Klerk jointly received the 1993 Nobel Peace Prize for their work in ending apartheid.", "label": "SUPPORTED", "category": "Politics"},
    {"id": "test_sup_08", "claim": "The United States Constitution was signed in September 1787 by delegates meeting at Independence Hall in Philadelphia.", "label": "SUPPORTED", "category": "History"},
    {"id": "test_sup_09", "claim": "Max Planck introduced the quantum theory of light and received the 1918 Nobel Prize in Physics for his work.", "label": "SUPPORTED", "category": "Physics"},
    {"id": "test_sup_10", "claim": "The United Kingdom left the European Union in 2020 following a national referendum held in June 2016.", "label": "SUPPORTED", "category": "History"},

    # --- REFUTED (10) ---
    {"id": "test_ref_01", "claim": "Marie Curie was awarded the Nobel Prize in Medicine in 1911 for inventing the mobile X-ray apparatus.", "label": "REFUTED", "category": "Science"},
    {"id": "test_ref_02", "claim": "Apollo 11 successfully landed three astronauts on the surface of the Moon simultaneously.", "label": "REFUTED", "category": "Space"},
    {"id": "test_ref_03", "claim": "Melbourne was selected as the permanent federal capital city of Australia by parliamentary act in 1908.", "label": "REFUTED", "category": "Geography"},
    {"id": "test_ref_04", "claim": "Albert Einstein was awarded the 1921 Nobel Prize in Physics for his discovery of gravitational waves.", "label": "REFUTED", "category": "Physics"},
    {"id": "test_ref_05", "claim": "Alexander Fleming discovered penicillin in 1945 at the University of Cambridge.", "label": "REFUTED", "category": "Medicine"},
    {"id": "test_ref_06", "claim": "The United Kingdom was one of the six original countries that signed the Treaty of Rome in 1957.", "label": "REFUTED", "category": "History"},
    {"id": "test_ref_07", "claim": "The United States Constitution was ratified in London, England before being presented to the Continental Congress.", "label": "REFUTED", "category": "History"},
    {"id": "test_ref_08", "claim": "Apollo 12 was cancelled before launch due to engine malfunctions and never reached lunar orbit.", "label": "REFUTED", "category": "Space"},
    {"id": "test_ref_09", "claim": "Max Planck won the Nobel Prize in Literature for his philosophical monographs on natural law.", "label": "REFUTED", "category": "Physics"},
    {"id": "test_ref_10", "claim": "Nelson Mandela served as the prime minister of the United Kingdom following his release from prison.", "label": "REFUTED", "category": "Politics"},

    # --- NOT ENOUGH INFO (10) [Believable historical/scientific assertions absent from reference corpus] ---
    {"id": "test_nei_01", "claim": "Marie Curie was an accomplished cellist who performed chamber music in Polish student cultural circles.", "label": "NOT ENOUGH INFO", "category": "Biography"},
    {"id": "test_nei_02", "claim": "Neil Armstrong carried a fragment of the Wright Brothers' aircraft fabric inside his Apollo 11 spacesuit pocket.", "label": "NOT ENOUGH INFO", "category": "Space"},
    {"id": "test_nei_03", "claim": "Alexander Fleming was nominated for the Nobel Prize on five separate occasions before finally winning in 1945.", "label": "NOT ENOUGH INFO", "category": "Medicine"},
    {"id": "test_nei_04", "claim": "The parliamentary triangle in Canberra was specifically engineered to align with lunar solstices observed in the southern sky.", "label": "NOT ENOUGH INFO", "category": "Geography"},
    {"id": "test_nei_05", "claim": "Albert Einstein frequently played violin duets with quantum physicist Max Planck during their years in Berlin.", "label": "NOT ENOUGH INFO", "category": "Physics"},
    {"id": "test_nei_06", "claim": "Tim Berners-Lee initially proposed naming the World Wide Web 'The Information Mesh' before adopting its final title.", "label": "NOT ENOUGH INFO", "category": "Technology"},
    {"id": "test_nei_07", "claim": "Nelson Mandela was offered an early conditional pardon by the South African government in 1985 which he formally declined.", "label": "NOT ENOUGH INFO", "category": "Politics"},
    {"id": "test_nei_08", "claim": "The inkwell used by George Washington to sign the US Constitution was crafted from solid silver mined in Connecticut.", "label": "NOT ENOUGH INFO", "category": "History"},
    {"id": "test_nei_09", "claim": "Apollo 12 astronaut Pete Conrad was the shortest astronaut selected in NASA's second astronaut group.", "label": "NOT ENOUGH INFO", "category": "Space"},
    {"id": "test_nei_10", "claim": "Max Planck served as an advisor on astronomical instrumentation for the Kaiser Wilhelm Institute in 1915.", "label": "NOT ENOUGH INFO", "category": "Physics"}
]

# =====================================================================
# 3. POLITICAL STRESS TEST CLAIMS (6 claims: 2 SUP, 2 REF, 2 NEI)
# =====================================================================
POLITICAL_CLAIMS = [
    {"id": "pol_01", "claim": "The United States established the Federal Reserve System via an Act of Congress in December 1913.", "label": "SUPPORTED", "category": "Political Economy"},
    {"id": "pol_02", "claim": "Barack Obama was elected as the 44th president of the United States in the 2008 general election.", "label": "SUPPORTED", "category": "US Politics"},
    {"id": "pol_03", "claim": "The United Kingdom voted to join the European Union in a national referendum held in 2016.", "label": "REFUTED", "category": "European Politics"},
    {"id": "pol_04", "claim": "The United Nations was established in 1845 immediately following the Treaty of Versailles.", "label": "REFUTED", "category": "International Affairs"},
    {"id": "pol_05", "claim": "The Federal Reserve's secret vault contains subterranean corridors connecting to the Treasury Department.", "label": "NOT ENOUGH INFO", "category": "Political Rumor"},
    {"id": "pol_06", "claim": "Nelson Mandela attended secret diplomatic summits in Zurich prior to the official 1993 peace accords.", "label": "NOT ENOUGH INFO", "category": "Diplomacy"}
]

def setup_evaluation_files():
    fever_dir = DATA_DIR / "fever"
    liar_dir = DATA_DIR / "liar"
    fever_dir.mkdir(parents=True, exist_ok=True)
    liar_dir.mkdir(parents=True, exist_ok=True)

    # Save training pool
    train_file = fever_dir / "train_claims.json"
    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(TRAIN_CLAIMS, f, indent=2)
    print(f"[Setup] Saved {len(TRAIN_CLAIMS)} training claims to {train_file}")

    # Save held-out test benchmark
    test_file = fever_dir / "test_claims.json"
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(HELD_OUT_TEST_CLAIMS, f, indent=2)
    print(f"[Setup] Saved {len(HELD_OUT_TEST_CLAIMS)} held-out test claims to {test_file}")

    # Overwrite fever_claims.json with the clean held-out test claims so legacy loaders use held-out test
    legacy_fever_file = fever_dir / "fever_claims.json"
    with open(legacy_fever_file, "w", encoding="utf-8") as f:
        json.dump(HELD_OUT_TEST_CLAIMS, f, indent=2)
    print(f"[Setup] Synchronized {legacy_fever_file} with held-out test benchmark ({len(HELD_OUT_TEST_CLAIMS)} claims)")

    # Save political stress test claims
    pol_file = liar_dir / "political_test_claims.json"
    with open(pol_file, "w", encoding="utf-8") as f:
        json.dump(POLITICAL_CLAIMS, f, indent=2)
    print(f"[Setup] Saved {len(POLITICAL_CLAIMS)} political stress test claims to {pol_file}")

    # Synchronize liar_claims.json
    legacy_liar_file = liar_dir / "liar_claims.json"
    with open(legacy_liar_file, "w", encoding="utf-8") as f:
        json.dump(POLITICAL_CLAIMS, f, indent=2)
    print(f"[Setup] Synchronized {legacy_liar_file} ({len(POLITICAL_CLAIMS)} claims)")

if __name__ == "__main__":
    setup_evaluation_files()

