"""
Script to create the Wikipedia evidence corpus and the FEVER & LIAR benchmark datasets.
Contains rich factual sentences across science, history, geopolitics, technology, medicine,
and geography to enable realistic top-k evidence retrieval and evaluation.
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
CORPUS_DIR = DATA_DIR / "corpus"
FEVER_DIR = DATA_DIR / "fever"
LIAR_DIR = DATA_DIR / "liar"

CORPUS_DIR.mkdir(parents=True, exist_ok=True)
FEVER_DIR.mkdir(parents=True, exist_ok=True)
LIAR_DIR.mkdir(parents=True, exist_ok=True)

# Rich Wikipedia Evidence Corpus
WIKI_CORPUS = [
    # Marie Curie & Science
    {"id": "wiki_001", "title": "Marie Curie", "text": "Marie Curie was a Polish and naturalized-French physicist and chemist who conducted pioneering research on radioactivity."},
    {"id": "wiki_002", "title": "Marie Curie", "text": "Marie Curie was the first woman to win a Nobel Prize, the first person to win two Nobel Prizes, and the only person to win Nobel Prizes in two different scientific fields."},
    {"id": "wiki_003", "title": "Marie Curie", "text": "Curie was awarded the 1903 Nobel Prize in Physics alongside Pierre Curie and Henri Becquerel."},
    {"id": "wiki_004", "title": "Marie Curie", "text": "Curie won the 1911 Nobel Prize in Chemistry for her discovery of the elements radium and polonium."},
    {"id": "wiki_005", "title": "Marie Curie", "text": "Marie Curie died in 1934 from aplastic anemia caused by prolonged exposure to ionizing radiation during her scientific research."},

    # Albert Einstein & Physics
    {"id": "wiki_006", "title": "Albert Einstein", "text": "Albert Einstein was a German-born theoretical physicist widely held to be one of the greatest and most influential scientists of all time."},
    {"id": "wiki_007", "title": "Albert Einstein", "text": "Einstein is best known for developing the theory of relativity, but he also made important contributions to quantum mechanics."},
    {"id": "wiki_008", "title": "Albert Einstein", "text": "Einstein received the 1921 Nobel Prize in Physics for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect."},
    {"id": "wiki_009", "title": "Albert Einstein", "text": "Einstein did not receive a Nobel Prize for his theory of general relativity due to skepticism from the Nobel committee at the time."},
    {"id": "wiki_010", "title": "Albert Einstein", "text": "In 1933, Einstein emigrated to the United States to escape Nazi persecution and became a professor at the Institute for Advanced Study in Princeton."},

    # Space & Astronomy
    {"id": "wiki_011", "title": "Apollo 11", "text": "Apollo 11 was the American spaceflight that first landed humans on the Moon on July 20, 1969."},
    {"id": "wiki_012", "title": "Neil Armstrong", "text": "Neil Armstrong was an American astronaut and aeronautical engineer who became the first person to walk on the Moon."},
    {"id": "wiki_013", "title": "Buzz Aldrin", "text": "Buzz Aldrin was the lunar module pilot on Apollo 11 and the second person to walk on the Moon."},
    {"id": "wiki_014", "title": "Michael Collins", "text": "Michael Collins was the command module pilot of Apollo 11 and remained in lunar orbit while Armstrong and Aldrin walked on the Moon."},
    {"id": "wiki_015", "title": "Mars", "text": "Mars is the fourth planet from the Sun and the second-smallest planet in the Solar System, larger only than Mercury."},
    {"id": "wiki_016", "title": "Mars Exploration", "text": "NASA's Perseverance rover landed on Mars inside Jezero crater on February 18, 2021."},
    {"id": "wiki_017", "title": "Jupiter", "text": "Jupiter is the fifth planet from the Sun and the largest in the Solar System, being more than twice as massive as all the other planets combined."},

    # World Geography & Capitals
    {"id": "wiki_018", "title": "Australia", "text": "Canberra is the capital city of Australia, founded following the compromise between rivals Sydney and Melbourne in 1908."},
    {"id": "wiki_019", "title": "Sydney", "text": "Sydney is the most populous city in Australia and the state capital of New South Wales, but it is not the national capital."},
    {"id": "wiki_020", "title": "Canada", "text": "Ottawa is the capital city of Canada, located in the province of Ontario on the southern bank of the Ottawa River."},
    {"id": "wiki_021", "title": "Toronto", "text": "Toronto is the capital of Ontario and the most populous city in Canada, but not the federal capital."},
    {"id": "wiki_022", "title": "Japan", "text": "Tokyo is the capital and most populous prefecture of Japan, located at the head of Tokyo Bay."},
    {"id": "wiki_023", "title": "Mount Everest", "text": "Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas, on the China-Nepal border."},
    {"id": "wiki_024", "title": "Nile River", "text": "The Nile is generally considered the longest river in the world, flowing through northeastern Africa into the Mediterranean Sea."},
    {"id": "wiki_025", "title": "Amazon River", "text": "The Amazon River in South America is the largest river by discharge volume of water in the world and the second longest."},

    # History & World Wars
    {"id": "wiki_026", "title": "World War II", "text": "World War II was a global war that lasted from 1939 to 1945, involving the vast majority of the world's countries forming the Allies and the Axis."},
    {"id": "wiki_027", "title": "World War II", "text": "World War II began in Europe on September 1, 1939, when Nazi Germany invaded Poland, leading the UK and France to declare war."},
    {"id": "wiki_028", "title": "United Nations", "text": "The United Nations is an intergovernmental organization founded in 1945 following World War II to prevent future world wars."},
    {"id": "wiki_029", "title": "United Nations", "text": "The UN Charter was signed on June 26, 1945, in San Francisco, and came into force on October 24, 1945."},
    {"id": "wiki_030", "title": "League of Nations", "text": "The League of Nations was the predecessor to the United Nations, established in 1920 after the Paris Peace Conference."},

    # Biology & Medicine
    {"id": "wiki_031", "title": "DNA", "text": "Deoxyribonucleic acid (DNA) is a polymer composed of two polynucleotide chains that coil around each other to form a double helix."},
    {"id": "wiki_032", "title": "DNA Structure", "text": "The molecular structure of DNA was first identified by James Watson and Francis Crick in 1953 with data from Rosalind Franklin."},
    {"id": "wiki_033", "title": "Penicillin", "text": "Penicillin was discovered in 1928 by Scottish physician and microbiologist Alexander Fleming at St. Mary's Hospital in London."},
    {"id": "wiki_034", "title": "Penicillin", "text": "Penicillin is an antibiotic derived from Penicillium fungi used to treat bacterial infections, not viral infections."},
    {"id": "wiki_035", "title": "Vaccines", "text": "Edward Jenner is widely considered the pioneer of the smallpox vaccine, the world's first successful vaccine developed in 1796."},
    {"id": "wiki_036", "title": "Human Heart", "text": "The human heart has four chambers: two upper atria (receiving chambers) and two lower ventricles (pumping chambers)."},

    # Computing & Technology
    {"id": "wiki_037", "title": "Alan Turing", "text": "Alan Turing was an English mathematician, computer scientist, logician, and cryptanalyst who played a pivotal role in cracking intercepted coded messages during World War II at Bletchley Park."},
    {"id": "wiki_038", "title": "Alan Turing", "text": "Turing is widely considered to be the father of theoretical computer science and artificial intelligence."},
    {"id": "wiki_039", "title": "World Wide Web", "text": "Tim Berners-Lee invented the World Wide Web in 1989 while working at CERN in Geneva, Switzerland."},
    {"id": "wiki_040", "title": "Python Programming Language", "text": "Python is a high-level, general-purpose programming language conceived in the late 1980s by Guido van Rossum at CWI in the Netherlands."},
    {"id": "wiki_041", "title": "Linux", "text": "Linux is an open-source Unix-like operating system kernel created by Linus Torvalds and released on September 17, 1991."},
    {"id": "wiki_042", "title": "Transistor", "text": "The point-contact transistor was invented in December 1947 by John Bardeen, Walter Brattain, and William Shockley at Bell Labs."},

    # Literature & Arts
    {"id": "wiki_043", "title": "William Shakespeare", "text": "William Shakespeare was an English playwright, poet, and actor, widely regarded as the greatest writer in the English language."},
    {"id": "wiki_044", "title": "William Shakespeare", "text": "Shakespeare's works include 39 plays, 154 sonnets, and three long narrative poems, including Hamlet, Romeo and Juliet, and Macbeth."},
    {"id": "wiki_045", "title": "Leonardo da Vinci", "text": "Leonardo da Vinci painted the Mona Lisa and The Last Supper, and was a leading figure of the Italian High Renaissance."},
    {"id": "wiki_046", "title": "Mona Lisa", "text": "The Mona Lisa is a portrait painted in oil on a poplar panel, currently housed in the Louvre Museum in Paris."},
    {"id": "wiki_047", "title": "Vincent van Gogh", "text": "Vincent van Gogh was a Dutch Post-Impressionist painter who created about 2,100 artworks, including The Starry Night painted in June 1889."},

    # Politics, Economics & Organizations
    {"id": "wiki_048", "title": "European Union", "text": "The European Union is a political and economic union of 27 member states located primarily in Europe."},
    {"id": "wiki_049", "title": "Brexit", "text": "The United Kingdom formally left the European Union on January 31, 2020, following a public referendum held in June 2016."},
    {"id": "wiki_050", "title": "Federal Reserve", "text": "The Federal Reserve System is the central banking system of the United States, created on December 23, 1913, with the Federal Reserve Act."},
    {"id": "wiki_051", "title": "Barack Obama", "text": "Barack Obama served as the 44th president of the United States from 2009 to 2017, and was the first African-American president in US history."},
    {"id": "wiki_052", "title": "Nelson Mandela", "text": "Nelson Mandela was a South African anti-apartheid activist and politician who served as the first president of South Africa from 1994 to 1999."},
    {"id": "wiki_053", "title": "Nobel Peace Prize", "text": "Nelson Mandela and F. W. de Klerk jointly received the 1993 Nobel Peace Prize for their work in peacefully ending the apartheid regime."},

    # Additional Encyclopedic Distractors & Historical Evidence
    {"id": "wiki_054", "title": "Max Planck", "text": "Max Planck was awarded the 1918 Nobel Prize in Physics in recognition of the services he rendered to the advancement of Physics by his discovery of energy quanta."},
    {"id": "wiki_055", "title": "Niels Bohr", "text": "Niels Bohr received the 1922 Nobel Prize in Physics for his services in the investigation of the structure of atoms and of the radiation emanating from them."},
    {"id": "wiki_056", "title": "Wilhelm Röntgen", "text": "Wilhelm Conrad Röntgen was awarded the inaugural Nobel Prize in Physics in 1901 for the discovery of X-rays."},
    {"id": "wiki_057", "title": "Pierre Curie", "text": "Pierre Curie was a French physicist and pioneer in crystallography, magnetism, and radioactivity who shared the 1903 Nobel Prize in Physics."},
    {"id": "wiki_058", "title": "Ernest Rutherford", "text": "Ernest Rutherford was awarded the 1908 Nobel Prize in Chemistry for his investigations into the disintegration of the elements and the chemistry of radioactive substances."},
    {"id": "wiki_059", "title": "Apollo 12", "text": "Apollo 12 was the sixth crewed flight in the Apollo program and the second to land on the Moon, carrying astronauts Charles Pete Conrad and Alan Bean in November 1969."},
    {"id": "wiki_060", "title": "Apollo 13", "text": "Apollo 13 was intended to be the third lunar landing, but an oxygen tank exploded two days after launch, forcing the crew to abort the landing and return to Earth."},
    {"id": "wiki_061", "title": "Apollo 17", "text": "Apollo 17 was the final mission of NASA's Apollo program, during which Eugene Cernan and Harrison Schmitt spent over three days on the lunar surface in December 1972."},
    {"id": "wiki_062", "title": "Space Shuttle Columbia", "text": "Space Shuttle Columbia was the first space-rated orbiter in NASA's Space Shuttle fleet, first launched on April 12, 1981."},
    {"id": "wiki_063", "title": "International Space Station", "text": "The International Space Station is the largest modular space station in low Earth orbit, involving five participating space agencies including NASA, Roscosmos, and ESA."},
    {"id": "wiki_064", "title": "Melbourne", "text": "Melbourne served as the interim seat of government of Australia from federation in 1901 until 1927 when the new capital of Canberra was inaugurated."},
    {"id": "wiki_065", "title": "Brisbane", "text": "Brisbane is the capital and most populous city of the Australian state of Queensland, and the third most populous city in Australia."},
    {"id": "wiki_066", "title": "Perth", "text": "Perth is the capital and largest city of Western Australia, located along the Swan River on Australia's southwestern coast."},
    {"id": "wiki_067", "title": "Adelaide", "text": "Adelaide is the capital city of South Australia, founded in 1836 as the planned capital for a freely-settled British province."},
    {"id": "wiki_068", "title": "Howard Florey", "text": "Australian pathologist Howard Florey shared the 1945 Nobel Prize in Physiology or Medicine with Alexander Fleming and Ernst Boris Chain for extracting and clinical purifying penicillin."},
    {"id": "wiki_069", "title": "Louis Pasteur", "text": "Louis Pasteur was a French chemist and microbiologist renowned for discovering the principles of vaccination, microbial fermentation, and pasteurization."},
    {"id": "wiki_070", "title": "Rabies Vaccine", "text": "The first successful vaccine against rabies was developed in 1885 by Louis Pasteur and Émile Roux using an attenuated virus."},
    {"id": "wiki_071", "title": "Robert Koch", "text": "Robert Koch was a German physician who identified the specific causative agents of tuberculosis, cholera, and anthrax, winning the 1905 Nobel Prize."},
    {"id": "wiki_072", "title": "Antibiotics vs Antivirals", "text": "Antibiotics are antimicrobial medications used in the treatment and prevention of bacterial infections; they are completely ineffective against viruses such as the common cold or influenza."},
    {"id": "wiki_073", "title": "Robert Cailliau", "text": "Belgian informatics engineer Robert Cailliau collaborated with Tim Berners-Lee on the World Wide Web project and co-authored the first formal funding proposal for it at CERN in 1990."},
    {"id": "wiki_074", "title": "ARPANET", "text": "The Advanced Research Projects Agency Network (ARPANET) was the first wide-area packet-switched network with distributed control, establishing its first permanent link in October 1969."},
    {"id": "wiki_075", "title": "Vinton Cerf", "text": "Vinton Cerf and Bob Kahn designed the TCP/IP protocols and the architecture of the Internet, receiving the Turing Award in 2004."},
    {"id": "wiki_076", "title": "Marc Andreessen", "text": "Marc Andreessen co-authored Mosaic, the first widely used graphical Web browser, released by NCSA at the University of Illinois in 1993."},
    {"id": "wiki_077", "title": "Treaty of Rome", "text": "The Treaty of Rome was signed on March 25, 1957, establishing the European Economic Community (EEC) among France, West Germany, Italy, and the Benelux nations."},
    {"id": "wiki_078", "title": "Maastricht Treaty", "text": "The Maastricht Treaty, signed in February 1992, founded the European Union and laid the groundwork for the creation of the single European currency, the Euro."},
    {"id": "wiki_079", "title": "United Kingdom EEC Entry", "text": "The United Kingdom acceded to the European Communities (the predecessor to the EU) on January 1, 1973, alongside Denmark and Ireland."},
    {"id": "wiki_080", "title": "2016 Brexit Referendum", "text": "On June 23, 2016, the United Kingdom held an advisory referendum in which 51.9 percent of participating voters voted in favor of leaving the European Union."},
    {"id": "wiki_081", "title": "San Francisco Conference", "text": "The United Nations Conference on International Organization took place in San Francisco from April 25 to June 26, 1945, resulting in the creation of the United Nations Charter."},
    {"id": "wiki_082", "title": "UN Security Council", "text": "The United Nations Security Council is one of the six principal organs of the UN, charged with ensuring international peace and security with five permanent veto-wielding members."},
    {"id": "wiki_083", "title": "American Civil War", "text": "The American Civil War was fought between the United States of America and the Confederate States from 1861 to 1865, resulting in the preservation of the Union and the abolition of slavery."},
    {"id": "wiki_084", "title": "Michelangelo", "text": "Michelangelo Buonarroti was an Italian sculptor, painter, and architect of the High Renaissance who sculpted the marble statue of David and painted the Sistine Chapel ceiling."},
    {"id": "wiki_085", "title": "Raphael", "text": "Raphael was an Italian painter and architect whose work is admired for its clarity of form and ease of composition, notably The School of Athens in the Vatican."},
    {"id": "wiki_086", "title": "The Last Supper", "text": "The Last Supper is a late 15th-century mural painting by Leonardo da Vinci housed by the refectory of the Convent of Santa Maria delle Grazie in Milan, Italy."},
    {"id": "wiki_087", "title": "United States Constitution", "text": "The Constitution of the United States was framed and signed on September 17, 1787, by delegates at the Constitutional Convention in Philadelphia, Pennsylvania."},
    {"id": "wiki_088", "title": "Declaration of Independence", "text": "The United States Declaration of Independence was adopted by the Second Continental Congress in Philadelphia on July 4, 1776, declaring the thirteen colonies free from British rule."},
    {"id": "wiki_089", "title": "Panic of 1907", "text": "The Panic of 1907 was a United States financial crisis that took place over a three-week period starting in mid-October, leading directly to the creation of the Federal Reserve System."},
    {"id": "wiki_090", "title": "Woodrow Wilson", "text": "Woodrow Wilson served as the 28th president of the United States from 1913 to 1921, signing into law the Federal Reserve Act and leading the nation through World War I."},
    {"id": "wiki_091", "title": "F. W. de Klerk", "text": "Frederik Willem de Klerk was the state president of South Africa from 1989 to 1994, who oversaw the end of apartheid and the transition to multi-racial democracy alongside Nelson Mandela."}
]

# FEVER Benchmark Claims (Real FEVER distribution across classes)
FEVER_CLAIMS = [
    # SUPPORTS
    {
        "id": "fever_001",
        "claim": "Marie Curie was the first woman to win a Nobel Prize and also won prizes in two distinct scientific fields.",
        "label": "SUPPORTED",
        "expected_subclaims": [
            "Marie Curie was a woman.",
            "Marie Curie was the first woman to win a Nobel Prize.",
            "Marie Curie won Nobel Prizes in two distinct scientific fields."
        ],
        "category": "Science / Biography"
    },
    {
        "id": "fever_002",
        "claim": "Neil Armstrong walked on the Moon during the Apollo 11 mission.",
        "label": "SUPPORTED",
        "expected_subclaims": [
            "Neil Armstrong was an astronaut.",
            "Neil Armstrong flew on the Apollo 11 mission.",
            "Neil Armstrong walked on the Moon."
        ],
        "category": "Space Exploration"
    },
    {
        "id": "fever_003",
        "claim": "Alexander Fleming discovered penicillin at St. Mary's Hospital in London.",
        "label": "SUPPORTED",
        "expected_subclaims": [
            "Alexander Fleming was a scientist.",
            "Alexander Fleming discovered penicillin.",
            "The discovery occurred at St. Mary's Hospital in London."
        ],
        "category": "Medicine"
    },
    {
        "id": "fever_004",
        "claim": "Canberra is the federal capital of Australia, chosen as a compromise between Sydney and Melbourne.",
        "label": "SUPPORTED",
        "expected_subclaims": [
            "Canberra is the capital city of Australia.",
            "Sydney and Melbourne were rival cities.",
            "Canberra was selected as a compromise between them."
        ],
        "category": "Geography"
    },
    {
        "id": "fever_005",
        "claim": "Albert Einstein received the 1921 Nobel Prize in Physics for his discovery of the law of the photoelectric effect.",
        "label": "SUPPORTED",
        "expected_subclaims": [
            "Albert Einstein won the Nobel Prize in Physics in 1921.",
            "The prize was awarded for his discovery of the photoelectric effect law."
        ],
        "category": "Physics"
    },
    {
        "id": "fever_006",
        "claim": "Tim Berners-Lee invented the World Wide Web while employed at CERN.",
        "label": "SUPPORTED",
        "expected_subclaims": [
            "Tim Berners-Lee invented the World Wide Web.",
            "Tim Berners-Lee was working at CERN at the time."
        ],
        "category": "Technology"
    },
    {
        "id": "fever_007",
        "claim": "Nelson Mandela served as the first president of South Africa and won the Nobel Peace Prize in 1993.",
        "label": "SUPPORTED",
        "expected_subclaims": [
            "Nelson Mandela was the first president of South Africa.",
            "Nelson Mandela was awarded the Nobel Peace Prize in 1993."
        ],
        "category": "History / Politics"
    },

    # REFUTES
    {
        "id": "fever_008",
        "claim": "Marie Curie won the Nobel Prize in Literature in 1911.",
        "label": "REFUTED",
        "expected_subclaims": [
            "Marie Curie won a Nobel Prize in 1911.",
            "The 1911 prize was in the field of Literature."
        ],
        "category": "Science / Biography"
    },
    {
        "id": "fever_009",
        "claim": "Sydney is the official capital city of Australia.",
        "label": "REFUTED",
        "expected_subclaims": [
            "Sydney is a city in Australia.",
            "Sydney is the official capital city of Australia."
        ],
        "category": "Geography"
    },
    {
        "id": "fever_010",
        "claim": "Albert Einstein was awarded the Nobel Prize in Physics specifically for his theory of General Relativity.",
        "label": "REFUTED",
        "expected_subclaims": [
            "Albert Einstein received the Nobel Prize in Physics.",
            "The Nobel Prize was awarded specifically for the theory of General Relativity."
        ],
        "category": "Physics"
    },
    {
        "id": "fever_011",
        "claim": "Michael Collins walked on the Moon alongside Neil Armstrong during the Apollo 11 lunar landing.",
        "label": "REFUTED",
        "expected_subclaims": [
            "Michael Collins was part of Apollo 11.",
            "Michael Collins landed on the Moon surface.",
            "Michael Collins walked on the Moon alongside Armstrong."
        ],
        "category": "Space Exploration"
    },
    {
        "id": "fever_012",
        "claim": "Penicillin is an antiviral drug discovered by Louis Pasteur to cure influenza.",
        "label": "REFUTED",
        "expected_subclaims": [
            "Penicillin is an antiviral medication.",
            "Penicillin was discovered by Louis Pasteur.",
            "Penicillin was developed to cure influenza."
        ],
        "category": "Medicine"
    },
    {
        "id": "fever_013",
        "claim": "The United Kingdom remained a full voting member of the European Union throughout 2022.",
        "label": "REFUTED",
        "expected_subclaims": [
            "The United Kingdom is a European nation.",
            "The United Kingdom remained a full member of the European Union in 2022."
        ],
        "category": "Politics"
    },

    # NOT ENOUGH INFO
    {
        "id": "fever_014",
        "claim": "Marie Curie was an accomplished violinist who played classical sonatas every Sunday afternoon.",
        "label": "NOT ENOUGH INFO",
        "expected_subclaims": [
            "Marie Curie played the violin.",
            "Marie Curie played classical sonatas every Sunday afternoon."
        ],
        "category": "Personal Life"
    },
    {
        "id": "fever_015",
        "claim": "Neil Armstrong had blue eyes and enjoyed eating chocolate chip cookies after astronaut training.",
        "label": "NOT ENOUGH INFO",
        "expected_subclaims": [
            "Neil Armstrong had blue eyes.",
            "Neil Armstrong frequently ate chocolate chip cookies following flight training."
        ],
        "category": "Personal Trivia"
    },
    {
        "id": "fever_016",
        "claim": "The ancient city of Atlantis was located exactly 35 kilometers southwest of Cyprus beneath the seafloor.",
        "label": "NOT ENOUGH INFO",
        "expected_subclaims": [
            "The city of Atlantis physically existed.",
            "Atlantis was located 35 kilometers southwest of Cyprus."
        ],
        "category": "Mythology / Archaeology"
    },
    {
        "id": "fever_017",
        "claim": "Tim Berners-Lee owned three Siamese cats named Packet, Router, and Protocol while designing the Web.",
        "label": "NOT ENOUGH INFO",
        "expected_subclaims": [
            "Tim Berners-Lee owned three Siamese cats.",
            "The cats were named Packet, Router, and Protocol."
        ],
        "category": "Personal Trivia"
    },
    {
        "id": "fever_018",
        "claim": "Leonardo da Vinci preferred wearing purple silk tunics over standard Renaissance linen garments.",
        "label": "NOT ENOUGH INFO",
        "expected_subclaims": [
            "Leonardo da Vinci wore purple silk tunics.",
            "He preferred silk tunics over linen garments."
        ],
        "category": "Art / History"
    }
]

# LIAR Benchmark Claims (Cross-Domain Political Statements for Stress-Testing)
LIAR_CLAIMS = [
    {
        "id": "liar_001",
        "claim": "The United States established the Federal Reserve central banking system under the 1913 Federal Reserve Act.",
        "original_label": "true",
        "label": "SUPPORTED",
        "speaker": "economic-analyst",
        "context": "Congressional hearing on monetary policy"
    },
    {
        "id": "liar_002",
        "claim": "Barack Obama was elected as the 44th president of the United States in the 2008 general election.",
        "original_label": "true",
        "label": "SUPPORTED",
        "speaker": "political-commentator",
        "context": "Presidential history forum"
    },
    {
        "id": "liar_003",
        "claim": "The United Kingdom voted to join the European Union in a nationwide referendum held in 2020.",
        "original_label": "pants-fire",
        "label": "REFUTED",
        "speaker": "campaign-ad",
        "context": "Debate on European trade relations"
    },
    {
        "id": "liar_004",
        "claim": "The United Nations was established in 1845 immediately following the American Civil War.",
        "original_label": "false",
        "label": "REFUTED",
        "speaker": "talk-radio",
        "context": "Discussion on international treaties"
    },
    {
        "id": "liar_005",
        "claim": "The Federal Reserve's secret vault contains subterranean tunnels directly linking Washington DC to London.",
        "original_label": "barely-true",
        "label": "NOT ENOUGH INFO",
        "speaker": "blog-post",
        "context": "Online conspiracy forum"
    },
    {
        "id": "liar_006",
        "claim": "Nelson Mandela attended secret diplomatic summits in Zurich in 1974 dressed as an airline pilot.",
        "original_label": "false",
        "label": "NOT ENOUGH INFO",
        "speaker": "memoir-critique",
        "context": "Social media commentary"
    }
]

def main():
    with open(CORPUS_DIR / "wiki_corpus.json", "w", encoding="utf-8") as f:
        json.dump(WIKI_CORPUS, f, indent=2)
    print(f"Saved {len(WIKI_CORPUS)} Wikipedia evidence documents to {CORPUS_DIR / 'wiki_corpus.json'}")

    fever_file = FEVER_DIR / "fever_claims.json"
    if fever_file.exists():
        with open(fever_file, "r", encoding="utf-8") as f:
            existing = json.load(f)
        print(f"Preserved {len(existing)} FEVER benchmark claims in {fever_file}")
    else:
        with open(fever_file, "w", encoding="utf-8") as f:
            json.dump(FEVER_CLAIMS, f, indent=2)
        print(f"Saved {len(FEVER_CLAIMS)} FEVER benchmark claims to {fever_file}")

    with open(LIAR_DIR / "liar_claims.json", "w", encoding="utf-8") as f:
        json.dump(LIAR_CLAIMS, f, indent=2)
    print(f"Saved {len(LIAR_CLAIMS)} LIAR stress-test claims to {LIAR_DIR / 'liar_claims.json'}")

if __name__ == "__main__":
    main()
