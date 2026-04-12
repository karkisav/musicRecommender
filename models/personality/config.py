TRAIT_QUESTION_MAP = {
    "Openness": {
        "Q1": ["Reading", "Art exhibitions", "Fantasy/Fairy tales"],
        "Q2": ["Foreign languages", "Psychology"],
    },
    "Conscientiousness": {
        "Q3": ["Prioritising workload", "Thinking ahead", "Writing notes"],
        "Q4": ["Reliability", "Keeping promises"],
    },
    "Extraversion": {
        "Q5": ["Socializing", "Fun with friends", "Public speaking"],
        "Q6": ["Energy levels", "Dancing"],
    },
    "Agreeableness": {
        "Q7": ["Empathy", "Giving", "Compassion to animals"],
        "Q8": ["Charity", "Children"],
    },
    "Neuroticism": {
        "Q9":  ["Mood swings", "Getting angry", "Self-criticism"],
        "Q10": ["Life struggles", "Fear of public speaking"],
    },
}
 
# Flat mapping: trait → all survey columns that contribute to it
# (derived from above — used to compute trait scores from survey rows)
TRAIT_COLUMNS = {
    trait: [col for q_cols in questions.values() for col in q_cols]
    for trait, questions in TRAIT_QUESTION_MAP.items()
}
 
# Short names used as feature names in the model
TRAIT_NAMES = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

GENRE_COLUMNS = [
    # "Music",           # general music appreciation — consider dropping if too broad
    # "Slow songs or fast songs", 
    "Dance",
    "Folk", "Country",
    "Classical music", "Musical",
    "Pop", "Rock",
    "Metal or Hardrock",
    "Punk", "Hiphop, Rap",
    "Reggae, Ska",
    "Swing, Jazz",
    "Rock n roll",
    "Alternative", "Latino", 
    "Techno, Trance", "Opera",
]

# Threshold for converting 1–5 genre rating → binary "likes this genre"
GENRE_LIKE_THRESHOLD = 4

# ---------------------------------------------------------------------------
# Pairwise interaction terms to add to X (optional but recommended)
# These capture things like high Openness AND high Extraversion together.
# Set ADD_INTERACTIONS = False to train on raw 5 traits only.
# ---------------------------------------------------------------------------
 
ADD_INTERACTIONS = True