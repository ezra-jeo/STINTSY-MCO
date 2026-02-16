import re
import pandas as pd

def clean_reddit_posts(text):
    """
    Cleans all of the combined title and posts.
    
    :param text: The text value
    """
    clean_text = text

    # Replace links with <URL> tag
    clean_text = re.sub(r"https?://\S+|www\.\S+", "<URL>", clean_text)

    # Escape characters (retain non alphabet)
    clean_text = re.sub(r"\\[A-Za-z]", " ", clean_text)
    clean_text = re.sub(r"\\", "", clean_text) # remove the remaining backslashes

    clean_text = clean_text.replace("\n", " ")
    clean_text = clean_text.replace("\r", " ")
    clean_text = clean_text.replace("\t", " ")
    
    # Extra spaces
    clean_text = re.sub(r"\s+", " ", clean_text) # normalize multispaces.

    return clean_text

def extract_emoji_features(text):
    """
    Track if ANY emojis present.
    
    :param text: data instance
    """
    
    # Has any positive emoticon
    has_positive = bool(re.search(r":\)|:D|;\)", text))
    
    # Has any negative emoticon
    has_negative = bool(re.search(r":\(|:\"|D:", text))
    
    # Has crying emoticon
    has_crying = bool(re.search(r":\"|T_T|Q_Q", text))
    
    return int(has_positive), int(has_negative), int(has_crying)

def extract_case_features(text):
    """
    Helper function for case feature extraction.
    
    :param text: data instance
    """
    
    words = text.split()
    if len(words) == 0:
        return 0, 0.0
    upper_count = sum([1 for word in words if word.isupper() and len(word) > 1])
    upper_ratio = upper_count / len(words)

    return upper_count, upper_ratio

def extract_linguistic_features(text_col: pd.Series):
    """
    Extract all linguistic features from CASE-PRESERVED text.
    
    :param text_col: The dataframe containing the text data
    """
    features = pd.DataFrame()
    
    # Pronoun features
    features["first_person_singular"] = text_col.str.count(
        r"\b(I(?:'|')?(?:m|ve|ll|d)?|me|my|mine|myself)\b", flags=re.IGNORECASE
    )
    features["first_person_plural"] = text_col.str.count(
        r"\b(we(?:'|')?(?:re|ve|ll|d)?|us|our|ours|ourselves)\b", flags=re.IGNORECASE
    )
    
    # Absolutist words (Obtained from paper)
    absolutist = r"\b(absolutely|all|always|complete|completely|constant|constantly|definitely|entire|entirely|ever|every|everyone|everything|full|must|never|nothing|totally|whole)\b"
    features["absolutist_count"] = text_col.str.count(absolutist, flags=re.IGNORECASE)
    
    # Negative emotion
    negative = r"\b(sad|miserable|unhappy|depressed|hopeless|worthless|alone|lonely|hurt|pain|suffer|cry|tears|awful|terrible|horrible)\b"
    features["negative_emotion"] = text_col.str.count(negative, flags=re.IGNORECASE)
    
    # Death-related
    death = r"\b(death|die|dead|dying|suicide|suicidal|kill|killed|killing|end|ending)\b"
    features["death_related"] = text_col.str.count(death, flags=re.IGNORECASE)
    
    # Past tense
    features["past_tense"] = text_col.str.count(
        r"\b(was|were|had|did|been)\b", flags=re.IGNORECASE
    )
    
    # Punctuation (emotional intensity)
    features["exclamation_count"] = text_col.str.count("!")
    features["question_count"] = text_col.str.count(r"\?")
    
    # Case features
    case_features = text_col.apply(extract_case_features)
    features["upper_word_count"] = [x[0] for x in case_features]
    features["upper_word_ratio"] = [x[1] for x in case_features]

    # Emojis
    emoji_features = text_col.apply(extract_emoji_features)
    features["has_positive_emoji"], features["has_negative_emoji"], features["has_crying_emoji"] = zip(*emoji_features)

    return features