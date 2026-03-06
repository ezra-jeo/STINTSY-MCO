import re
import pandas as pd
import numpy as np
import emoji

# Plan 1, revise the dataset from the getgo and just say the limitaiton of tf-idf as the error for error analysis, model improvements will be made in neural network anyways.
# Plan 2, revise format, don't rely on data_cleaning.ipynb for cleaning, make a separate function for it. Pipeline. Easy calling and allows for you to split dataset immediately first to not worry about this.

def extract_emoji_features(text):
    """
    Track if ANY emojis present.
    
    :param text: data instance
    """
    
    # Has any positive emoticon
    has_positive = bool(re.search(r":\)|:D|;\)", text))
    
    # Has any negative emoticon
    has_negative = bool(re.search(r":\(|:\"|D:|</3", text))
    
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
    
    :param text_col: The column containing the text data
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
    
    # Negative emotion (Find backing, and revise if ever)
    negative = r"\b(sad|miserable|unhappy|depressed|hopeless|worthless|alone|lonely|hurt|pain|suffer|cry|tears|awful|terrible|horrible)\b"
    features["negative_emotion"] = text_col.str.count(negative, flags=re.IGNORECASE)
    
    # Death-related (Find backing, and revise if ever)
    death = r"\b(death|die|dead|dying|suicide|suicidal|kill|killed|killing|end|ending|struggle|struggling)\b"
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

# Revisions from Error Analysis
# Separate emoticons and emojis
def extract_emoticon_features(text):
    """
    Extract ASCII emoticon features (NOT Unicode emojis).
    These are text-based expressions like :) :( T_T
    """
    # Positive emoticons
    positive_pattern = r":\)|:D|;\)|=\)|:-\)|:\]|=D|:>|8\)"
    positive_count = len(re.findall(positive_pattern, text))
    
    # Negative emoticons (sad/frowning)
    negative_pattern = r":\(|:-\(|:\[|=\(|:<|D:|>:\(|8\("
    negative_count = len(re.findall(negative_pattern, text))
    
    # Crying emoticons (important for distress)
    crying_pattern = r":\"|:'[\(\|]|T_T|T\.T|Q_Q|;_;|ToT|;-;|QQ"
    crying_count = len(re.findall(crying_pattern, text))
    
    # Neutral/ambivalent emoticons
    neutral_pattern = r":-?[/\\]|:-?\||=\||:\|"
    neutral_count = len(re.findall(neutral_pattern, text))
    
    # Love/heart emoticons
    heart_pattern = r"<3|</3"  # </3 is broken heart
    heart_count = len(re.findall(heart_pattern, text))
    # Separate broken heart
    broken_heart_count = text.count("</3")
    
    return (positive_count, negative_count, crying_count, 
            neutral_count, heart_count, broken_heart_count)


def improved_extract_emoji_features(text):
    """
    Extract Unicode emoji features (NOT ASCII emoticons).
    Uses emoji library for comprehensive detection.
    """
    emojis_found = emoji.emoji_list(text)
    
    if not emojis_found:
        return 0, 0, 0, 0
    
    # Define categories based on CLDR emoji names
    negative_keywords = {
        'cry', 'tear', 'sad', 'frown', 'disappointed', 'worried',
        'anxious', 'confused', 'tired', 'weary', 'pensive', 'pleading',
        'broken', 'wilted', 'melting', 'downcast', 'persevering',
        'anguished', 'fearful', 'cold_sweat'
    }
    
    positive_keywords = {
        'smile', 'grin', 'joy', 'happy', 'heart', 'love', 'kiss',
        'star', 'sparkle', 'relieved', 'blush', 'hug', 'celebrate',
        'laughing', 'beaming', 'hugging'
    }
    
    crisis_keywords = {
        'skull', 'coffin', 'headstone', 'grave', 'dead', 'death',
        'gun', 'pistol', 'revolver', 'knife', 'dagger', 'blade',
        'pill', 'syringe', 'needle', 'bomb'
    }
    
    # Implicit distress emojis
    implicit_distress = {'🫠', '😶', '😐', '🙃'}
    
    negative_count = 0
    positive_count = 0
    crisis_count = 0
    implicit_count = 0
    
    for emoji_info in emojis_found:
        emoji_char = emoji_info['emoji']
        
        if emoji_char in implicit_distress:
            implicit_count += 1
            continue
        
        emoji_name = emoji.demojize(emoji_char).lower()
        
        if any(keyword in emoji_name for keyword in crisis_keywords):
            crisis_count += 1
        elif any(keyword in emoji_name for keyword in negative_keywords):
            negative_count += 1
        elif any(keyword in emoji_name for keyword in positive_keywords):
            positive_count += 1
    
    return negative_count, positive_count, crisis_count, implicit_count

def extract_all_features(text_col):
    """
    Extract all linguistic features from CASE-PRESERVED text.
    
    :param text_col: The column containing the text data
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
    
    # Negative emotion (Find backing, and revise if ever)
    negative = r"\b(sad|miserable|unhappy|depressed|hopeless|worthless|alone|lonely|hurt|pain|suffer|cry|tears|awful|terrible|horrible)\b"
    features["negative_emotion"] = text_col.str.count(negative, flags=re.IGNORECASE)
    
    # Death-related (Find backing, and revise if ever)
    death = r"\b(death|die|dead|dying|suicide|suicidal|kill|killed|killing|end|ending|struggle|struggling)\b"
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
    
    # ASCII EMOTICON FEATURES (separate from emojis!)
    print("Extracting ASCII emoticon features...")
    emoticon_feat = text_col.apply(extract_emoticon_features)
    features['emoticon_positive'] = [x[0] for x in emoticon_feat]
    features['emoticon_negative'] = [x[1] for x in emoticon_feat]
    features['emoticon_crying'] = [x[2] for x in emoticon_feat]
    features['emoticon_neutral'] = [x[3] for x in emoticon_feat]
    features['emoticon_heart'] = [x[4] for x in emoticon_feat]
    features['emoticon_broken_heart'] = [x[5] for x in emoticon_feat]
    
    # UNICODE EMOJI FEATURES (separate from emoticons!)
    print("Extracting Unicode emoji features...")
    emoji_feat = text_col.apply(extract_emoji_features)
    features['emoji_negative'] = [x[0] for x in emoji_feat]
    features['emoji_positive'] = [x[1] for x in emoji_feat]
    features['emoji_crisis'] = [x[2] for x in emoji_feat]
    features['emoji_implicit_distress'] = [x[3] for x in emoji_feat]
    
    return features

def preprocess_traditional(X: pd.DataFrame, y: np.array, train: bool) -> tuple:
    """
    Function that takes train and labels and preprocesses.
    """