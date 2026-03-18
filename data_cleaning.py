
import re

def clean_reddit_posts(text):
    """
    Cleans all of the combined title and posts.
    This function holds all of the cleaning done in the data cleaning notebook. It unifies it for easier changes.
    
    :param text: The text value
    """
    clean_text = text

    # Replace links with <URL> tag
    clean_text = re.sub(r"https?://\S+|www\.\S+", "<URL>", clean_text)

    # Escape characters (retain non alphabet)
    clean_text = re.sub(r"\\[A-Za-z]", " ", clean_text)
    clean_text = re.sub(r"\\", "", clean_text) # remove the remaining backslashes

    clean_text = re.sub(r"\s+", " ", clean_text).strip() 

    # Extra spaces
    clean_text = re.sub(r"\s+", " ", clean_text) # normalize multispaces.

    return clean_text
