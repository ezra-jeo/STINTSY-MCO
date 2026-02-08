import re

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

