import re


def remove_links(s):
    """swap a link with a _URL_ token"""
    return re.sub(r"http\S+", "_URL_", s)


def remove_html_tags(s):
    """Remove html tags from a string"""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', s)


def remove_standard_punctuation(s):
    punctuation = '¿!"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~'  # We are leaving out _ for the _URL_ token
    return s.translate(str.maketrans(punctuation, ' ' * len(punctuation)))


def remove_extra_whitespace(s):
    s = ' '.join(s.split())
    return s


def clean_text(s):
    # s = s.lower()
    # s = remove_links(s)
    # s = remove_html_tags(s)
    # s = remove_standard_punctuation(s)
    # s = remove_extra_whitespace(s)
    return s
