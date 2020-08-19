from datetime import datetime
import re
from typing import List

import en_core_web_sm
import pytz

# Regex to look for all URLs (mailto:, x-whatever://, etc.) https://gist.github.com/gruber/249502
# Removed case insensitive flag from the start: (?i)
url_all_re = r'\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'
url_all_re = re.compile(url_all_re, flags=re.IGNORECASE)

nlp = en_core_web_sm.load(disable=["tagger", "parser", "ner"])


def check_tweet(
    status,
    valid_place_types: List[str] = ['neighborhood', 'poi'],
    ignore_words: List[str] = [],
    ignore_user_screen_names: List[str] = [],
    ignore_user_id_str: List[str] = [],
) -> bool:
    '''Return True if tweet satisfies specific criteria
    '''
    return all([
        ('text' in status),
        (all([x not in ignore_words for x in status['text'].split()])),
        (status['coordinates'] or (status['place'] and status['place']['place_type'] in valid_place_types)),
        (status['user']['screen_name'] not in ignore_user_screen_names),
        (status['user']['id_str'] not in ignore_user_id_str),
        (status['user']['friends_count'] > 0),  # following
        (status['user']['followers_count'] > 0),  # followers
    ])


def date_string_to_datetime(
    date_string: str,
    fmt: str = '%a %b %d %H:%M:%S +0000 %Y',
    tzinfo=pytz.UTC,
) -> datetime:
    return datetime.strptime(date_string, fmt).replace(tzinfo=tzinfo)


def clean_text(text: str) -> str:
    # Remove URLs
    text = re.sub(url_all_re, '', text)

    return text


def stopword_lemma(text: str) -> str:
    def _token_filter(token):
        return not (token.is_punct | token.is_space | token.is_stop | (token.lemma_ == '-PRON-') | (len(token.text) <= 1))

    tokens = [token.lemma_ for token in nlp(text) if _token_filter(token)]

    return ' '.join(tokens)
