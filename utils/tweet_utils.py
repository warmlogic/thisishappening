from collections import Counter
from datetime import datetime
import re
from typing import List

import en_core_web_sm
import pytz

# Regex to look for all URLs (mailto:, x-whatever://, etc.) https://gist.github.com/gruber/249502
# Removed case insensitive flag from the start: (?i)
url_all_re = r'\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'
url_all_re = re.compile(url_all_re, flags=re.IGNORECASE)

ellipsis_unicode = '\u2026'

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


def remove_punct_from_end(text):
    # idx = next((i for i, j in enumerate(reversed(text)) if not j in string.punctuation), 0)
    idx = next((i for i, j in enumerate(reversed(text)) if j.isalnum()), 0)
    return text[:-idx] if idx else text


def is_user_or_hashtag(text):
    return any([text.startswith('@'), text.startswith('#')])


def stopword_lemma(text: str) -> str:
    def _token_filter(token):
        return not any([
            token.is_punct,
            token.is_space,
            token.is_stop,
            (token.lemma_ == '-PRON-'),
            (token.is_ascii and (len(token.text) <= 1)),
        ])

    tokens = [token.lemma_ for token in nlp(text) if _token_filter(token)]

    return ' '.join(tokens)


def get_tokens_to_tweet(event_tweets: List, token_count_min: int = None):
    if token_count_min is None:
        token_count_min = 2

    tweets = [clean_text(et.tweet_body) for et in event_tweets]
    # remove tokens with ellipses
    tweets = [' '.join([token for token in tweet.split() if not (ellipsis_unicode in token)]) for tweet in tweets]

    # pull out user names and hashtags, remove punctuation from end
    users_hashtags = [[remove_punct_from_end(token) for token in tweet.split() if is_user_or_hashtag(token)] for tweet in tweets]

    # get the remaining text
    tweets = [' '.join([token for token in tweet.split() if not is_user_or_hashtag(token)]) for tweet in tweets]
    # remove stopwords and lemmatize
    tweets = [stopword_lemma(tweet) for tweet in tweets]

    # flatten list of lists
    tokens = [token.lower() for tweet in tweets for token in tweet.split()]
    users_hashtags = [token for tweet in users_hashtags for token in tweet]

    counter = Counter(tokens + users_hashtags).most_common()
    # get tokens; reduce threshold if no tokens are above the threshold
    tokens_to_tweet = []
    while not tokens_to_tweet and token_count_min > 0:
        # keep those above the threshold
        tokens_to_tweet = [tc[0] for tc in counter if tc[1] >= token_count_min]
        token_count_min -= 1

    if not tokens_to_tweet:
        tokens_to_tweet = ['No tweet text found']

    return tokens_to_tweet
