from collections import Counter
from datetime import datetime
import functools
import operator
import re
import string
from typing import List

import emoji
import en_core_web_sm
from ftfy import fix_text
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
    ignore_words_cf = [y.casefold() for y in ignore_words]
    return all([
        ('text' in status),
        (all([x.casefold() not in ignore_words_cf for x in clean_text(status['text']).split()])),
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


def split_text(text: str) -> List[str]:
    # Put whitespace between all words and emojis
    # https://stackoverflow.com/a/49930688/2592858
    text_split_emoji = emoji.get_emoji_regexp().split(text)
    text_split_whitespace = [substr.split() for substr in text_split_emoji]
    tokens = functools.reduce(operator.concat, text_split_whitespace)
    return tokens


def is_username(text):
    return text.startswith('@')


def is_hashtag(text):
    return text.startswith('#')


def is_username_or_hashtag(text):
    return any([is_username(text), is_hashtag(text)])


def clean_token(token: str) -> str:
    def remove_punct_from_end(text):
        idx = next((i for i, j in enumerate(reversed(text)) if j.isalnum()), 0)
        return text[:-idx] if idx else text

    # Replace some punctuation with space; everything in string.punctuation except: # ' . @
    punct_to_remove = '!"$%&()*+,-/:;<=>?[\\]^_`{|}~'
    punct_to_remove = f'[{re.escape(punct_to_remove)}]'
    token = re.sub(punct_to_remove, ' ', token).strip()

    # Remove all periods
    # TODO: why not replace with space?
    token = re.sub(re.escape('.'), '', token)

    # Remove possessive "apostrophe s" from usernames and hashtags so they are tweetable
    if is_username_or_hashtag(token):
        token = re.sub(r"(.+)'s$", r'\1', token)

    # Remove any trailing punctuation
    token = remove_punct_from_end(token)

    # The only remaining characters in this token are punctuation; don't keep any
    if all([c in string.punctuation for c in token]):
        token = ''

    return token


def remove_urls(text: str) -> str:
    # Remove URLs
    return re.sub(url_all_re, '', text)


def clean_text(text: str) -> str:
    # Remove tokens with ellipses; assume they are truncated words
    text = ' '.join([token for token in text.split() if not (ellipsis_unicode in token)])

    # Fix wonky characters
    text = fix_text(text)

    # Ensure emojis are surrounded by whitespace
    tokens = split_text(text)

    # Clean up punctuation
    tokens = [clean_token(token) for token in tokens]

    return ' '.join(tokens)


def filter_tokens(text: str) -> str:
    def _token_filter(token):
        return not any([
            token.is_punct,
            token.is_space,
            token.is_stop,
            (token.lemma_ == '-PRON-'),
            (len(token.text) <= 1),
        ])

    # tokens = [token.lemma_ for token in nlp(text) if _token_filter(token)]
    tokens = [token.text for token in nlp(text) if _token_filter(token)]

    return ' '.join(tokens)


def get_tokens_to_tweet(event_tweets: List, token_count_min: int = None, remove_username_at: bool = True):
    if token_count_min is None:
        token_count_min = 2

    tweets = [remove_urls(et.tweet_body) for et in event_tweets]

    tweets = [clean_text(tweet) for tweet in tweets]

    # Pull out user names, hashtags, and emojis
    users_hashtags_emojis = [[token for token in split_text(tweet) if is_username_or_hashtag(token) or emoji.emoji_count(token)] for tweet in tweets]

    # Get the remaining text
    tweets = [' '.join([token for token in tweet.split() if (not is_username_or_hashtag(token)) and (not emoji.emoji_count(token))]) for tweet in tweets]

    # Remove stopwords
    tweets = [filter_tokens(tweet) for tweet in tweets]

    # Flatten list of lists
    tokens = [token.lower() for tweet in tweets for token in tweet.split()]
    users_hashtags_emojis = [token for tweet in users_hashtags_emojis for token in tweet]

    # Optionally remove @ from username so user won't be tagged when event is posted
    if remove_username_at:
        users_hashtags_emojis = [
            token.replace('@', '') if is_username(token) else token
            for token in users_hashtags_emojis
        ]

    # Keep alphanumeric tokens
    tokens = [token for token in tokens if token.isalnum()]

    counter = Counter(tokens + users_hashtags_emojis).most_common()
    # Get tokens; reduce threshold if no tokens are above the threshold
    tokens_to_tweet = []
    while not tokens_to_tweet and token_count_min > 0:
        # keep those above the threshold
        tokens_to_tweet = [tc[0] for tc in counter if tc[1] >= token_count_min]
        token_count_min -= 1

    if not tokens_to_tweet:
        tokens_to_tweet = ['No tweet text found']

    return tokens_to_tweet
