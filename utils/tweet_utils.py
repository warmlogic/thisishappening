from collections import Counter, namedtuple
from datetime import datetime
import functools
import logging
import operator
import re
import string
from typing import Dict, List

import emoji
import en_core_web_sm
from ftfy import fix_text
import pytz

logger = logging.getLogger("happeninglogger")

# Regex to look for all URLs (mailto:, x-whatever://, etc.) https://gist.github.com/gruber/249502
# Removed case insensitive flag from the start: (?i)
url_all_re = r'\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'
url_all_re = re.compile(url_all_re, flags=re.IGNORECASE)

ellipsis_unicode = '\u2026'

nlp = en_core_web_sm.load(exclude=["parser", "ner"])

TweetInfo = namedtuple(
    'TweetInfo',
    [
        'status_id_str',
        'user_screen_name',
        'user_id_str',
        'created_at',
        'tweet_body',
        'tweet_language',
        'longitude',
        'latitude',
        'place_name',
        'place_type',
    ],
)


def get_tweet_info(status: Dict) -> Dict:
    status_id_str = status['id_str']
    user_screen_name = status['user']['screen_name']
    user_id_str = status['user']['id_str']
    created_at = date_string_to_datetime(status['created_at'])
    tweet_body = status['text']
    tweet_language = status['lang']
    if status['coordinates']:
        longitude = status['coordinates']['coordinates'][0]
        latitude = status['coordinates']['coordinates'][1]
    elif status['place']:
        lons = [x[0] for x in status['place']['bounding_box']['coordinates'][0]]
        longitude = sum(lons) / len(lons)
        lats = [x[1] for x in status['place']['bounding_box']['coordinates'][0]]
        latitude = sum(lats) / len(lats)
    else:
        longitude = None
        latitude = None
    place_name = status['place']['name']
    # Possible place_type values: country, admin, city, neighborhood, poi
    place_type = status['place']['place_type']

    tweet_info = TweetInfo(
        status_id_str=status_id_str,
        user_screen_name=user_screen_name,
        user_id_str=user_id_str,
        created_at=created_at,
        tweet_body=tweet_body,
        tweet_language=tweet_language,
        longitude=longitude,
        latitude=latitude,
        place_name=place_name,
        place_type=place_type,
    )

    return tweet_info


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


def filter_tokens(text: str, lemmatize: bool = False) -> str:
    def _token_filter(token):
        return not any([
            token.is_punct,
            token.is_space,
            token.is_stop,
            (token.lemma_ == '-PRON-'),
            (len(token.text) <= 1),
        ])

    tokens = [
        token.lemma_ if lemmatize else token.text for token in nlp(text)
        if _token_filter(token)
    ]

    return ' '.join(tokens)


def get_tokens_to_tweet(tweets: List, token_count_min: int = None, remove_username_at: bool = True):
    if token_count_min is None:
        token_count_min = 2

    try:
        tweets = [remove_urls(x.tweet_body) for x in tweets]
    except AttributeError:
        tweets = [remove_urls(x['tweet_body']) for x in tweets]
    except TypeError:
        logger.exception(f"Unsupported tweet dtype: {type(tweets[0])}")

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


def get_coords(tweets):
    try:
        lons = [x.longitude for x in tweets]
        lats = [x.latitude for x in tweets]
    except AttributeError:
        lons = [x['longitude'] for x in tweets]
        lats = [x['latitude'] for x in tweets]
    except TypeError:
        logger.exception(f"Unsupported tweet dtype: {type(tweets[0])}")

    return lons, lats


def get_place_name(tweets, valid_place_types: List[str] = ['neighborhood', 'poi']):
    # Get the most common place name from these tweets; only consider neighborhood or poi
    try:
        place_names = [x.place_name for x in tweets if x.place_type in valid_place_types]
    except AttributeError:
        place_names = [x['place_name'] for x in tweets if x['place_type'] in valid_place_types]
    except TypeError:
        logger.exception(f"Unsupported tweet dtype: {type(tweets[0])}")

    place_name = Counter(place_names).most_common(1)[0][0] if place_names else None

    return place_name


def get_status_ids(tweets):
    try:
        status_ids = [x.status_id_str for x in tweets]
    except AttributeError:
        status_ids = [x['status_id_str'] for x in tweets]
    except TypeError:
        logger.exception(f"Unsupported tweet dtype: {type(tweets[0])}")

    return status_ids
