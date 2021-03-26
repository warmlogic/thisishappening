from collections import Counter, namedtuple
from datetime import datetime
import functools
import logging
import operator
import re
import string
from time import sleep
from typing import Dict, List
import unicodedata
import urllib

import emoji
import en_core_web_sm
from ftfy import fix_text
import pytz
from unidecode import unidecode

logger = logging.getLogger("happeninglogger")

# Regex to look for all URLs (mailto:, x-whatever://, etc.) https://gist.github.com/gruber/249502
# Removed case insensitive flag from the start: (?i)
url_all_re = r'\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'
url_all_re = re.compile(url_all_re, flags=re.IGNORECASE)

UNICODE_ELLIPSIS = '\\u2026'  # Ellipsis
UNICODE_IGNORE = [
    UNICODE_ELLIPSIS,
    '\\u3164',  # Hangul Filler
]

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
        'place_id',
        'place_name',
        'place_full_name',
        'place_country',
        'place_country_code',
        'place_type',
    ],
)

EventInfo = namedtuple(
    'EventInfo',
    [
        'timestamp',
        'n',
        'event_str',
        'longitude',
        'latitude',
        'west_lon',
        'south_lat',
        'east_lon',
        'north_lat',
        'place_id',
        'place_name',
        'tokens_str',
        'status_ids',
    ],
)


def get_tweet_body(status):
    if 'extended_tweet' in status:
        tweet_body = status['extended_tweet']['full_text']
    elif 'text' in status:
        tweet_body = status['text']
    else:
        tweet_body = ''
    return tweet_body


def get_tweet_info(status: Dict) -> Dict:
    status_id_str = status['id_str']
    user_screen_name = status['user']['screen_name']
    user_id_str = status['user']['id_str']
    created_at = date_string_to_datetime(status['created_at'])
    tweet_body = get_tweet_body(status)
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
    place_id = status['place'].get('id')
    place_name = status['place'].get('name')
    place_full_name = status['place'].get('full_name')
    place_country = status['place'].get('country')
    place_country_code = status['place'].get('country_code')
    # Possible place_type values: country, admin, city, neighborhood, poi
    place_type = status['place'].get('place_type')

    tweet_info = TweetInfo(
        status_id_str=status_id_str,
        user_screen_name=user_screen_name,
        user_id_str=user_id_str,
        created_at=created_at,
        tweet_body=tweet_body,
        tweet_language=tweet_language,
        longitude=longitude,
        latitude=latitude,
        place_id=place_id,
        place_name=place_name,
        place_full_name=place_full_name,
        place_country=place_country,
        place_country_code=place_country_code,
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
    tweet_body = get_tweet_body(status)

    if not tweet_body:
        return False

    ignore_words_cf = [y.casefold() for y in ignore_words]

    return all([
        all([x.casefold() not in ignore_words_cf for x in clean_text(tweet_body).split()]),
        (status['coordinates'] or (status['place'] and status['place']['place_type'] in valid_place_types)),
        all([re.search(name, status['user']['screen_name'], flags=re.IGNORECASE) is None for name in ignore_user_screen_names]),
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

    # Keep URLs
    if re.search(url_all_re, token) is not None:
        return token

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
    # Remove token if it contains an ellipsis; assume it is a truncated word
    text_cleaned = ' '.join(
        [
            token for token in text.split() if not (UNICODE_ELLIPSIS in token.encode('unicode-escape').decode())
        ]
    )

    # Remove some unicode letters
    text_cleaned = ' '.join(
        [''.join(
            [letter for letter in word if letter.encode('unicode-escape').decode() not in UNICODE_IGNORE]
        ) for word in fix_text(text_cleaned).split()]
    )

    # Decode unicode letters and keep emojis
    text_cleaned = ' '.join(
        [''.join(
            [unidecode(letter) if (str(letter.encode('unicode-escape'))[2] != '\\')
                else letter for letter in word]
        ) for word in text_cleaned.split()]
    )

    # Normalize unicode letters
    # NFKD: decomposes, NFKC: composes pre-combined characters again
    text_cleaned = unicodedata.normalize('NFKC', text_cleaned)

    # Ensure emojis are surrounded by whitespace
    tokens = split_text(text_cleaned)

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


def get_tokens_to_tweet(
    tweets: List,
    token_count_min: int = None,
    remove_username_at: bool = None,
    deduplicate_each_tweet: bool = None,
):
    token_count_min = token_count_min if token_count_min else 2
    remove_username_at = remove_username_at if remove_username_at else True
    deduplicate_each_tweet = deduplicate_each_tweet if deduplicate_each_tweet else True

    try:
        tweets = [remove_urls(x.tweet_body) for x in tweets]
    except AttributeError:
        tweets = [remove_urls(x['tweet_body']) for x in tweets]
    except TypeError:
        logger.exception(f"Unsupported tweet dtype: {type(tweets[0])}")

    tweets = [clean_text(tweet) for tweet in tweets]

    # Pull out user names, hashtags, and emojis
    users_hashtags_emojis = [
        [
            token.lower() for token in split_text(tweet) if is_username_or_hashtag(token) or emoji.emoji_count(token)
        ]
        for tweet in tweets
    ]

    # Get the remaining text
    tweets = [
        ' '.join([
            token for token in split_text(tweet) if (not is_username_or_hashtag(token)) and (not emoji.emoji_count(token))
        ]) for tweet in tweets
    ]

    # Remove stopwords
    tweets = [filter_tokens(tweet) for tweet in tweets]

    # Lowercase and split each tweet into a list
    tweets = [tweet.lower().split() for tweet in tweets]

    # Keep alphanumeric tokens
    tweets = [[token for token in tweet if token.isalnum()] for tweet in tweets]

    # Optionally remove @ from username so user won't be tagged when event is posted
    if remove_username_at:
        users_hashtags_emojis = [
            [
                token.replace('@', '') if is_username(token) else token
                for token in tweet
            ]
            for tweet in users_hashtags_emojis
        ]

    # Combine tokens and usernames/hashtags/emojis from each tweet
    tweets = [t + u for t, u in zip(tweets, users_hashtags_emojis)]

    if deduplicate_each_tweet:
        tweets = [list(dict.fromkeys(tweet)) for tweet in tweets]

    # Flatten list of lists
    tokens = [token for tweet in tweets for token in tweet]

    counter = Counter(tokens).most_common()
    # Get tokens; reduce threshold if no tokens are above the threshold
    tokens_to_tweet = []
    while not tokens_to_tweet and token_count_min > 0:
        # keep those above the threshold
        tokens_to_tweet = [tc[0] for tc in counter if tc[1] >= token_count_min]
        token_count_min -= 1

    if len(tokens_to_tweet) == 0:
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


def reverse_geocode(twitter, longitude: float, latitude: float, sleep_seconds: int = 10) -> Dict:
    # Reverse geocode latitude and longitude value
    geo_granularity = ['neighborhood', 'city', 'admin', 'country']

    unsuccessful_tries = 0
    try_threshold = 10

    rev_geo = {
        'longitude': longitude,
        'latitude': latitude,
    }

    while unsuccessful_tries < try_threshold:
        response = twitter.reverse_geocode(lat=latitude, long=longitude, granularity='neighborhood')
        if 'result' in response:
            unsuccessful_tries = try_threshold
        else:
            unsuccessful_tries += 1
            logger.info(f'Sleeping for {sleep_seconds} seconds due to failed reverse geocode')
            sleep(sleep_seconds)

    for gg in geo_granularity:
        if 'result' in response:
            name = [x['name'] for x in response['result']['places'] if x['place_type'] == gg]
        else:
            name = []
        rev_geo[gg] = name[0] if name else None

    return rev_geo


def get_event_info(
    twitter,
    event_tweets: List,
    tweet_max_length: int,
    tweet_url_length: int,
    base_event_url: str,
    token_count_min: int = None,
    remove_username_at: bool = None,
    tweet_lat_lon: bool = False,
):
    # Compute the average tweet location
    lons, lats = get_coords(event_tweets)
    west_lon = min(lons)
    south_lat = min(lats)
    east_lon = max(lons)
    north_lat = max(lats)
    longitude = sum(lons) / len(lons)
    latitude = sum(lats) / len(lats)

    # Event timestamp is the most recent tweet
    timestamp = max(map(operator.itemgetter('created_at'), event_tweets)).replace(tzinfo=pytz.UTC)

    place_name = get_place_name(event_tweets, valid_place_types=['neighborhood', 'poi'])

    # Get a larger granular name to include after a neighborhood or poi
    city_name = get_place_name(event_tweets, valid_place_types=['city'])

    # If the tweets didn't have a valid place name, reverse geocode to get the neighborhood name
    if place_name is None:
        rev_geo = reverse_geocode(twitter, longitude=longitude, latitude=latitude)
        try:
            place_name = rev_geo['neighborhood']
        except KeyError:
            logger.info("No place name found for event")

    # TODO - only if poi
    place_id = None
    if place_name is not None:
        place_id = None

    # Prepare the tweet text
    tokens_to_tweet = get_tokens_to_tweet(event_tweets, token_count_min=token_count_min, remove_username_at=remove_username_at)
    tokens_str = ' '.join(tokens_to_tweet)

    # Construct the message to tweet
    event_str = "Something's happening"
    event_str = f'{event_str} in {place_name}' if place_name else event_str
    event_str = f'{event_str}, {city_name}' if city_name else event_str
    event_str = f'{event_str} ({latitude:.4f}, {longitude:.4f}):' if tweet_lat_lon else f'{event_str}:'
    remaining_chars = tweet_max_length - len(event_str) - 2 - tweet_url_length
    # Find the largest set of tokens to fill out the remaining charaters
    possible_token_sets = [' '.join(tokens_to_tweet[:i]) for i in range(1, len(tokens_to_tweet) + 1)[::-1]]
    mask = [len(x) <= remaining_chars for x in possible_token_sets]
    tokens = [t for t, m in zip(possible_token_sets, mask) if m][0]

    # tweets are ordered newest to oldest
    coords = ','.join([f'{lon}+{lat}' for lon, lat in zip(lons, lats)])
    status_ids = get_status_ids(event_tweets)
    tweet_ids = ','.join(sorted(status_ids)[::-1])

    urlparams = {
        'words': tokens,
        'coords': coords,
        'tweets': tweet_ids,
    }
    event_url = base_event_url + urllib.parse.urlencode(urlparams)
    event_str = f'{event_str} {tokens} {event_url}'

    logger.info(f'{place_name}: Found event with {len(event_tweets)} tweets: {tokens_str}')
    logger.info(event_str)

    event_info = EventInfo(
        timestamp=timestamp,
        n=len(event_tweets),
        event_str=event_str,
        longitude=longitude,
        latitude=latitude,
        west_lon=west_lon,
        south_lat=south_lat,
        east_lon=east_lon,
        north_lat=north_lat,
        place_id=place_id,
        place_name=place_name,
        tokens_str=tokens_str,
        status_ids=status_ids,
    )

    return event_info
