import functools
import logging
import operator
import re
import string
import unicodedata
import urllib
from collections import Counter, namedtuple
from datetime import datetime
from time import sleep
from typing import Dict, List

import emoji
import en_core_web_sm
import pytz
from ftfy import fix_text
from unidecode import unidecode

from .data_utils import inbounds

logger = logging.getLogger("happeninglogger")

# Regex to look for all URLs (mailto:, x-whatever://, etc.)
# https://gist.github.com/gruber/249502
# Removed case insensitive flag from the start: (?i)
url_all_re = (
    r"\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)"
    + r"(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|"
    + r"(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
)
url_all_re = re.compile(url_all_re, flags=re.IGNORECASE)

UNICODE_ELLIPSIS = "\\u2026"  # Ellipsis
UNICODE_IGNORE = [
    UNICODE_ELLIPSIS,
    "\\u3164",  # Hangul Filler
]

nlp = en_core_web_sm.load(exclude=["parser", "ner"])

TweetInfo = namedtuple(
    "TweetInfo",
    [
        "status_id_str",
        "user_screen_name",
        "user_id_str",
        "created_at",
        "deleted_at",
        "tweet_body",
        "tweet_language",
        "is_quote_status",
        "is_reply_status",
        "possibly_sensitive",
        "has_coords",
        "longitude",
        "latitude",
        "place_id",
        "place_name",
        "place_full_name",
        "place_country",
        "place_country_code",
        "place_type",
    ],
)

EventInfo = namedtuple(
    "EventInfo",
    [
        "timestamp",
        "n",
        "event_str",
        "longitude",
        "latitude",
        "west_lon",
        "south_lat",
        "east_lon",
        "north_lat",
        "place_id",
        "place_name",
        "tokens_str",
        "status_ids",
    ],
)


def get_tweet_body(status):
    if "extended_tweet" in status:
        tweet_body = status["extended_tweet"]["full_text"]
    elif "text" in status:
        tweet_body = status["text"]
    else:
        tweet_body = ""
    return tweet_body


def get_lon_lat(status):
    has_coords = False
    if status["coordinates"]:
        has_coords = True
        longitude = status["coordinates"]["coordinates"][0]
        latitude = status["coordinates"]["coordinates"][1]
    elif status["place"]:
        lons = [x[0] for x in status["place"]["bounding_box"]["coordinates"][0]]
        longitude = sum(lons) / len(lons)
        lats = [x[1] for x in status["place"]["bounding_box"]["coordinates"][0]]
        latitude = sum(lats) / len(lats)
    else:
        longitude = None
        latitude = None
    return longitude, latitude, has_coords


def get_place_bounding_box(status):
    if status["coordinates"]:
        place_bounding_box = [
            min([c[0] for c in status["place"]["bounding_box"]["coordinates"][0]]),
            min([c[1] for c in status["place"]["bounding_box"]["coordinates"][0]]),
            max([c[0] for c in status["place"]["bounding_box"]["coordinates"][0]]),
            max([c[1] for c in status["place"]["bounding_box"]["coordinates"][0]]),
        ]
    else:
        place_bounding_box = None
    return place_bounding_box


def get_tweet_info(status: Dict) -> Dict:
    status_id_str = status["id_str"]
    user_screen_name = status["user"]["screen_name"]
    user_id_str = status["user"]["id_str"]
    created_at = date_string_to_datetime(status["created_at"])
    tweet_body = get_tweet_body(status)
    tweet_language = status["lang"]
    is_quote_status = status.get("is_quote_status")
    is_reply_status = status.get("in_reply_to_status_id_str") is not None
    possibly_sensitive = status.get("possibly_sensitive")
    longitude, latitude, has_coords = get_lon_lat(status)
    place_id = status["place"].get("id")
    place_name = status["place"].get("name")
    place_full_name = status["place"].get("full_name")
    place_country = status["place"].get("country")
    place_country_code = status["place"].get("country_code")
    # Possible place_type values: country, admin, city, neighborhood, poi
    place_type = status["place"].get("place_type")

    tweet_info = TweetInfo(
        status_id_str=status_id_str,
        user_screen_name=user_screen_name,
        user_id_str=user_id_str,
        created_at=created_at,
        deleted_at=None,
        tweet_body=tweet_body,
        tweet_language=tweet_language,
        is_quote_status=is_quote_status,
        is_reply_status=is_reply_status,
        possibly_sensitive=possibly_sensitive,
        has_coords=has_coords,
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
    bounding_box: List[float],
    valid_place_types: List[str] = ["admin", "city", "neighborhood", "poi"],
    ignore_words: List[str] = [],
    ignore_user_screen_names: List[str] = [],
    ignore_user_id_str: List[str] = [],
    ignore_lon_lat: List[List[str]] = [],
    ignore_possibly_sensitive: bool = False,
    ignore_quote_status: bool = False,
    ignore_reply_status: bool = False,
    min_friends_count: int = 1,
    min_followers_count: int = 1,
) -> bool:
    """Return True if tweet satisfies specific criteria"""
    tweet_body = get_tweet_body(status)

    if not tweet_body:
        return False

    try:
        quoted_tweet_body = status["quoted_status"]["text"]
    except KeyError:
        quoted_tweet_body = ""

    longitude, latitude, _ = get_lon_lat(status)

    in_bounding_box = inbounds(longitude, latitude, bounding_box)

    # If the tweet has a place, is the tweet's location actually in that place?
    in_place_bounding_box = True
    place_bounding_box = get_place_bounding_box(status)
    if place_bounding_box:
        in_place_bounding_box = inbounds(longitude, latitude, place_bounding_box)

    tweet_ignore_words = all(
        [
            re.search(ignore_word, word, flags=re.IGNORECASE) is None
            for word in clean_text(tweet_body).split()
            for ignore_word in ignore_words
        ]
    )

    quote_tweet_ignore_words = all(
        [
            re.search(ignore_word, word, flags=re.IGNORECASE) is None
            for word in clean_text(quoted_tweet_body).split()
            for ignore_word in ignore_words
        ]
    )

    # Always keep if has coordinates, otherwise keep if has place in valid_place_types
    valid_location = status["coordinates"] or (
        status["place"] and status["place"]["place_type"] in valid_place_types
    )

    valid_screen_name = all(
        [
            re.search(name, status["user"]["screen_name"], flags=re.IGNORECASE) is None
            for name in ignore_user_screen_names
        ]
    )

    valid_user_id = status["user"]["id_str"] not in ignore_user_id_str

    valid_lat_lon = all(
        [
            longitude != lon_lat[0]
            if longitude
            else True and latitude != lon_lat[1]
            if latitude
            else True
            for lon_lat in ignore_lon_lat
        ]
    )

    valid_possibly_sensitive = (
        not status.get("possibly_sensitive", False)
        if ignore_possibly_sensitive
        else True
    )

    valid_quoted = (
        not status.get("is_quote_status", False) if ignore_quote_status else True
    )

    valid_reply = (
        status.get("in_reply_to_status_id_str") is None if ignore_reply_status else True
    )

    # following
    valid_friends_count = status["user"]["friends_count"] >= min_friends_count
    # followers
    valid_followers_count = status["user"]["followers_count"] >= min_followers_count

    checks = {
        "in_bounding_box": in_bounding_box,
        "in_place_bounding_box": in_place_bounding_box,
        "tweet_ignore_words": tweet_ignore_words,
        "quote_tweet_ignore_words": quote_tweet_ignore_words,
        "valid_location": valid_location,
        "valid_screen_name": valid_screen_name,
        "valid_user_id": valid_user_id,
        "valid_lat_lon": valid_lat_lon,
        "valid_possibly_sensitive": valid_possibly_sensitive,
        "valid_quoted": valid_quoted,
        "valid_reply": valid_reply,
        "valid_friends_count": valid_friends_count,
        "valid_followers_count": valid_followers_count,
    }

    for check, value in checks.items():
        if not value:
            logger.debug(f"Tweet {status['id_str']} failed check {str(check)}")

    return all(checks.values())


def date_string_to_datetime(
    date_string: str,
    fmt: str = "%a %b %d %H:%M:%S +0000 %Y",
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
    return text.startswith("@")


def is_hashtag(text):
    return text.startswith("#")


def is_username_or_hashtag(text):
    return any([is_username(text), is_hashtag(text)])


def clean_token(token: str) -> str:
    def remove_punct_from_end(text):
        idx = next((i for i, j in enumerate(reversed(text)) if j.isalnum()), 0)
        return text[:-idx] if idx else text

    # Keep URLs
    if re.search(url_all_re, token) is not None:
        return token

    # Replace some punctuation with space;
    # everything in string.punctuation except: # ' . @
    punct_to_remove = '!"$%&()*+,-/:;<=>?[\\]^_`{|}~'
    punct_to_remove = f"[{re.escape(punct_to_remove)}]"
    token = re.sub(punct_to_remove, " ", token).strip()

    # Remove all periods
    # TODO: why not replace with space?
    token = re.sub(re.escape("."), "", token)

    # Remove possessive "apostrophe s" from usernames and hashtags so they are tweetable
    if is_username_or_hashtag(token):
        token = re.sub(r"(.+)'s$", r"\1", token)

    # Remove any trailing punctuation
    token = remove_punct_from_end(token)

    # The only remaining characters in this token are punctuation; don't keep any
    if all([c in string.punctuation for c in token]):
        token = ""

    return token


def remove_urls(text: str) -> str:
    # Remove URLs
    return re.sub(url_all_re, "", text)


def clean_text(text: str) -> str:
    # Remove token if it contains an ellipsis; assume it is a truncated word
    text_cleaned = " ".join(
        [
            token
            for token in text.split()
            if not (UNICODE_ELLIPSIS in token.encode("unicode-escape").decode())
        ]
    )

    # Remove some unicode letters
    text_cleaned = " ".join(
        [
            "".join(
                [
                    letter
                    for letter in word
                    if letter.encode("unicode-escape").decode() not in UNICODE_IGNORE
                ]
            )
            for word in fix_text(text_cleaned).split()
        ]
    )

    # Decode unicode letters and keep emojis
    text_cleaned = " ".join(
        [
            "".join(
                [
                    unidecode(letter)
                    if (str(letter.encode("unicode-escape"))[2] != "\\")
                    else letter
                    for letter in word
                ]
            )
            for word in text_cleaned.split()
        ]
    )

    # Normalize unicode letters
    # NFKD: decomposes, NFKC: composes pre-combined characters again
    text_cleaned = unicodedata.normalize("NFKC", text_cleaned)

    # Ensure emojis are surrounded by whitespace
    tokens = split_text(text_cleaned)

    # Clean up punctuation
    tokens = [clean_token(token) for token in tokens]

    return " ".join(tokens)


def filter_tokens(text: str, lemmatize: bool = False) -> str:
    def _token_filter(token):
        return not any(
            [
                token.is_punct,
                token.is_space,
                token.is_stop,
                (token.lemma_ == "-PRON-"),
                (len(token.text) <= 1),
            ]
        )

    tokens = [
        token.lemma_ if lemmatize else token.text
        for token in nlp(text)
        if _token_filter(token)
    ]

    return " ".join(tokens)


def get_tokens_to_tweet(
    tweets: List,
    token_count_min: int = None,
    remove_username_at: bool = None,
    deduplicate_each_tweet: bool = None,
):
    token_count_min = token_count_min or 2
    remove_username_at = remove_username_at or True
    deduplicate_each_tweet = deduplicate_each_tweet or True

    try:
        tweets = [remove_urls(x.tweet_body) for x in tweets]
    except AttributeError:
        tweets = [remove_urls(x["tweet_body"]) for x in tweets]
    except TypeError:
        logger.exception(f"Unsupported tweet dtype: {type(tweets[0])}")

    tweets = [clean_text(tweet) for tweet in tweets]

    # Pull out user names, hashtags, and emojis
    users_hashtags_emojis = [
        [
            token.lower()
            for token in split_text(tweet)
            if is_username_or_hashtag(token) or emoji.emoji_count(token)
        ]
        for tweet in tweets
    ]

    # Get the remaining text
    tweets = [
        " ".join(
            [
                token
                for token in split_text(tweet)
                if (not is_username_or_hashtag(token))
                and (not emoji.emoji_count(token))
            ]
        )
        for tweet in tweets
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
            [token.replace("@", "") if is_username(token) else token for token in tweet]
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
        tokens_to_tweet = ["No tweet text found"]

    return tokens_to_tweet


def get_coords(tweets):
    try:
        lons = [x.longitude for x in tweets]
        lats = [x.latitude for x in tweets]
    except AttributeError:
        lons = [x["longitude"] for x in tweets]
        lats = [x["latitude"] for x in tweets]
    except TypeError:
        logger.exception(f"Unsupported tweet dtype: {type(tweets[0])}")

    return lons, lats


def get_place_name(tweets, valid_place_types: List[str] = ["neighborhood", "poi"]):
    # Get the most common place name from these tweets;
    # only consider neighborhood or poi
    try:
        place_names = [
            x.place_name for x in tweets if x.place_type in valid_place_types
        ]
    except AttributeError:
        place_names = [
            x["place_name"] for x in tweets if x["place_type"] in valid_place_types
        ]
    except TypeError:
        logger.exception(f"Unsupported tweet dtype: {type(tweets[0])}")

    place_name = Counter(place_names).most_common(1)[0][0] if place_names else None

    return place_name


def get_status_ids(tweets):
    try:
        status_ids = [x.status_id_str for x in tweets]
    except AttributeError:
        status_ids = [x["status_id_str"] for x in tweets]
    except TypeError:
        logger.exception(f"Unsupported tweet dtype: {type(tweets[0])}")

    return status_ids


def reverse_geocode(
    twitter, longitude: float, latitude: float, sleep_seconds: int = 10
) -> Dict:
    # Reverse geocode latitude and longitude value
    geo_granularity = ["neighborhood", "city", "admin", "country"]

    unsuccessful_tries = 0
    try_threshold = 10

    rev_geo = {
        "longitude": longitude,
        "latitude": latitude,
    }

    while unsuccessful_tries < try_threshold:
        response = twitter.reverse_geocode(
            lat=latitude, long=longitude, granularity="neighborhood"
        )
        if "result" in response:
            unsuccessful_tries = try_threshold
        else:
            unsuccessful_tries += 1
            logger.info(
                f"Sleeping for {sleep_seconds} seconds due to failed reverse geocode"
            )
            sleep(sleep_seconds)

    for gg in geo_granularity:
        if "result" in response:
            name = [
                x["name"] for x in response["result"]["places"] if x["place_type"] == gg
            ]
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
    timestamp = max(map(operator.itemgetter("created_at"), event_tweets)).replace(
        tzinfo=pytz.UTC
    )

    place_name = get_place_name(event_tweets, valid_place_types=["neighborhood", "poi"])

    # Get a larger granular name to include after a neighborhood or poi
    city_name = get_place_name(event_tweets, valid_place_types=["city"])

    # If the tweets didn't have a valid place name,
    # reverse geocode to get the neighborhood name
    if place_name is None:
        rev_geo = reverse_geocode(twitter, longitude=longitude, latitude=latitude)
        try:
            place_name = rev_geo["neighborhood"]
        except KeyError:
            logger.info("No place name found for event")

    # TODO - only if poi
    place_id = None
    if place_name is not None:
        place_id = None

    # Prepare the tweet text
    tokens_to_tweet = get_tokens_to_tweet(
        event_tweets,
        token_count_min=token_count_min,
        remove_username_at=remove_username_at,
    )
    tokens_str = " ".join(tokens_to_tweet)

    # Construct the message to tweet
    event_str = "Something's happening"
    event_str = f"{event_str} in {place_name}" if place_name else event_str
    event_str = f"{event_str}, {city_name}" if city_name else event_str
    event_str = (
        f"{event_str} ({latitude:.4f}, {longitude:.4f}):"
        if tweet_lat_lon
        else f"{event_str}:"
    )
    remaining_chars = tweet_max_length - len(event_str) - 2 - tweet_url_length
    # Find the largest set of tokens to fill out the remaining charaters
    possible_token_sets = [
        " ".join(tokens_to_tweet[:i]) for i in range(1, len(tokens_to_tweet) + 1)[::-1]
    ]
    mask = [len(x) <= remaining_chars for x in possible_token_sets]
    tokens = [t for t, m in zip(possible_token_sets, mask) if m][0]

    # tweets are ordered newest to oldest
    coords = ",".join([f"{lon}+{lat}" for lon, lat in zip(lons, lats)])
    status_ids = get_status_ids(event_tweets)
    tweet_ids = ",".join(sorted(status_ids)[::-1])

    urlparams = {
        "words": tokens,
        "coords": coords,
        "tweets": tweet_ids,
    }
    event_url = base_event_url + urllib.parse.urlencode(urlparams)
    event_str = f"{event_str} {tokens} {event_url}"

    logger.info(
        f"{place_name}: Found event with {len(event_tweets)} tweets: {tokens_str}"
    )
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
