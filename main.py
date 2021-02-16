from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
from time import sleep
import urllib

from dotenv import load_dotenv
import numpy as np
import pytz
from twython import Twython, TwythonStreamer
from twython import TwythonError, TwythonRateLimitError, TwythonAuthError

from utils.data_base import session_factory, RecentTweets, Events
from utils.tweet_utils import TweetInfo, get_tweet_info, check_tweet, get_tokens_to_tweet, get_coords, get_place_name, get_status_ids
from utils.data_utils import get_grid_coords, inbounds, reverse_geocode, compare_activity_kde
from utils.cluster_utils import cluster_activity

logging.basicConfig(format='{asctime} : {levelname} : {message}', style='{')
logger = logging.getLogger("happeninglogger")

IS_PROD = os.getenv("IS_PROD", default=None)

if IS_PROD is None:
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        raise OSError(f"{env_path} not found. Did you set it up?")

DEBUG_RUN = os.getenv("DEBUG_RUN", default="False")
if DEBUG_RUN not in ["True", "False"]:
    raise ValueError(f"DEBUG_RUN must be True or False, current value: {DEBUG_RUN}")
DEBUG_RUN = DEBUG_RUN == "True"

if DEBUG_RUN:
    logger.setLevel(logging.DEBUG)
    TEMPORAL_GRANULARITY_HOURS = 1
    POST_EVENT = False
    LOG_TWEETS = False
    LOG_STATS = False
    LOG_EVENTS = False
    ECHO = False
else:
    logger.setLevel(logging.INFO)
    TEMPORAL_GRANULARITY_HOURS = int(os.getenv("TEMPORAL_GRANULARITY_HOURS", default="1"))
    POST_EVENT = os.getenv("POST_EVENT", default="False") == "True"
    LOG_TWEETS = True
    LOG_STATS = True
    LOG_EVENTS = True
    ECHO = False

APP_KEY = os.getenv("API_KEY", default=None)
APP_SECRET = os.getenv("API_SECRET", default=None)
OAUTH_TOKEN = os.getenv("ACCESS_TOKEN", default=None)
OAUTH_TOKEN_SECRET = os.getenv("ACCESS_TOKEN_SECRET", default=None)
DATABASE_URL = os.getenv("DATABASE_URL", default=None)
MY_SCREEN_NAME = os.getenv("MY_SCREEN_NAME", default="twitter")
LANGUAGE = os.getenv("LANGUAGE", default="en")
BOUNDING_BOX = os.getenv("BOUNDING_BOX", default=None)
BOUNDING_BOX = [float(coord) for coord in BOUNDING_BOX.split(',')] if BOUNDING_BOX else []
assert len(BOUNDING_BOX) == 4
EVENT_MIN_TWEETS = int(os.getenv("EVENT_MIN_TWEETS", default="9"))
TWEET_MAX_LENGTH = int(os.getenv("TWEET_MAX_LENGTH", default="280"))
TWEET_URL_LENGTH = int(os.getenv("TWEET_URL_LENGTH", default="23"))
RECENT_TWEETS_DAYS_TO_KEEP = float(os.getenv("RECENT_TWEETS_DAYS_TO_KEEP", default="4.0"))
EVENTS_DAYS_TO_KEEP = float(os.getenv("EVENTS_DAYS_TO_KEEP", default="365.0"))
MAX_ROWS_HISTORICAL_STATS = int(os.getenv("MAX_ROWS_HISTORICAL_STATS", default="6000"))
BASE_EVENT_URL = os.getenv("BASE_EVENT_URL", default="https://mattmollison.com/thisishappening/?")

VALID_PLACE_TYPES = os.getenv("VALID_PLACE_TYPES", default="neighborhood, poi")
VALID_PLACE_TYPES = [x.strip() for x in VALID_PLACE_TYPES.split(',')] if VALID_PLACE_TYPES else []
IGNORE_WORDS = os.getenv("IGNORE_WORDS", default=None)
IGNORE_WORDS = [x.strip() for x in IGNORE_WORDS.split(',')] if IGNORE_WORDS else []
IGNORE_USER_SCREEN_NAMES = os.getenv("IGNORE_USER_SCREEN_NAMES", default=None)
IGNORE_USER_SCREEN_NAMES = [x.strip() for x in IGNORE_USER_SCREEN_NAMES.split(',')] if IGNORE_USER_SCREEN_NAMES else []
IGNORE_USER_ID_STR = os.getenv("IGNORE_USER_ID_STR", default=None)
IGNORE_USER_ID_STR = [x.strip() for x in IGNORE_USER_ID_STR.split(',')] if IGNORE_USER_ID_STR else []

TOKEN_COUNT_MIN = int(os.getenv("TOKEN_COUNT_MIN", default="2"))
REMOVE_USERNAME_AT = os.getenv("REMOVE_USERNAME_AT", default="True") == "True"

TEMP_TILE_ID = 0

grid_coords, x_flat, y_flat = get_grid_coords(BOUNDING_BOX)
bw_method = 0.3
weighted = True
weight_factor = 1.0
activity_threshold_day = 500
activity_threshold_hour = 3000


class MyStreamer(TwythonStreamer):
    def __init__(self, running_stats=None, comparison_timestamp=None, *args, **kwargs):
        super(MyStreamer, self).__init__(*args, **kwargs)
        self.running_stats = running_stats if running_stats else {}
        if comparison_timestamp is None:
            comparison_timestamp = datetime.utcnow().replace(tzinfo=pytz.UTC)
        self.comparison_timestamp = comparison_timestamp
        self.sleep_seconds = 2
        self.sleep_exponent = 0

    def on_success(self, status):
        # Reset sleep seconds exponent
        self.sleep_exponent = 0

        if check_tweet(
            status=status,
            valid_place_types=VALID_PLACE_TYPES,
            ignore_words=IGNORE_WORDS,
            ignore_user_screen_names=IGNORE_USER_SCREEN_NAMES,
            ignore_user_id_str=IGNORE_USER_ID_STR,
        ):
            tweet_info = get_tweet_info(status)

            if inbounds(longitude=tweet_info.longitude, latitude=tweet_info.latitude, bounding_box=BOUNDING_BOX):
                if LOG_TWEETS:
                    log_tweet(tweet_info=tweet_info, tile_id=TEMP_TILE_ID)
                else:
                    logger.info(f'Not logging tweet due to environment variable settings: {tweet_info.status_id_str}, {tweet_info.place_name} ({tweet_info.place_type})')

                if tweet_info.created_at - self.comparison_timestamp >= timedelta(hours=TEMPORAL_GRANULARITY_HOURS):
                    logger.info(f'Current tweet time: {tweet_info.created_at}')
                    logger.info(f'Been more than {TEMPORAL_GRANULARITY_HOURS} hours between oldest and current tweet')

                    activity_curr_day = RecentTweets.get_recent_tweets(
                        session,
                        timestamp=tweet_info.created_at,
                        hours=24,
                    )
                    activity_prev_day = RecentTweets.get_recent_tweets(
                        session,
                        timestamp=tweet_info.created_at - timedelta(days=1),
                        hours=24,
                    )

                    activity_curr_hour = RecentTweets.get_recent_tweets(
                        session,
                        timestamp=tweet_info.created_at,
                        hours=TEMPORAL_GRANULARITY_HOURS,
                    )
                    activity_prev_hour = RecentTweets.get_recent_tweets(
                        session,
                        timestamp=tweet_info.created_at - timedelta(hours=TEMPORAL_GRANULARITY_HOURS),
                        hours=TEMPORAL_GRANULARITY_HOURS,
                    )

                    z_diff_day, activity_prev_day, activity_curr_day = compare_activity_kde(
                        grid_coords,
                        activity_prev_day, activity_curr_day,
                        bw_method=bw_method, weighted=weighted, weight_factor=weight_factor,
                    )
                    z_diff_hour, activity_prev_hour, activity_curr_hour = compare_activity_kde(
                        grid_coords,
                        activity_prev_hour, activity_curr_hour,
                        bw_method=bw_method, weighted=weighted, weight_factor=weight_factor,
                    )

                    lat_activity_day, lon_activity_day = np.where(z_diff_day > activity_threshold_day)
                    lat_activity_hour, lon_activity_hour = np.where(z_diff_hour > activity_threshold_hour)

                    # Decide whether an event occurred
                    event_day = False
                    event_hour = False
                    found_event = False
                    if (lat_activity_day.size > 0) and (lon_activity_day.size > 0):
                        event_day = True
                    if (lat_activity_hour.size > 0) and (lon_activity_hour.size > 0):
                        event_hour = True
                    if event_day and event_hour:
                        found_event = True

                    if found_event:
                        sample_weight = [x["weight"] for x in activity_curr_hour]
                        clusters = cluster_activity(activity=activity_curr_hour, min_samples=EVENT_MIN_TWEETS, sample_weight=sample_weight)

                        for cluster in clusters.values():
                            event_info = get_event_info(
                                event_tweets=cluster['event_tweets'],
                                token_count_min=TOKEN_COUNT_MIN,
                            )

                            if LOG_EVENTS:
                                log_event(
                                    cluster['event_tweets'],
                                    tweet_info.created_at,
                                    event_info['longitude'],
                                    event_info['latitude'],
                                    event_info['place_name'],
                                    event_info['tokens_str'],
                                    TEMP_TILE_ID,
                                )
                            else:
                                logger.info(f"Not logging event due to environment variable settings: {tweet_info.created_at} {event_info['place_name']}: {event_info['tokens_str']}")

                            if POST_EVENT:
                                try:
                                    status = twitter.update_status(status=event_info['event_str'])
                                except TwythonAuthError:
                                    logger.exception('Authorization error, did you create read+write credentials?')
                                except TwythonRateLimitError:
                                    logger.exception('Rate limit error')
                                except TwythonError:
                                    logger.exception('Encountered some other error')
                            else:
                                logger.info('Not posting event due to environment variable settings')

                        # Update the comparison tweet time
                        self.comparison_timestamp = tweet_info.created_at

                    # Delete old recent tweets rows
                    logger.info('Deleting old recent tweets')
                    RecentTweets.delete_tweets_older_than(session, timestamp=tweet_info.created_at, days=RECENT_TWEETS_DAYS_TO_KEEP)

                    # Delete old events rows
                    logger.info('Deleting old events')
                    Events.delete_events_older_than(session, timestamp=tweet_info.created_at, days=EVENTS_DAYS_TO_KEEP)
            else:
                logger.info(f'Tweet {tweet_info.status_id_str} out of bounds: coordinates: ({tweet_info.latitude}, {tweet_info.longitude}), {tweet_info.place_name}, {tweet_info.place_type})')

    def on_error(self, status_code, content, headers=None):
        logger.info('Error while streaming.')
        logger.info(f'status_code: {status_code}')
        logger.info(f'content: {content}')
        logger.info(f'headers: {headers}')
        content = content.decode().strip() if isinstance(content, bytes) else content.strip()
        if 'Server overloaded, try again in a few seconds'.lower() in content.lower():
            seconds = self.sleep_seconds ** self.sleep_exponent
            logger.warning(f'Server overloaded. Sleeping for {seconds} seconds.')
            sleep(seconds)
            self.sleep_exponent += 1
        elif 'Exceeded connection limit for user'.lower() in content.lower():
            seconds = self.sleep_seconds ** self.sleep_exponent
            logger.warning(f'Exceeded connection limit. Sleeping for {seconds} seconds.')
            sleep(seconds)
            self.sleep_exponent += 1
        else:
            seconds = self.sleep_seconds ** self.sleep_exponent
            logger.warning(f'Some other error occurred. Sleeping for {seconds} seconds.')
            sleep(seconds)


def log_tweet(tweet_info: TweetInfo, tile_id: int):
    tweet = RecentTweets(
        status_id_str=tweet_info.status_id_str,
        user_screen_name=tweet_info.user_screen_name,
        user_id_str=tweet_info.user_id_str,
        created_at=tweet_info.created_at,
        tweet_body=tweet_info.tweet_body,
        tweet_language=tweet_info.tweet_language,
        longitude=tweet_info.longitude,
        latitude=tweet_info.latitude,
        place_name=tweet_info.place_name,
        place_type=tweet_info.place_type,
        tile_id=tile_id,
    )
    session.add(tweet)
    try:
        session.commit()
        logger.info(f'Logged tweet: {tweet_info.status_id_str}, {tweet_info.place_name} ({tweet_info.place_type})')
    except Exception:
        logger.exception(f'Exception when logging tweet: {tweet_info.status_id_str}, {tweet_info.place_name} ({tweet_info.place_type})')
        session.rollback()


def get_event_info(event_tweets, token_count_min: int = None):
    # Compute the average tweet location
    lons, lats = get_coords(event_tweets)
    longitude = sum(lons) / len(lons)
    latitude = sum(lats) / len(lats)
    lat_long_str = f'{latitude:.4f}, {longitude:.4f}'

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

    # Prepare the tweet text
    tokens_to_tweet = get_tokens_to_tweet(event_tweets, token_count_min=token_count_min, remove_username_at=REMOVE_USERNAME_AT)
    tokens_str = ' '.join(tokens_to_tweet)

    # Construct the message to tweet
    event_str = "Something's happening"
    event_str = f'{event_str} in {place_name}' if place_name else event_str
    event_str = f'{event_str}, {city_name}' if city_name else event_str
    event_str = f'{event_str} ({lat_long_str}):'
    remaining_chars = TWEET_MAX_LENGTH - len(event_str) - 2 - TWEET_URL_LENGTH
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
    event_url = BASE_EVENT_URL + urllib.parse.urlencode(urlparams)
    event_str = f'{event_str} {tokens} {event_url}'

    logger.info(f'{place_name}: Found event with {len(event_tweets)} tweets: {tokens_str}')
    logger.info(event_str)

    event_info = {
        'event_str': event_str,
        'longitude': longitude,
        'latitude': latitude,
        'place_name': place_name,
        'tokens_str': tokens_str,
    }

    return event_info


def log_event(event_tweets, timestamp, longitude, latitude, place_name, tokens_str, tile_id: int):
    status_ids = get_status_ids(event_tweets)

    # Add to events table
    ev = Events(
        tile_id=tile_id,
        timestamp=timestamp,
        count=len(event_tweets),
        longitude=longitude,
        latitude=latitude,
        place_name=place_name,
        description=tokens_str,
        status_ids=status_ids,
    )
    session.add(ev)

    try:
        session.commit()
        logger.info(f'Logged event: {timestamp} {place_name}: {tokens_str}')
    except Exception:
        logger.exception(f'Exception when logging event: {timestamp} {place_name}: {tokens_str}')
        session.rollback()


# Establish connection to Twitter
# Uses OAuth1 ("user auth") for authentication
twitter = Twython(
    app_key=APP_KEY,
    app_secret=APP_SECRET,
    oauth_token=OAUTH_TOKEN,
    oauth_token_secret=OAUTH_TOKEN_SECRET,
)

# Establish connection to database
session = session_factory(DATABASE_URL, echo=ECHO)

# Find out when the last event happened
most_recent_event = Events.get_most_recent_event(session)
if most_recent_event is not None:
    comparison_timestamp = most_recent_event.timestamp.replace(tzinfo=pytz.UTC)
else:
    comparison_timestamp = None

if __name__ == '__main__':
    logger.info('Initializing tweet streamer...')
    stream = MyStreamer(
        comparison_timestamp=comparison_timestamp,
        app_key=APP_KEY,
        app_secret=APP_SECRET,
        oauth_token=OAUTH_TOKEN,
        oauth_token_secret=OAUTH_TOKEN_SECRET,
    )

    bounding_box_str = ','.join([str(x) for x in BOUNDING_BOX])
    logger.info(f'Looking for tweets in bounding box: {bounding_box_str}')
    while True:
        try:
            stream.statuses.filter(locations=bounding_box_str)
        except Exception:
            logger.exception('Exception when streaming tweets')
            continue
