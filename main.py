from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
from time import sleep

from dotenv import load_dotenv
import numpy as np
import pytz
from twython import (
    Twython,
    TwythonStreamer,
    TwythonError,
    TwythonRateLimitError,
    TwythonAuthError,
)

from utils.data_base import session_factory, RecentTweets, Events
from utils.tweet_utils import date_string_to_datetime, check_tweet, get_tweet_info, get_event_info
from utils.data_utils import get_grid_coords, inbounds, compare_activity_kde
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

DEBUG_RUN = os.getenv("DEBUG_RUN", default="False").casefold()
if DEBUG_RUN not in ["true".casefold(), "false".casefold()]:
    raise ValueError(f"DEBUG_RUN must be True or False, current value: {DEBUG_RUN}")
DEBUG_RUN = DEBUG_RUN == "true".casefold()

if DEBUG_RUN:
    logger.setLevel(logging.DEBUG)
    TEMPORAL_GRANULARITY_HOURS = 1
    POST_EVENT = False
    LOG_TWEETS = False
    LOG_EVENTS = False
    PURGE_OLD_DATA = False
    RECENT_TWEETS_ROWS_TO_KEEP = None
    EVENTS_ROWS_TO_KEEP = None
    RECENT_TWEETS_DAYS_TO_KEEP = None
    EVENTS_DAYS_TO_KEEP = None
    ECHO = False
else:
    logger.setLevel(logging.INFO)
    TEMPORAL_GRANULARITY_HOURS = int(os.getenv("TEMPORAL_GRANULARITY_HOURS", default="1"))
    POST_EVENT = os.getenv("POST_EVENT", default="False").casefold() == "true".casefold()
    LOG_TWEETS = True
    LOG_EVENTS = True
    PURGE_OLD_DATA = True
    RECENT_TWEETS_ROWS_TO_KEEP = os.getenv("RECENT_TWEETS_ROWS_TO_KEEP", default=None)
    RECENT_TWEETS_ROWS_TO_KEEP = int(RECENT_TWEETS_ROWS_TO_KEEP) if RECENT_TWEETS_ROWS_TO_KEEP else None
    EVENTS_ROWS_TO_KEEP = os.getenv("EVENTS_ROWS_TO_KEEP", default=None)
    EVENTS_ROWS_TO_KEEP = int(EVENTS_ROWS_TO_KEEP) if EVENTS_ROWS_TO_KEEP else None
    RECENT_TWEETS_DAYS_TO_KEEP = os.getenv("RECENT_TWEETS_DAYS_TO_KEEP", default=None)
    RECENT_TWEETS_DAYS_TO_KEEP = float(RECENT_TWEETS_DAYS_TO_KEEP) if RECENT_TWEETS_DAYS_TO_KEEP else None
    EVENTS_DAYS_TO_KEEP = os.getenv("EVENTS_DAYS_TO_KEEP", default=None)
    EVENTS_DAYS_TO_KEEP = float(EVENTS_DAYS_TO_KEEP) if EVENTS_DAYS_TO_KEEP else None
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
EVENT_MIN_TWEETS = int(os.getenv("EVENT_MIN_TWEETS", default="5"))
KM_START = float(os.getenv("KM_START", default="0.05"))
KM_STOP = float(os.getenv("KM_STOP", default="0.3"))
KM_STEP = int(os.getenv("KM_STEP", default="5"))
MIN_N_CLUSTERS = int(os.getenv("MIN_N_CLUSTERS", default="1"))
TWEET_MAX_LENGTH = int(os.getenv("TWEET_MAX_LENGTH", default="280"))
TWEET_URL_LENGTH = int(os.getenv("TWEET_URL_LENGTH", default="23"))
# Use docs/index.html to render words and map of tweets
BASE_EVENT_URL = os.getenv("BASE_EVENT_URL", default="https://USERNAME.github.io/thisishappening/?")

VALID_PLACE_TYPES = os.getenv("VALID_PLACE_TYPES", default="neighborhood, poi")
VALID_PLACE_TYPES = [x.strip() for x in VALID_PLACE_TYPES.split(',')] if VALID_PLACE_TYPES else []
IGNORE_WORDS = os.getenv("IGNORE_WORDS", default=None)
IGNORE_WORDS = [x.strip() for x in IGNORE_WORDS.split(',')] if IGNORE_WORDS else []
IGNORE_USER_SCREEN_NAMES = os.getenv("IGNORE_USER_SCREEN_NAMES", default=None)
IGNORE_USER_SCREEN_NAMES = [x.strip() for x in IGNORE_USER_SCREEN_NAMES.split(',')] if IGNORE_USER_SCREEN_NAMES else []
IGNORE_USER_ID_STR = os.getenv("IGNORE_USER_ID_STR", default=None)
IGNORE_USER_ID_STR = [x.strip() for x in IGNORE_USER_ID_STR.split(',')] if IGNORE_USER_ID_STR else []

TOKEN_COUNT_MIN = int(os.getenv("TOKEN_COUNT_MIN", default="2"))
REMOVE_USERNAME_AT = os.getenv("REMOVE_USERNAME_AT", default="True").casefold() == "true".casefold()

GRID_RESOLUTION = int(os.getenv("GRID_RESOLUTION", default="128"))
BW_METHOD = float(os.getenv("BW_METHOD", default="0.3"))
WEIGHTED = os.getenv("WEIGHTED", default="True").casefold() == "true".casefold()
WEIGHT_FACTOR = float(os.getenv("WEIGHT_FACTOR", default="1.0"))
ACTIVITY_THRESHOLD_DAY = float(os.getenv("ACTIVITY_THRESHOLD_DAY", default="100.0"))
ACTIVITY_THRESHOLD_HOUR = float(os.getenv("ACTIVITY_THRESHOLD_HOUR", default="1000.0"))


class MyStreamer(TwythonStreamer):
    def __init__(self, grid_coords, event_comparison_ts=None, *args, **kwargs):
        super(MyStreamer, self).__init__(*args, **kwargs)
        self.grid_coords = grid_coords
        if event_comparison_ts is None:
            event_comparison_ts = datetime.utcnow().replace(tzinfo=pytz.UTC) - timedelta(hours=TEMPORAL_GRANULARITY_HOURS)
        self.event_comparison_ts = event_comparison_ts
        self.purge_data_comparison_ts = datetime.utcnow().replace(tzinfo=pytz.UTC)
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
                    _ = RecentTweets.log_tweet(session, tweet_info=tweet_info)
                else:
                    logger.info(f'Not logging tweet due to environment variable settings: {tweet_info.status_id_str}, {tweet_info.place_name} ({tweet_info.place_type})')

                if tweet_info.created_at - self.event_comparison_ts >= timedelta(hours=TEMPORAL_GRANULARITY_HOURS):
                    logger.info(f'{tweet_info.created_at} Been more than {TEMPORAL_GRANULARITY_HOURS} hour(s) since an event occurred, comparing activity...')

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

                    # Decide whether an event occurred
                    event_day = False
                    event_hour = False

                    if (len(activity_prev_day) > 1) and (len(activity_curr_day) > 1):
                        z_diff_day, _, _ = compare_activity_kde(
                            self.grid_coords,
                            activity_prev_day, activity_curr_day,
                            bw_method=BW_METHOD, weighted=WEIGHTED, weight_factor=WEIGHT_FACTOR,
                        )

                        lat_activity_day, lon_activity_day = np.where(z_diff_day > ACTIVITY_THRESHOLD_DAY)

                        if (lat_activity_day.size > 0) and (lon_activity_day.size > 0):
                            event_day = True

                        logger.info(f"Day event: {event_day}, current: {len(activity_curr_day)}, previous: {len(activity_prev_day)}, max diff: {z_diff_day.max():.2f}, threshold: {ACTIVITY_THRESHOLD_DAY}")

                    if (len(activity_prev_hour) > 1) and (len(activity_curr_hour) > 1):
                        z_diff_hour, _, activity_curr_hour_w = compare_activity_kde(
                            self.grid_coords,
                            activity_prev_hour, activity_curr_hour,
                            bw_method=BW_METHOD, weighted=WEIGHTED, weight_factor=WEIGHT_FACTOR,
                        )

                        lat_activity_hour, lon_activity_hour = np.where(z_diff_hour > ACTIVITY_THRESHOLD_HOUR)

                        if (lat_activity_hour.size > 0) and (lon_activity_hour.size > 0):
                            event_hour = True

                        logger.info(f"Hour event: {event_hour}, current: {len(activity_curr_hour)}, previous: {len(activity_prev_hour)}, max diff: {z_diff_hour.max():.2f}, threshold: {ACTIVITY_THRESHOLD_HOUR}")

                    if event_day and event_hour:
                        sample_weight = [x["weight"] for x in activity_curr_hour_w]
                        clusters = cluster_activity(
                            activity=activity_curr_hour_w,
                            min_samples=EVENT_MIN_TWEETS,
                            km_start=KM_START,
                            km_stop=KM_STOP,
                            km_step=KM_STEP,
                            min_n_clusters=MIN_N_CLUSTERS,
                            sample_weight=sample_weight,
                        )

                        for cluster in clusters.values():
                            event_info = get_event_info(
                                twitter,
                                event_tweets=cluster['event_tweets'],
                                tweet_max_length=TWEET_MAX_LENGTH,
                                tweet_url_length=TWEET_URL_LENGTH,
                                base_event_url=BASE_EVENT_URL,
                                token_count_min=TOKEN_COUNT_MIN,
                                remove_username_at=REMOVE_USERNAME_AT,
                            )

                            if LOG_EVENTS:
                                _ = Events.log_event(session, event_info=event_info)
                            else:
                                logger.info(f"Not logging event due to environment variable settings: {tweet_info.created_at} {event_info.place_name}: {event_info.tokens_str}")

                            if POST_EVENT:
                                try:
                                    status = twitter.update_status(
                                        status=event_info.event_str,
                                        lat=event_info.latitude,
                                        long=event_info.longitude,
                                        # place_id=event_info.place_id,
                                    )
                                except TwythonAuthError:
                                    logger.exception('Authorization error, did you create read+write credentials?')
                                except TwythonRateLimitError:
                                    logger.exception('Rate limit error')
                                except TwythonError:
                                    logger.exception('Encountered some other error')
                            else:
                                logger.info('Not posting event due to environment variable settings')

                        # Update the comparison tweet time
                        self.event_comparison_ts = tweet_info.created_at

                    # Purge old data every so often
                    if PURGE_OLD_DATA and (datetime.utcnow().replace(tzinfo=pytz.UTC) - self.purge_data_comparison_ts >= timedelta(minutes=10)):
                        # Delete old data by row count
                        RecentTweets.keep_tweets_n_rows(session, n=RECENT_TWEETS_ROWS_TO_KEEP)
                        Events.keep_events_n_rows(session, n=EVENTS_ROWS_TO_KEEP)

                        # Delete old data by timestamp
                        RecentTweets.delete_tweets_older_than(session, timestamp=tweet_info.created_at, days=RECENT_TWEETS_DAYS_TO_KEEP)
                        Events.delete_events_older_than(session, timestamp=tweet_info.created_at, days=EVENTS_DAYS_TO_KEEP)

                        # Update
                        self.purge_data_comparison_ts = datetime.utcnow().replace(tzinfo=pytz.UTC)
            else:
                logger.info(f'Tweet {tweet_info.status_id_str} out of bounds: coordinates: ({tweet_info.latitude}, {tweet_info.longitude}), {tweet_info.place_name} ({tweet_info.place_type})')

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
            self.sleep_exponent += 1


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
# First check the database
most_recent_event = Events.get_most_recent_event(session)
if most_recent_event is not None:
    event_comparison_ts = most_recent_event.timestamp.replace(tzinfo=pytz.UTC)
else:
    # If no db events, use most recent tweet timestamp as the time of the last event
    most_recent_tweet = twitter.get_user_timeline(screen_name=MY_SCREEN_NAME, count=1, trim_user=True)
    if len(most_recent_tweet) > 0:
        event_comparison_ts = date_string_to_datetime(most_recent_tweet[0]['created_at'])
    else:
        event_comparison_ts = None

if __name__ == '__main__':
    logger.info('Initializing tweet streamer...')
    grid_coords, _, _ = get_grid_coords(bounding_box=BOUNDING_BOX, grid_resolution=GRID_RESOLUTION)
    stream = MyStreamer(
        grid_coords=grid_coords,
        event_comparison_ts=event_comparison_ts,
        app_key=APP_KEY,
        app_secret=APP_SECRET,
        oauth_token=OAUTH_TOKEN,
        oauth_token_secret=OAUTH_TOKEN_SECRET,
    )

    bounding_box_str = ','.join([str(x) for x in BOUNDING_BOX])
    logger.info(f'Looking for tweets in bounding box: {bounding_box_str}')
    while True:
        # Use try/except to avoid ChunkedEncodingError
        # https://github.com/ryanmcgrath/twython/issues/288
        try:
            stream.statuses.filter(locations=bounding_box_str)
        except Exception:
            logger.exception('Exception when streaming tweets')
            continue
