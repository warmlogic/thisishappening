import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytz
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed
from twython import (
    Twython,
    TwythonAuthError,
    TwythonError,
    TwythonRateLimitError,
    TwythonStreamer,
)
from utils.cluster_utils import cluster_activity
from utils.data_base import Events, RecentTweets, session_factory
from utils.data_utils import compare_activity_kde, get_grid_coords
from utils.tweet_utils import (
    check_tweet,
    date_string_to_datetime,
    get_event_info,
    get_place_bounding_box,
    get_tweet_info,
)

logging.basicConfig(format="{asctime} : {levelname} : {message}", style="{")
logger = logging.getLogger("happening_logger")

ENVIRONMENT = os.getenv("ENVIRONMENT", default="development").lower()
assert ENVIRONMENT in [
    "development",
    "production",
], f"Invalid ENVIRONMENT: {ENVIRONMENT}"

# Read .env file for local development
if ENVIRONMENT == "development":
    if (Path.cwd() / "data").exists():
        root_dir = Path.cwd()
    elif (Path.cwd().parent / "data").exists():
        root_dir = Path.cwd().parent
    else:
        raise OSError(f"Running from unsupported directory: {Path.cwd()}")

    dotenv_file = root_dir / ".env"
    try:
        with open(dotenv_file, "r") as fp:
            _ = load_dotenv(stream=fp)
    except FileNotFoundError:
        logger.info(f"{dotenv_file} file not found. Did you set it up?")
        raise

DEBUG_MODE = os.getenv("DEBUG_MODE", default="true").lower() == "true"

if DEBUG_MODE:
    logger.setLevel(logging.DEBUG)
    POST_EVENT = False
    POST_DAILY_EVENTS = False
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
    POST_EVENT = os.getenv("POST_EVENT", default="false").lower() == "true"
    POST_DAILY_EVENTS = (
        os.getenv("POST_DAILY_EVENTS", default="false").lower() == "true"
    )
    LOG_TWEETS = True
    LOG_EVENTS = True
    PURGE_OLD_DATA = True
    RECENT_TWEETS_ROWS_TO_KEEP = os.getenv("RECENT_TWEETS_ROWS_TO_KEEP", default=None)
    RECENT_TWEETS_ROWS_TO_KEEP = (
        int(RECENT_TWEETS_ROWS_TO_KEEP) if RECENT_TWEETS_ROWS_TO_KEEP else None
    )
    EVENTS_ROWS_TO_KEEP = os.getenv("EVENTS_ROWS_TO_KEEP", default=None)
    EVENTS_ROWS_TO_KEEP = int(EVENTS_ROWS_TO_KEEP) if EVENTS_ROWS_TO_KEEP else None
    RECENT_TWEETS_DAYS_TO_KEEP = os.getenv("RECENT_TWEETS_DAYS_TO_KEEP", default=None)
    RECENT_TWEETS_DAYS_TO_KEEP = (
        float(RECENT_TWEETS_DAYS_TO_KEEP) if RECENT_TWEETS_DAYS_TO_KEEP else None
    )
    EVENTS_DAYS_TO_KEEP = os.getenv("EVENTS_DAYS_TO_KEEP", default=None)
    EVENTS_DAYS_TO_KEEP = float(EVENTS_DAYS_TO_KEEP) if EVENTS_DAYS_TO_KEEP else None
    ECHO = False

APP_KEY = os.getenv("API_KEY", default=None)
APP_SECRET = os.getenv("API_SECRET", default=None)
OAUTH_TOKEN = os.getenv("ACCESS_TOKEN", default=None)
OAUTH_TOKEN_SECRET = os.getenv("ACCESS_TOKEN_SECRET", default=None)
DATABASE_URL = os.getenv("DATABASE_URL", default=None)
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://")
RETRY_WAIT_SECONDS = int(os.getenv("RETRY_WAIT_SECONDS", default="60"))
MY_SCREEN_NAME = os.getenv("MY_SCREEN_NAME", default=None)
assert MY_SCREEN_NAME is not None
LANGUAGE = os.getenv("LANGUAGE", default="en")
TIMEZONE = os.getenv("TIMEZONE", default="UTC")  # http://pytz.sourceforge.net/#helpers
# Longitude and latitude pairs for SW and NE corner (in that order)
BOUNDING_BOX = os.getenv("BOUNDING_BOX", default=None)
BOUNDING_BOX = (
    [float(coord) for coord in BOUNDING_BOX.split(",")] if BOUNDING_BOX else []
)
assert len(BOUNDING_BOX) == 4
TEMPORAL_GRANULARITY_HOURS = float(os.getenv("TEMPORAL_GRANULARITY_HOURS", default="1"))
MIN_HOURS_BETWEEN_EVENTS = float(os.getenv("MIN_HOURS_BETWEEN_EVENTS", default="1"))
EVENT_MIN_TWEETS = int(os.getenv("EVENT_MIN_TWEETS", default="5"))
MIN_N_CLUSTERS = int(os.getenv("MIN_N_CLUSTERS", default="1"))
DAILY_EVENT_MIN_TWEETS = int(os.getenv("DAILY_EVENT_MIN_TWEETS", default="8"))
DAILY_MIN_N_CLUSTERS = int(os.getenv("DAILY_MIN_N_CLUSTERS", default="2"))
DAILY_EVENT_HOUR = int(os.getenv("DAILY_EVENT_HOUR", default="22"))
KM_START = float(os.getenv("KM_START", default="0.05"))
KM_STOP = float(os.getenv("KM_STOP", default="0.25"))
KM_STEP = int(os.getenv("KM_STEP", default="5"))
TWEET_MAX_LENGTH = int(os.getenv("TWEET_MAX_LENGTH", default="280"))
TWEET_URL_LENGTH = int(os.getenv("TWEET_URL_LENGTH", default="23"))
TWEET_LAT_LON = os.getenv("TWEET_LAT_LON", default="false").lower() == "true"
SHOW_TWEETS_ON_EVENT = (
    os.getenv("SHOW_TWEETS_ON_EVENT", default="true").lower() == "true"
)
TWEET_GEOTAG = os.getenv("TWEET_GEOTAG", default="true").lower() == "true"
# Use docs/index.html to render words and map of tweets
BASE_EVENT_URL = os.getenv(
    "BASE_EVENT_URL", default="https://USERNAME.github.io/thisishappening/?"
)

VALID_PLACE_TYPES = os.getenv(
    "VALID_PLACE_TYPES", default="admin, city, neighborhood, poi"
)
VALID_PLACE_TYPES = (
    [x.strip() for x in VALID_PLACE_TYPES.split(",")] if VALID_PLACE_TYPES else []
)
VALID_PLACE_TYPES = list(set(VALID_PLACE_TYPES))
IGNORE_WORDS = os.getenv("IGNORE_WORDS", default=None)
IGNORE_WORDS = (
    [rf"\b{re.escape(x.strip())}\b" for x in IGNORE_WORDS.split(",")]
    if IGNORE_WORDS
    else []
)
IGNORE_WORDS = list(set(IGNORE_WORDS))
# swap order of hashtag marker (#) and beginning of word marker (\b)
IGNORE_WORDS = [
    re.sub(r"(\\b)(\\#)(.*)", r"\2\1\3", w) if w.startswith("\\b\\#") else w
    for w in IGNORE_WORDS
]
IGNORE_USER_SCREEN_NAMES = os.getenv("IGNORE_USER_SCREEN_NAMES", default=None)
IGNORE_USER_SCREEN_NAMES = (
    [rf"{x.strip()}" for x in IGNORE_USER_SCREEN_NAMES.split(",")]
    if IGNORE_USER_SCREEN_NAMES
    else []
)
IGNORE_USER_SCREEN_NAMES.append(MY_SCREEN_NAME)  # Ignore tweets from own screen name
IGNORE_USER_SCREEN_NAMES = list(set(IGNORE_USER_SCREEN_NAMES))
IGNORE_USER_ID_STR = os.getenv("IGNORE_USER_ID_STR", default=None)
IGNORE_USER_ID_STR = (
    [x.strip() for x in IGNORE_USER_ID_STR.split(",")] if IGNORE_USER_ID_STR else []
)
IGNORE_USER_ID_STR = list(set(IGNORE_USER_ID_STR))

MIN_FRIENDS_COUNT = int(os.getenv("MIN_FRIENDS_COUNT", default="1"))
MIN_FOLLOWERS_COUNT = int(os.getenv("MIN_FOLLOWERS_COUNT", default="1"))
IGNORE_POSSIBLY_SENSITIVE = (
    os.getenv("IGNORE_POSSIBLY_SENSITIVE", default="false").lower() == "true"
)
IGNORE_QUOTE_STATUS = (
    os.getenv("IGNORE_QUOTE_STATUS", default="false").lower() == "true"
)
IGNORE_REPLY_STATUS = (
    os.getenv("IGNORE_REPLY_STATUS", default="false").lower() == "true"
)

IGNORE_LON_LAT = os.getenv("IGNORE_LON_LAT", default=None)
IGNORE_LON_LAT = (
    [
        (float(c[0].strip()), float(c[1].strip()))
        for c in [coords.split(",") for coords in IGNORE_LON_LAT.split(";")]
    ]
    if IGNORE_LON_LAT
    else []
)
IGNORE_LON_LAT = list(set(IGNORE_LON_LAT))

TOKEN_COUNT_MIN = int(os.getenv("TOKEN_COUNT_MIN", default="2"))
REDUCE_TOKEN_COUNT_MIN = (
    os.getenv("REDUCE_TOKEN_COUNT_MIN", default="true").lower() == "true"
)
REMOVE_USERNAME_AT = os.getenv("REMOVE_USERNAME_AT", default="true").lower() == "true"

GRID_RESOLUTION_KM = float(os.getenv("GRID_RESOLUTION", default="0.25"))
BW_METHOD = float(os.getenv("BW_METHOD", default="0.3"))
ACTIVITY_THRESHOLD_DAY = float(os.getenv("ACTIVITY_THRESHOLD_DAY", default="1.0"))
ACTIVITY_THRESHOLD_HOUR = float(os.getenv("ACTIVITY_THRESHOLD_HOUR", default="50.0"))

WEIGHTED = os.getenv("WEIGHTED", default="false").lower() == "true"
REDUCE_WEIGHT_LON_LAT = os.getenv("REDUCE_WEIGHT_LON_LAT", default=None)
REDUCE_WEIGHT_LON_LAT = (
    [
        (f"{float(c[0].strip()):.5f}", f"{float(c[1].strip()):.5f}")
        for c in [coords.split(",") for coords in REDUCE_WEIGHT_LON_LAT.split(";")]
    ]
    if REDUCE_WEIGHT_LON_LAT
    else []
)
REDUCE_WEIGHT_LON_LAT = list(set(REDUCE_WEIGHT_LON_LAT))
WEIGHT_FACTOR_LON_LAT = os.getenv("WEIGHT_FACTOR_LON_LAT", default=None)
WEIGHT_FACTOR_LON_LAT = float(WEIGHT_FACTOR_LON_LAT) if WEIGHT_FACTOR_LON_LAT else None
WEIGHT_FACTOR_USER = os.getenv("WEIGHT_FACTOR_USER", default=None)
WEIGHT_FACTOR_USER = float(WEIGHT_FACTOR_USER) if WEIGHT_FACTOR_USER else None
WEIGHT_FACTOR_NO_COORDS = os.getenv("WEIGHT_FACTOR_NO_COORDS", default=None)
WEIGHT_FACTOR_NO_COORDS = (
    float(WEIGHT_FACTOR_NO_COORDS) if WEIGHT_FACTOR_NO_COORDS else None
)

QUERY_HAS_COORDS_ONLY = (
    os.getenv("QUERY_HAS_COORDS_ONLY", default="false").lower() == "true"
)
QUERY_HAS_COORDS_ONLY = QUERY_HAS_COORDS_ONLY if QUERY_HAS_COORDS_ONLY else None

QUERY_INCLUDE_QUOTE_STATUS = (
    os.getenv("QUERY_INCLUDE_QUOTE_STATUS", default="true").lower() == "true"
)
QUERY_INCLUDE_REPLY_STATUS = (
    os.getenv("QUERY_INCLUDE_REPLY_STATUS", default="false").lower() == "true"
)
QUERY_INCLUDE_DELETED_STATUS = (
    os.getenv("QUERY_INCLUDE_DELETED_STATUS", default="false").lower() == "true"
)


class MyTwitterClient(Twython):
    """Wrapper around the Twython Twitter client."""

    DEFAULT_LAST_POST_TIME = datetime(1970, 1, 1).replace(tzinfo=pytz.UTC)

    def __init__(self, *args, **kwargs):
        super(MyTwitterClient, self).__init__(*args, **kwargs)

    @retry(wait=wait_fixed(RETRY_WAIT_SECONDS))
    def get_last_post_time(self):
        """get the time of our screen_name's most recent tweet"""
        if DEBUG_MODE:
            return self.DEFAULT_LAST_POST_TIME

        try:
            most_recent_tweet = self.get_user_timeline(
                screen_name=MY_SCREEN_NAME, count=1, trim_user=True
            )
            if len(most_recent_tweet) > 0:
                last_post_time = date_string_to_datetime(
                    most_recent_tweet[0]["created_at"]
                )
            else:
                last_post_time = None
        except TwythonRateLimitError as e:
            logger.info(f"Rate limit exceeded when getting recent tweet: {e}")
            raise
        except Exception as e:
            logger.info(f"Exception when getting recent tweet: {e}")
            last_post_time = self.DEFAULT_LAST_POST_TIME

        return last_post_time

    @retry(wait=wait_fixed(RETRY_WAIT_SECONDS), stop=stop_after_attempt(3))
    def _update_status(self, *args, **kwargs):
        try:
            _ = self.update_status(*args, **kwargs)
            logger.info("Successfully updated status")
        except TwythonAuthError as e:
            logger.info(
                f"Authorization error. Did you create read+write credentials? {e}"
            )
            raise
        except TwythonRateLimitError as e:
            logger.info(f"Rate limit exceeded when posting event: {e}")
            raise
        except TwythonError as e:
            logger.info(f"Encountered some other error: {e}")
            raise


class MyStreamer(TwythonStreamer):
    def __init__(
        self,
        twitter,
        db_session,
        bounding_box,
        *args,
        **kwargs,
    ):
        super(MyStreamer, self).__init__(*args, **kwargs)
        self.twitter = twitter
        self.db_session = db_session
        self.grid_coords = get_grid_coords(
            bounding_box=bounding_box, grid_resolution_km=GRID_RESOLUTION_KM
        )
        self.bounding_box_str = ",".join([str(x) for x in bounding_box])
        self.event_comparison_ts = self.get_event_comparison_ts()
        self.purge_data_comparison_ts = datetime.utcnow().replace(tzinfo=pytz.UTC)
        self.posted_daily_events = False

    @retry(wait=wait_fixed(RETRY_WAIT_SECONDS))
    def stream_tweets(self):
        # Use try/except to avoid ChunkedEncodingError
        # https://github.com/ryanmcgrath/twython/issues/288#issuecomment-66360160
        try:
            self.statuses.filter(locations=self.bounding_box_str)
        except TwythonRateLimitError as e:
            logger.info(f"Rate limit exceeded when streaming tweets: {e}")
            raise
        except Exception as e:
            logger.info(f"Exception when streaming tweets: {e}")
            raise

    def get_event_comparison_ts(self):
        # Find out when the last event happened, checking the database first
        most_recent_event = Events.get_most_recent_event(
            self.db_session, event_type=["moment"]
        )
        if most_recent_event is not None:
            event_comparison_ts = most_recent_event.timestamp.replace(tzinfo=pytz.UTC)
        else:
            # If no db events, use most recent tweet timestamp
            event_comparison_ts = self.twitter.get_last_post_time()

        return event_comparison_ts

    def on_success(self, status):
        # If this tweet was truncated, get the full text
        if "truncated" in status and status["truncated"]:
            status_full = self.twitter.get_user_timeline(
                user_id=status["user"]["id"],
                tweet_mode="extended",
                max_id=status["id"],
                count=1,
            )
            if status_full and (status_full[0]["id"] == status["id"]):
                logger.debug(f"Retrieved full text for truncated tweet {status['id']}")
                status = status_full[0]
            else:
                logger.debug(f"Didn't get full text for truncated tweet {status['id']}")

        use_status = check_tweet(
            status=status,
            bounding_box=BOUNDING_BOX,
            valid_place_types=VALID_PLACE_TYPES,
            ignore_words=IGNORE_WORDS,
            ignore_user_screen_names=IGNORE_USER_SCREEN_NAMES,
            ignore_user_id_str=IGNORE_USER_ID_STR,
            ignore_lon_lat=IGNORE_LON_LAT,
            ignore_possibly_sensitive=IGNORE_POSSIBLY_SENSITIVE,
            ignore_quote_status=IGNORE_QUOTE_STATUS,
            ignore_reply_status=IGNORE_REPLY_STATUS,
            min_friends_count=MIN_FRIENDS_COUNT,
            min_followers_count=MIN_FOLLOWERS_COUNT,
        )

        if not use_status:
            logger.debug(
                f"Tweet {status.get('id_str')} failed check tweet:"
                f" screen name: {status['user'].get('screen_name')}"
                f" (id: {status['user'].get('id_str')}),"
                f" following: {status['user'].get('friends_count')},"
                f" followers: {status['user'].get('followers_count')},"
                f" is quote status: {status.get('is_quote_status')},"
                " is reply status: "
                f"{status.get('in_reply_to_status_id_str') is not None},"
                f" possibly sensitive: {status.get('possibly_sensitive')},"
                f" coordinates: {status.get('coordinates')},"
                f" place type: {status['place'].get('place_type')},"
                f" place name: {status['place'].get('full_name')},"
                f" place bounding box: {get_place_bounding_box(status)},"
                f" text: {status.get('text')}"
            )
            return

        tweet_info = get_tweet_info(status)

        if LOG_TWEETS:
            _ = RecentTweets.log_tweet(self.db_session, tweet_info=tweet_info)

            # Check on whether recent tweets have been deleted, and update if so.
            # Assumption is that tweets are deleted quickly; then we don't need to spend
            # time checking whether every recent tweet has been deleted.
            self.update_deleted_tweets(
                timestamp=tweet_info.created_at, hours=TEMPORAL_GRANULARITY_HOURS
            )
        else:
            logger.debug(
                "Not logging tweet due to environment variable settings:"
                f" {tweet_info.status_id_str},"
                f" {tweet_info.place_name} ({tweet_info.place_type})"
            )

        activity_curr_day = RecentTweets.get_recent_tweets(
            self.db_session,
            timestamp=tweet_info.created_at,
            hours=24,
            place_type=VALID_PLACE_TYPES,
            has_coords=QUERY_HAS_COORDS_ONLY,
            place_type_or_coords=True,
            include_quote_status=QUERY_INCLUDE_QUOTE_STATUS,
            include_reply_status=QUERY_INCLUDE_REPLY_STATUS,
            include_deleted_status=QUERY_INCLUDE_DELETED_STATUS,
        )
        activity_prev_day = RecentTweets.get_recent_tweets(
            self.db_session,
            timestamp=tweet_info.created_at - timedelta(days=1),
            hours=24,
            place_type=VALID_PLACE_TYPES,
            has_coords=QUERY_HAS_COORDS_ONLY,
            place_type_or_coords=True,
            include_quote_status=QUERY_INCLUDE_QUOTE_STATUS,
            include_reply_status=QUERY_INCLUDE_REPLY_STATUS,
            include_deleted_status=QUERY_INCLUDE_DELETED_STATUS,
        )

        activity_curr_hour = RecentTweets.get_recent_tweets(
            self.db_session,
            timestamp=tweet_info.created_at,
            hours=TEMPORAL_GRANULARITY_HOURS,
            place_type=VALID_PLACE_TYPES,
            has_coords=QUERY_HAS_COORDS_ONLY,
            place_type_or_coords=True,
            include_quote_status=QUERY_INCLUDE_QUOTE_STATUS,
            include_reply_status=QUERY_INCLUDE_REPLY_STATUS,
            include_deleted_status=QUERY_INCLUDE_DELETED_STATUS,
        )

        activity_prev_hour = RecentTweets.get_recent_tweets(
            self.db_session,
            timestamp=tweet_info.created_at
            - timedelta(hours=TEMPORAL_GRANULARITY_HOURS),
            hours=TEMPORAL_GRANULARITY_HOURS,
            place_type=VALID_PLACE_TYPES,
            has_coords=QUERY_HAS_COORDS_ONLY,
            place_type_or_coords=True,
            include_quote_status=QUERY_INCLUDE_QUOTE_STATUS,
            include_reply_status=QUERY_INCLUDE_REPLY_STATUS,
            include_deleted_status=QUERY_INCLUDE_DELETED_STATUS,
        )

        # Decide whether an event occurred
        event_day, activity_curr_day_w = self.determine_if_event_occurred(
            activity_curr_day, activity_prev_day, ACTIVITY_THRESHOLD_DAY, "Day"
        )

        event_hour, activity_curr_hour_w = self.determine_if_event_occurred(
            activity_curr_hour, activity_prev_hour, ACTIVITY_THRESHOLD_HOUR, "Hour"
        )

        if event_day and event_hour:
            if tweet_info.created_at - self.event_comparison_ts >= timedelta(
                hours=MIN_HOURS_BETWEEN_EVENTS
            ):
                logger.info(
                    f"{tweet_info.created_at}: Been more than"
                    f" {MIN_HOURS_BETWEEN_EVENTS} hour(s) since an event occurred"
                    f" ({self.event_comparison_ts}), clustering activity..."
                )

                self.find_and_tweet_events(
                    activity_curr_hour_w,
                    min_samples=EVENT_MIN_TWEETS,
                    km_start=KM_START,
                    km_stop=KM_STOP,
                    km_step=KM_STEP,
                    min_n_clusters=MIN_N_CLUSTERS,
                    token_count_min=TOKEN_COUNT_MIN,
                    reduce_token_count_min=REDUCE_TOKEN_COUNT_MIN,
                    event_type="moment",
                )
            else:
                logger.info(
                    f"{tweet_info.created_at}: Not clustering activity to find event,"
                    f" there was an event in the last {MIN_HOURS_BETWEEN_EVENTS}"
                    f" hours ({self.event_comparison_ts})"
                )

        # Find events using today's tweets
        if POST_DAILY_EVENTS:
            try:
                tz = pytz.timezone(TIMEZONE)
            except pytz.exceptions.UnknownTimeZoneError:
                logger.info(f"Could not find timezone {TIMEZONE}, using UTC")
                tz = pytz.UTC
            current_time = datetime.now(tz)

            if current_time.hour == DAILY_EVENT_HOUR:
                if not self.posted_daily_events:
                    # Get tweets that occurred today, after midnight local time
                    activity_today_w = [
                        a
                        for a in activity_curr_day_w
                        if a["created_at"].replace(tzinfo=pytz.UTC).astimezone(tz).day
                        == current_time.day
                    ]
                    self.find_and_tweet_events(
                        activity_today_w,
                        min_samples=DAILY_EVENT_MIN_TWEETS,
                        km_start=KM_START,
                        km_stop=KM_STOP,
                        km_step=KM_STEP,
                        min_n_clusters=DAILY_MIN_N_CLUSTERS,
                        token_count_min=TOKEN_COUNT_MIN,
                        reduce_token_count_min=False,
                        event_type="daily",
                        event_str="Something happened today",
                        update_event_comparison_ts=False,
                    )
                    self.posted_daily_events = True
            else:
                # reset
                self.posted_daily_events = False

        # Purge old data every so often
        if PURGE_OLD_DATA and (
            datetime.utcnow().replace(tzinfo=pytz.UTC) - self.purge_data_comparison_ts
            >= timedelta(minutes=10)
        ):
            # Delete old data by row count
            RecentTweets.keep_tweets_n_rows(
                self.db_session, n=RECENT_TWEETS_ROWS_TO_KEEP
            )
            Events.keep_events_n_rows(self.db_session, n=EVENTS_ROWS_TO_KEEP)

            # Delete old data by timestamp
            RecentTweets.delete_tweets_older_than(
                self.db_session,
                timestamp=tweet_info.created_at,
                days=RECENT_TWEETS_DAYS_TO_KEEP,
            )
            Events.delete_events_older_than(
                self.db_session,
                timestamp=tweet_info.created_at,
                days=EVENTS_DAYS_TO_KEEP,
            )

            # Update
            self.purge_data_comparison_ts = datetime.utcnow().replace(tzinfo=pytz.UTC)

    def on_error(self, status_code, content, headers=None):
        content = (
            content.decode().strip() if isinstance(content, bytes) else content.strip()
        )
        logger.info("Error while streaming.")
        logger.info(f"status_code: {status_code}")
        logger.info(f"content: {content}")
        logger.info(f"headers: {headers}")
        if status_code == 420:
            # Server overloaded, try again in a few seconds
            # Exceeded connection limit for user
            # Too many requests recently
            raise TwythonRateLimitError("Too many requests recently")
        else:
            # Unable to decode response
            # (or something else)
            pass

    def update_deleted_tweets(self, timestamp: datetime, hours: float):
        tweets = RecentTweets.get_recent_tweets(
            self.db_session,
            timestamp=timestamp,
            hours=hours,
            place_type=VALID_PLACE_TYPES,
            has_coords=QUERY_HAS_COORDS_ONLY,
            place_type_or_coords=True,
            include_quote_status=QUERY_INCLUDE_QUOTE_STATUS,
            include_reply_status=QUERY_INCLUDE_REPLY_STATUS,
            include_deleted_status=QUERY_INCLUDE_DELETED_STATUS,
        )

        for t in tweets:
            try:
                _ = self.twitter.show_status(id=t.status_id_str)
            except TwythonError as e:
                if "No status found with that ID" in e.msg:
                    logger.info(
                        f"Tweet {t.user_screen_name}/status/{t.status_id_str}"
                        " not found, marking as deleted"
                    )
                    RecentTweets.update_tweet_deleted(self.db_session, t.status_id_str)

    def determine_if_event_occurred(
        self,
        activity_curr,
        activity_prev,
        activity_threshold: float = None,
        time_str: str = None,
    ):
        activity_threshold = activity_threshold or 1.0
        time_str = time_str or "Time window"
        event = False
        activity_curr_w = []
        if (len(activity_curr) > 1) and (len(activity_prev) > 1):
            z_diff, activity_curr_w, _ = compare_activity_kde(
                self.grid_coords,
                activity_curr,
                activity_prev,
                bw_method=BW_METHOD,
                weighted=WEIGHTED,
                weight_factor_user=WEIGHT_FACTOR_USER,
                reduce_weight_lon_lat=REDUCE_WEIGHT_LON_LAT,
                weight_factor_lon_lat=WEIGHT_FACTOR_LON_LAT,
                weight_factor_no_coords=WEIGHT_FACTOR_NO_COORDS,
            )

            lat_activity, lon_activity = np.where(z_diff > activity_threshold)

            if (lat_activity.size > 0) and (lon_activity.size > 0):
                event = True

            logger.info(
                f"{time_str} event: {event}, current: {len(activity_curr)},"
                f" previous: {len(activity_prev)},"
                f" max diff: {z_diff.max():.2f},"
                f" threshold: {activity_threshold}"
            )
        else:
            logger.info(
                f"{time_str} event: {event}, current: {len(activity_curr)},"
                f" previous: {len(activity_prev)},"
                " not enough activity,"
                f" threshold: {activity_threshold}"
            )

        return event, activity_curr_w

    def find_and_tweet_events(
        self,
        activity_w,
        min_samples: int,
        km_start: float,
        km_stop: float,
        km_step: int,
        min_n_clusters: int,
        token_count_min: int,
        reduce_token_count_min: bool,
        event_str: str = None,
        event_type: str = None,
        update_event_comparison_ts: bool = None,
    ):
        update_event_comparison_ts = update_event_comparison_ts or True

        clusters = cluster_activity(
            activity=activity_w,
            min_samples=min_samples,
            km_start=km_start,
            km_stop=km_stop,
            km_step=km_step,
            min_n_clusters=min_n_clusters,
            sample_weight=[x["weight"] for x in activity_w],
        )

        for cluster in clusters.values():
            event_info = get_event_info(
                self.twitter,
                event_tweets=cluster["event_tweets"],
                tweet_max_length=TWEET_MAX_LENGTH,
                tweet_url_length=TWEET_URL_LENGTH,
                base_event_url=BASE_EVENT_URL,
                event_str=event_str,
                event_type=event_type,
                token_count_min=token_count_min,
                reduce_token_count_min=reduce_token_count_min,
                remove_username_at=REMOVE_USERNAME_AT,
                tweet_lat_lon=TWEET_LAT_LON,
                show_tweets_on_event=SHOW_TWEETS_ON_EVENT,
            )

            if LOG_EVENTS:
                _ = Events.log_event(self.db_session, event_info=event_info)
            else:
                logger.info(
                    "Not logging event due to environment variable settings:"
                    f" {event_info.timestamp} {event_info.place_name}:"
                    f" {event_info.tokens_str}"
                )

            if POST_EVENT:
                self.twitter._update_status(
                    status=event_info.event_str,
                    lat=event_info.latitude if TWEET_GEOTAG else None,
                    long=event_info.longitude if TWEET_GEOTAG else None,
                    # place_id=event_info.place_id if TWEET_GEOTAG else None,
                )
                # Update the comparison tweet time
                if update_event_comparison_ts:
                    self.event_comparison_ts = event_info.timestamp
            else:
                logger.info("Not posting event due to environment variable settings")


def main():
    logger.info(f"Looking for tweets in bounding box: {BOUNDING_BOX}")

    logger.info(
        f"Keeping tweets with coordinates, or that have place type: {VALID_PLACE_TYPES}"
    )

    logger.info(f"Ignoring tweets containing these words: {IGNORE_WORDS}")
    logger.info(f"Ignoring tweets from these screen names: {IGNORE_USER_SCREEN_NAMES}")
    logger.info(f"Ignoring tweets from these user IDs: {IGNORE_USER_ID_STR}")
    logger.info(f"Ignoring tweets with these coordinates: {IGNORE_LON_LAT}")
    if IGNORE_POSSIBLY_SENSITIVE:
        logger.info("Ignoring possibly sensitive tweets")
    else:
        logger.info("Keeping possibly sensitive tweets")
    if IGNORE_QUOTE_STATUS:
        logger.info("Ignoring quote tweets")
    else:
        logger.info("Keeping quote tweets")
    if IGNORE_REPLY_STATUS:
        logger.info("Ignoring reply tweets")
    else:
        logger.info("Keeping reply tweets")

    # Establish connection to Twitter;
    # Uses OAuth1 ("user auth") for authentication
    twitter = MyTwitterClient(
        app_key=APP_KEY,
        app_secret=APP_SECRET,
        oauth_token=OAUTH_TOKEN,
        oauth_token_secret=OAUTH_TOKEN_SECRET,
    )

    # Establish connection to database
    db_session = session_factory(DATABASE_URL, echo=ECHO)

    logger.info("Initializing tweet streamer...")
    stream = MyStreamer(
        twitter=twitter,
        db_session=db_session,
        bounding_box=BOUNDING_BOX,
        app_key=APP_KEY,
        app_secret=APP_SECRET,
        oauth_token=OAUTH_TOKEN,
        oauth_token_secret=OAUTH_TOKEN_SECRET,
    )

    logger.info("Looking for events...")
    stream.stream_tweets()


if __name__ == "__main__":
    main()
