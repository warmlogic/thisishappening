import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from time import sleep

import numpy as np
import pytz
from dotenv import load_dotenv
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
logger = logging.getLogger("happeninglogger")

IS_PROD = os.getenv("IS_PROD", default=None)

if IS_PROD is None:
    env_path = Path.cwd().parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        env_path = Path.cwd() / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
        else:
            raise OSError(".env file not found. Did you set it up?")

DEBUG_RUN = os.getenv("DEBUG_RUN", default="False").casefold()
if DEBUG_RUN not in ["true".casefold(), "false".casefold()]:
    raise ValueError(f"DEBUG_RUN must be True or False, current value: {DEBUG_RUN}")
DEBUG_RUN = DEBUG_RUN == "true".casefold()

if DEBUG_RUN:
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
    POST_EVENT = (
        os.getenv("POST_EVENT", default="False").casefold() == "true".casefold()
    )
    POST_DAILY_EVENTS = (
        os.getenv("POST_DAILY_EVENTS", default="False").casefold() == "true".casefold()
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
MY_SCREEN_NAME = os.getenv("MY_SCREEN_NAME", default=None)
assert MY_SCREEN_NAME is not None
LANGUAGE = os.getenv("LANGUAGE", default="en")
TIMEZONE = os.getenv("TIMEZONE", default="UTC")
BOUNDING_BOX = os.getenv("BOUNDING_BOX", default=None)
BOUNDING_BOX = (
    [float(coord) for coord in BOUNDING_BOX.split(",")] if BOUNDING_BOX else []
)
assert len(BOUNDING_BOX) == 4
TEMPORAL_GRANULARITY_HOURS = float(os.getenv("TEMPORAL_GRANULARITY_HOURS", default="1"))
MIN_HOURS_BETWEEN_EVENTS = float(os.getenv("MIN_HOURS_BETWEEN_EVENTS", default="1"))
EVENT_MIN_TWEETS = int(os.getenv("EVENT_MIN_TWEETS", default="5"))
DAILY_EVENT_MIN_TWEETS = int(os.getenv("DAILY_EVENT_MIN_TWEETS", default="8"))
DAILY_EVENT_HOUR = int(os.getenv("DAILY_EVENT_HOUR", default="23"))
KM_START = float(os.getenv("KM_START", default="0.05"))
KM_STOP = float(os.getenv("KM_STOP", default="0.3"))
KM_STEP = int(os.getenv("KM_STEP", default="9"))
MIN_N_CLUSTERS = int(os.getenv("MIN_N_CLUSTERS", default="1"))
TWEET_MAX_LENGTH = int(os.getenv("TWEET_MAX_LENGTH", default="280"))
TWEET_URL_LENGTH = int(os.getenv("TWEET_URL_LENGTH", default="23"))
TWEET_LAT_LON = (
    os.getenv("TWEET_LAT_LON", default="False").casefold() == "true".casefold()
)
SHOW_TWEETS_ON_EVENT = (
    os.getenv("SHOW_TWEETS_ON_EVENT", default="True").casefold() == "true".casefold()
)
TWEET_GEOTAG = os.getenv("TWEET_GEOTAG", default="True").casefold() == "true".casefold()
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
    [rf"\b{x.strip()}\b" for x in IGNORE_WORDS.split(",")] if IGNORE_WORDS else []
)
IGNORE_WORDS = list(set(IGNORE_WORDS))
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
    os.getenv("IGNORE_POSSIBLY_SENSITIVE", default="False").casefold()
    == "true".casefold()
)
IGNORE_QUOTE_STATUS = (
    os.getenv("IGNORE_QUOTE_STATUS", default="False").casefold() == "true".casefold()
)
IGNORE_REPLY_STATUS = (
    os.getenv("IGNORE_REPLY_STATUS", default="False").casefold() == "true".casefold()
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
    os.getenv("REDUCE_TOKEN_COUNT_MIN", default="True").casefold() == "true".casefold()
)
REMOVE_USERNAME_AT = (
    os.getenv("REMOVE_USERNAME_AT", default="True").casefold() == "true".casefold()
)

GRID_RESOLUTION = int(os.getenv("GRID_RESOLUTION", default="128"))
BW_METHOD = float(os.getenv("BW_METHOD", default="0.3"))
ACTIVITY_THRESHOLD_DAY = float(os.getenv("ACTIVITY_THRESHOLD_DAY", default="30.0"))
ACTIVITY_THRESHOLD_HOUR = float(os.getenv("ACTIVITY_THRESHOLD_HOUR", default="300.0"))

WEIGHTED = os.getenv("WEIGHTED", default="False").casefold() == "true".casefold()
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
    os.getenv("QUERY_HAS_COORDS_ONLY", default="False").casefold() == "true".casefold()
)
QUERY_HAS_COORDS_ONLY = QUERY_HAS_COORDS_ONLY if QUERY_HAS_COORDS_ONLY else None

QUERY_INCLUDE_QUOTE_STATUS = (
    os.getenv("QUERY_INCLUDE_QUOTE_STATUS", default="True").casefold()
    == "true".casefold()
)
QUERY_INCLUDE_REPLY_STATUS = (
    os.getenv("QUERY_INCLUDE_REPLY_STATUS", default="False").casefold()
    == "true".casefold()
)
QUERY_INCLUDE_DELETED_STATUS = (
    os.getenv("QUERY_INCLUDE_DELETED_STATUS", default="False").casefold()
    == "true".casefold()
)


class MyStreamer(TwythonStreamer):
    def __init__(self, grid_coords, event_comparison_ts=None, *args, **kwargs):
        super(MyStreamer, self).__init__(*args, **kwargs)
        self.grid_coords = grid_coords
        if event_comparison_ts is None:
            event_comparison_ts = datetime.utcnow().replace(
                tzinfo=pytz.UTC
            ) - timedelta(hours=MIN_HOURS_BETWEEN_EVENTS)
        self.event_comparison_ts = event_comparison_ts
        self.purge_data_comparison_ts = datetime.utcnow().replace(tzinfo=pytz.UTC)
        self.posted_daily_events = False
        self.sleep_seconds = 2
        self.sleep_exponent = 0

    def on_success(self, status):
        # Reset sleep seconds exponent
        self.sleep_exponent = 0

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
                + f" screen name: {status['user'].get('screen_name')}"
                + f" (id: {status['user'].get('id_str')}),"
                + f" following: {status['user'].get('friends_count')},"
                + f" followers: {status['user'].get('followers_count')},"
                + f" is quote status: {status.get('is_quote_status')},"
                + " is reply status: "
                + f"{status.get('in_reply_to_status_id_str') is not None},"
                + f" possibly sensitive: {status.get('possibly_sensitive')},"
                + f" coordinates: {status.get('coordinates')},"
                + f" place type: {status['place'].get('place_type')},"
                + f" place name: {status['place'].get('full_name')},"
                + f" place bounding box: {get_place_bounding_box(status)},"
                + f" text: {status.get('text')}"
            )
            return

        tweet_info = get_tweet_info(status)

        if LOG_TWEETS:
            _ = RecentTweets.log_tweet(session, tweet_info=tweet_info)
        else:
            logger.debug(
                "Not logging tweet due to environment variable settings:"
                + f" {tweet_info.status_id_str},"
                + f" {tweet_info.place_name} ({tweet_info.place_type})"
            )

        if tweet_info.created_at - self.event_comparison_ts < timedelta(
            hours=MIN_HOURS_BETWEEN_EVENTS
        ):
            logger.info(
                "Not looking for new event, recent event in the last"
                + f" {MIN_HOURS_BETWEEN_EVENTS} hours"
                + f" ({self.event_comparison_ts})"
            )
            return

        logger.info(
            f"{tweet_info.created_at} Been more than"
            + f" {MIN_HOURS_BETWEEN_EVENTS} hour(s)"
            + " since an event occurred, comparing activity..."
        )

        # Check on whether recent tweets have been deleted, and update if so.
        # Assumption is that tweets are deleted quickly; then we don't need to spend
        # time checking whether every recent tweet has been deleted.
        self.update_deleted_tweets(
            timestamp=tweet_info.created_at, hours=TEMPORAL_GRANULARITY_HOURS
        )

        activity_curr_day = RecentTweets.get_recent_tweets(
            session,
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
            session,
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
            session,
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
            session,
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
            activity_prev_day, activity_curr_day, ACTIVITY_THRESHOLD_DAY, "Day"
        )

        event_hour, activity_curr_hour_w = self.determine_if_event_occurred(
            activity_prev_hour, activity_curr_hour, ACTIVITY_THRESHOLD_HOUR, "Hour"
        )

        if event_day and event_hour:
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

        # Only post once per day, at 11pm local time
        if POST_DAILY_EVENTS:
            try:
                tz = pytz.timezone(TIMEZONE)
            except pytz.exceptions.UnknownTimeZoneError:
                tz = "UTC"
                logger.info(f"Could not find timezone {TIMEZONE}, defaulting to {tz}")
            current_time = datetime.now(tz)

            if not self.posted_daily_events and current_time.hour == DAILY_EVENT_HOUR:
                self.find_and_tweet_events(
                    activity_curr_day_w,
                    min_samples=DAILY_EVENT_MIN_TWEETS,
                    km_start=KM_START,
                    km_stop=KM_STOP,
                    km_step=KM_STEP,
                    min_n_clusters=MIN_N_CLUSTERS,
                    token_count_min=TOKEN_COUNT_MIN,
                    reduce_token_count_min=False,
                    event_type="daily",
                    event_str="Something happened today",
                    update_event_comparison_ts=False,
                )
                self.posted_daily_events = True
            else:
                self.posted_daily_events = False

        # Purge old data every so often
        if PURGE_OLD_DATA and (
            datetime.utcnow().replace(tzinfo=pytz.UTC) - self.purge_data_comparison_ts
            >= timedelta(minutes=10)
        ):
            # Delete old data by row count
            RecentTweets.keep_tweets_n_rows(session, n=RECENT_TWEETS_ROWS_TO_KEEP)
            Events.keep_events_n_rows(session, n=EVENTS_ROWS_TO_KEEP)

            # Delete old data by timestamp
            RecentTweets.delete_tweets_older_than(
                session,
                timestamp=tweet_info.created_at,
                days=RECENT_TWEETS_DAYS_TO_KEEP,
            )
            Events.delete_events_older_than(
                session,
                timestamp=tweet_info.created_at,
                days=EVENTS_DAYS_TO_KEEP,
            )

            # Update
            self.purge_data_comparison_ts = datetime.utcnow().replace(tzinfo=pytz.UTC)

    def on_error(self, status_code, content, headers=None):
        logger.info("Error while streaming.")
        logger.info(f"status_code: {status_code}")
        logger.info(f"content: {content}")
        logger.info(f"headers: {headers}")
        content = (
            content.decode().strip() if isinstance(content, bytes) else content.strip()
        )
        if "Server overloaded, try again in a few seconds".lower() in content.lower():
            seconds = self.sleep_seconds**self.sleep_exponent
            logger.warning(f"Server overloaded. Sleeping for {seconds} seconds.")
            sleep(seconds)
            self.sleep_exponent += 1
        elif "Exceeded connection limit for user".lower() in content.lower():
            seconds = self.sleep_seconds**self.sleep_exponent
            logger.warning(
                f"Exceeded connection limit. Sleeping for {seconds} seconds."
            )
            sleep(seconds)
            self.sleep_exponent += 1
        else:
            seconds = self.sleep_seconds**self.sleep_exponent
            logger.warning(
                f"Some other error occurred. Sleeping for {seconds} seconds."
            )
            sleep(seconds)
            self.sleep_exponent += 1

    def update_deleted_tweets(self, timestamp: datetime, hours: float):
        tweets = RecentTweets.get_recent_tweets(
            session,
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
                _ = twitter.show_status(id=t.status_id_str)
            except TwythonError as e:
                if "No status found with that ID" in e.msg:
                    logger.info(
                        f"Tweet {t.user_screen_name}/status/{t.status_id_str}"
                        + " not found, marking as deleted"
                    )
                    RecentTweets.update_tweet_deleted(session, t.status_id_str)

    def determine_if_event_occurred(
        self, activity_prev, activity_curr, activity_threshold, time_str
    ):
        event = False
        if (len(activity_prev) > 1) and (len(activity_curr) > 1):
            z_diff, _, activity_curr_w = compare_activity_kde(
                self.grid_coords,
                activity_prev,
                activity_curr,
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
                + f" previous: {len(activity_prev)},"
                + f" max diff: {z_diff.max():.2f},"
                + f" threshold: {activity_threshold}"
            )
        else:
            logger.info(
                f"{time_str} event: {event}, current: {len(activity_curr)},"
                + f" previous: {len(activity_prev)},"
                + " not enough activity,"
                + f" threshold: {activity_threshold}"
            )

        return event, activity_curr_w

    def find_and_tweet_events(
        self,
        activity_curr_w,
        min_samples: int,
        km_start: float,
        km_stop: float,
        km_step: int,
        min_n_clusters: int,
        token_count_min: int,
        reduce_token_count_min: bool,
        event_str: str = None,
        event_type: str = None,
        update_event_comparison_ts: bool = True,
    ):
        clusters = cluster_activity(
            activity=activity_curr_w,
            min_samples=min_samples,
            km_start=km_start,
            km_stop=km_stop,
            km_step=km_step,
            min_n_clusters=min_n_clusters,
            sample_weight=[x["weight"] for x in activity_curr_w],
        )

        for cluster in clusters.values():
            event_info = get_event_info(
                twitter,
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
                _ = Events.log_event(session, event_info=event_info)
            else:
                logger.info(
                    "Not logging event due to environment variable settings:"
                    + f" {event_info.timestamp} {event_info.place_name}:"
                    + f" {event_info.tokens_str}"
                )

            if POST_EVENT:
                try:
                    _ = twitter.update_status(
                        status=event_info.event_str,
                        lat=event_info.latitude if TWEET_GEOTAG else None,
                        long=event_info.longitude if TWEET_GEOTAG else None,
                        # place_id=(event_info.place_id if TWEET_GEOTAG else None),
                    )

                    # Update the comparison tweet time
                    if update_event_comparison_ts:
                        self.event_comparison_ts = event_info.timestamp
                except TwythonAuthError:
                    logger.exception(
                        "Authorization error, did you create read+write credentials?"
                    )
                except TwythonRateLimitError:
                    logger.exception("Rate limit error")
                except TwythonError:
                    logger.exception("Encountered some other error")
            else:
                logger.info("Not posting event due to environment variable settings")


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
    most_recent_tweet = twitter.get_user_timeline(
        screen_name=MY_SCREEN_NAME, count=1, trim_user=True
    )
    if len(most_recent_tweet) > 0:
        event_comparison_ts = date_string_to_datetime(
            most_recent_tweet[0]["created_at"]
        )
    else:
        event_comparison_ts = None

if __name__ == "__main__":
    logger.info("Initializing tweet streamer...")
    grid_coords, _, _ = get_grid_coords(
        bounding_box=BOUNDING_BOX, grid_resolution=GRID_RESOLUTION
    )
    stream = MyStreamer(
        grid_coords=grid_coords,
        event_comparison_ts=event_comparison_ts,
        app_key=APP_KEY,
        app_secret=APP_SECRET,
        oauth_token=OAUTH_TOKEN,
        oauth_token_secret=OAUTH_TOKEN_SECRET,
    )

    bounding_box_str = ",".join([str(x) for x in BOUNDING_BOX])
    logger.info(f"Looking for tweets in bounding box: {bounding_box_str}")

    logger.info(
        "Keeping tweets with coordinates,"
        + f" or that have place type: {VALID_PLACE_TYPES}"
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

    while True:
        # Use try/except to avoid ChunkedEncodingError
        # https://github.com/ryanmcgrath/twython/issues/288
        try:
            stream.statuses.filter(locations=bounding_box_str)
        except Exception as e:
            logger.info(f"Exception when streaming tweets: {e}")
            continue
