from collections import Counter, namedtuple
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
from time import sleep
from typing import Dict

from dotenv import load_dotenv
import numpy as np
import pytz
from runstats import Statistics
from twython import Twython, TwythonStreamer
from twython import TwythonError, TwythonRateLimitError, TwythonAuthError

from utils.data_base import session_factory, Tiles, RecentTweets, HistoricalStats, Events
from utils.tweet_utils import check_tweet, date_string_to_datetime, clean_text, stopword_lemma
from utils.data_utils import n_wise


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
    # POST_EVENT = False
    EVERY_N_SECONDS = 5
    TEMPORAL_GRANULARITY_HOURS = 1
    INITIAL_TIME = datetime(1970, 1, 1).replace(tzinfo=pytz.UTC)
    ECHO = False
else:
    logger.setLevel(logging.INFO)
    # POST_EVENT = os.getenv("POST_EVENT", default="False") == "True"
    EVERY_N_SECONDS = int(os.getenv("EVERY_N_SECONDS", default="60"))
    TEMPORAL_GRANULARITY_HOURS = int(os.getenv("TEMPORAL_GRANULARITY_HOURS", default="1"))
    # Wait the rate limit time before making first post
    INITIAL_TIME = datetime.utcnow().replace(tzinfo=pytz.UTC) - timedelta(seconds=EVERY_N_SECONDS)
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
TILE_SIZE = float(os.getenv("TILE_SIZE", default="0.01"))
EVENT_MIN_TWEETS = int(os.getenv("EVENT_MIN_TWEETS", default="4"))
TWEET_MAX_LENGTH = int(os.getenv("TWEET_MAX_LENGTH", default="280"))
HISTORICAL_STATS_DAYS_TO_KEEP = float(os.getenv("HISTORICAL_STATS_DAYS_TO_KEEP", default="1.0"))
RECENT_TWEETS_DAYS_TO_KEEP = float(os.getenv("RECENT_TWEETS_DAYS_TO_KEEP", default="4.0"))
MAX_ROWS_HISTORICAL_STATS = int(os.getenv("MAX_ROWS_HISTORICAL_STATS", default="6000"))

VALID_PLACE_TYPES = os.getenv("VALID_PLACE_TYPES", default="neighborhood, poi")
VALID_PLACE_TYPES = [x.strip() for x in VALID_PLACE_TYPES.split(',')] if VALID_PLACE_TYPES else []
IGNORE_WORDS = os.getenv("IGNORE_WORDS", default=None)
IGNORE_WORDS = [x.strip() for x in IGNORE_WORDS.split(',')] if IGNORE_WORDS else []
IGNORE_USER_SCREEN_NAMES = os.getenv("IGNORE_USER_SCREEN_NAMES", default=None)
IGNORE_USER_SCREEN_NAMES = [x.strip() for x in IGNORE_USER_SCREEN_NAMES.split(',')] if IGNORE_USER_SCREEN_NAMES else []
IGNORE_USER_ID_STR = os.getenv("IGNORE_USER_ID_STR", default=None)
IGNORE_USER_ID_STR = [x.strip() for x in IGNORE_USER_ID_STR.split(',')] if IGNORE_USER_ID_STR else []

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


class MyTwitterClient(Twython):
    '''Wrapper around the Twython Twitter client.
    Limits status update rate.
    '''
    def __init__(self, initial_time=None, *args, **kwargs):
        super(MyTwitterClient, self).__init__(*args, **kwargs)
        if initial_time is None:
            # Wait half the rate limit time before making first post
            initial_time = datetime.utcnow().replace(tzinfo=pytz.UTC) - timedelta(hours=1)
        self.last_post_time = initial_time

    def update_status_check_rate(self, *args, **kwargs):
        current_time = datetime.utcnow().replace(tzinfo=pytz.UTC)
        logger.info(f'Current time: {current_time}')
        logger.info(f'Previous post time: {self.last_post_time}')
        logger.info(f'Difference: {current_time - self.last_post_time}')
        status = {}
        if (current_time - self.last_post_time).total_seconds() > EVERY_N_SECONDS:
            try:
                status = self.update_status(*args, **kwargs)
            except TwythonAuthError:
                logger.exception('Authorization error, did you create read+write credentials?')
            except TwythonRateLimitError:
                logger.exception('Rate limit error')
            except TwythonError:
                logger.exception('Encountered some other error')

            if status:
                self.last_post_time = current_time
                logger.info('Successfully posted')
            else:
                logger.info('Did not successfully post')
        else:
            logger.info('Not posting due to rate limit')

        return status


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
            tiles = Tiles.find_id_by_coords(session, tweet_info.longitude, tweet_info.latitude)

            if tiles:
                tile = tiles[0]
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
                    tile_id=tile.id,
                )
                session.add(tweet)
                try:
                    session.commit()
                    logger.info(f'Logged tweet {tweet_info.status_id_str} in {tweet_info.place_name} (tile {tile.id})')
                except Exception:
                    logger.exception(f'Exception when adding recent tweet {tweet_info.status_id_str}')
                    session.rollback()

                if tweet_info.created_at - self.comparison_timestamp >= timedelta(hours=1):
                    logger.info(f'current time: {tweet_info.created_at}')
                    logger.info('Been more than 1 hour between oldest and current tweet')

                    # Get recent stats
                    tweet_counts, hs_hour, hs_day = get_stats(tweet_info.created_at)

                    # Initialize to state whether an event occurred
                    events = {i: False for i in range(1, num_tiles + 1)}

                    for tile_id, stats in self.running_stats.items():
                        # Update each tile's running stats with the current count
                        if tile_id in tweet_counts:
                            tweet_count = tweet_counts[tile_id]
                        else:
                            tweet_count = 0
                        stats.push(tweet_count)
                        mean = stats.mean()
                        try:
                            variance = stats.variance()
                        except ZeroDivisionError:
                            variance = 0
                        try:
                            stddev = stats.stddev()
                        except ZeroDivisionError:
                            stddev = 0

                        # Add current stats to historical stats table
                        hs = HistoricalStats(
                            tile_id=tile_id,
                            timestamp=tweet_info.created_at,
                            count=tweet_count,
                            mean=mean,
                            variance=variance,
                            stddev=stddev,
                        )
                        session.add(hs)

                        try:
                            session.commit()
                            logger.debug(f'Added historical stats: tile {tile_id} {tweet_info.created_at}')
                        except Exception:
                            logger.exception(f'Exception when adding historical stats: tile {tile_id} {tweet_info.created_at}')
                            session.rollback()

                        event_hour = False
                        event_day = False
                        if tweet_count > 0:
                            logger.info(f'tile_id: {tile_id}, tweet_count: {tweet_count}')
                            if hs_hour:
                                threshold_hour = hs_hour[tile_id].mean + (hs_hour[tile_id].stddev * 2)
                                event_hour = (tweet_count >= EVENT_MIN_TWEETS) and (tweet_count > threshold_hour)
                                # logger.info(f'Now vs hour: {event_hour}')
                                # logger.info(f'    hour time: {hs_hour[tile_id].timestamp}, count: {hs_hour[tile_id].count}')
                                logger.info(f'    hour threshold: {threshold_hour} = {hs_hour[tile_id].mean} + ({hs_hour[tile_id].stddev} * 2)')
                            if hs_day:
                                threshold_day = hs_day[tile_id].mean + (hs_day[tile_id].stddev * 2)
                                event_day = (tweet_count >= EVENT_MIN_TWEETS) and (tweet_count > threshold_day)
                                # logger.info(f'Now vs day: {event_day}')
                                # logger.info(f'    day time: {hs_day[tile_id].timestamp}, count: {hs_day[tile_id].count}')
                                logger.info(f'    day threshold: {threshold_day} = {hs_day[tile_id].mean} + ({hs_day[tile_id].stddev} * 2)')

                        # Note that this tile had an event
                        if event_hour and event_day:
                            events[tile_id] = True

                    if any(events.values()):
                        tiles_with_events = [tile_id for tile_id, event in events.items() if event]
                        for tile_id in tiles_with_events:
                            event_str = event_log_and_string(tile_id=tile_id, timestamp=tweet_info.created_at)
                            try:
                                status = twitter.update_status(status=event_str[:TWEET_MAX_LENGTH])
                            except TwythonAuthError:
                                logger.exception('Authorization error, did you create read+write credentials?')
                            except TwythonRateLimitError:
                                logger.exception('Rate limit error')
                            except TwythonError:
                                logger.exception('Encountered some other error')

                    # Update the comparison tweet time
                    self.comparison_timestamp = tweet_info.created_at

                    # Delete old historical stats rows
                    logger.info('Deleting old historical stats')
                    HistoricalStats.delete_stats_older_than(session, timestamp=tweet_info.created_at, days=HISTORICAL_STATS_DAYS_TO_KEEP)

                    # Delete old recent tweets rows
                    logger.info('Deleting old recent tweets')
                    RecentTweets.delete_tweets_older_than(session, timestamp=tweet_info.created_at, days=RECENT_TWEETS_DAYS_TO_KEEP)
            else:
                logger.info(f'Tweet {tweet_info.status_id_str} coordinates ({tweet_info.latitude}, {tweet_info.longitude}, {tweet_info.place_name}, {tweet_info.place_type}) matched incorrect number of tiles: {len(tiles)}')

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
    place_name = status['place']['full_name']
    # Possible values: country, admin, city, neighborhood, poi; more?
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


def get_stats(timestamp: datetime):
    # Get tweets from the last hour
    tweet_counts = RecentTweets.count_tweets_per_tile(session, timestamp=timestamp, hours=1)
    tweet_counts = {tile_id: count for tile_id, count in tweet_counts}

    # Get historical stats for the previous hour
    hs_hour = HistoricalStats.get_recent_stats(session, timestamp=timestamp, hours=1)
    hs_hour = {tile_id: row for tile_id, row in hs_hour}

    # Get historical stats for the previous day
    hs_day = HistoricalStats.get_recent_stats(session, timestamp=timestamp, days=1)
    hs_day = {tile_id: row for tile_id, row in hs_day}

    return tweet_counts, hs_hour, hs_day


def event_log_and_string(tile_id: int, timestamp: datetime):
    event_tweets = RecentTweets.get_recent_tweets(session, timestamp=timestamp, hours=1, tile_id=tile_id)
    tile_event_tweets = [et for et in event_tweets if et.tile_id == tile_id]
    tile_event_tokens = [stopword_lemma(clean_text(et.tweet_body)) for et in tile_event_tweets]
    tokens = [token.lower() for tweet in tile_event_tokens for token in tweet.split()]
    counter = Counter(tokens)
    tokens_to_show = [(k, v) for k, v in counter.items() if v > 1]
    tokens_str = ' '.join([t[0] for t in tokens_to_show])

    lons = [et.longitude for et in tile_event_tweets]
    longitude = sum(lons) / len(lons)
    lats = [et.latitude for et in tile_event_tweets]
    latitude = sum(lats) / len(lats)
    lat_long_str = f'{latitude:.4f}, {longitude:.4f}'

    place_names = [
        et.place_name for et in tile_event_tweets if et.place_type in ['city', 'neighborhood', 'poi']
    ]
    place_name = Counter(place_names).most_common(1)[0][0] if place_names else None

    event_str = "Something's happening"
    event_str = f'{event_str} in {place_name}' if place_name else event_str
    event_str = f'{event_str} ({lat_long_str}): {tokens_str}'
    logger.info(f'{timestamp} Tile {tile_id}: Found event with {len(tile_event_tweets)} tweets')
    logger.info(event_str)

    # Add to events table
    ev = Events(
        tile_id=tile_id,
        timestamp=timestamp,
        count=len(tile_event_tweets),
        longitude=longitude,
        latitude=latitude,
        place_name=place_name,
        description=tokens_str,
    )
    session.add(ev)

    try:
        session.commit()
        logger.info(f'Added event for tile {tile_id} {timestamp} {tokens_str}')
    except Exception:
        logger.exception(f'Exception when adding event for tile {tile_id} {timestamp} {tokens_str}')
        session.rollback()

    return event_str


# Establish connection to Twitter
# Uses OAuth1 ("user auth") for authentication
twitter = MyTwitterClient(
    initial_time=INITIAL_TIME,
    app_key=APP_KEY,
    app_secret=APP_SECRET,
    oauth_token=OAUTH_TOKEN,
    oauth_token_secret=OAUTH_TOKEN_SECRET,
)

# If this screen_name has a recent tweet, use that timestamp as the time of the last post
my_most_recent_tweet = twitter.get_user_timeline(screen_name=MY_SCREEN_NAME, count=1, trim_user=True)
if my_most_recent_tweet:
    twitter.last_post_time = date_string_to_datetime(my_most_recent_tweet[0]['created_at'])

# Establish connection to database
session = session_factory(DATABASE_URL, echo=ECHO)

# Add tile rows if none exist
if session.query(Tiles).count() == 0:
    logger.info('Populating tiles table')
    tile_longitudes = np.arange(BOUNDING_BOX[0], BOUNDING_BOX[2], TILE_SIZE).tolist()
    # np.arange does not include the end point; add the edge of bounding box
    tile_longitudes.append(BOUNDING_BOX[2])
    tile_latitudes = np.arange(BOUNDING_BOX[1], BOUNDING_BOX[3], TILE_SIZE).tolist()
    # np.arange does not include the end point; add the edge of bounding box
    tile_latitudes.append(BOUNDING_BOX[3])

    num_tiles = len(list(n_wise(tile_latitudes, 2))) * len(list(n_wise(tile_longitudes, 2)))
    assert (num_tiles * HISTORICAL_STATS_DAYS_TO_KEEP * 24) <= MAX_ROWS_HISTORICAL_STATS

    for tile_lats in n_wise(tile_latitudes, 2):
        south_lat, north_lat = tile_lats[0], tile_lats[1]
        for tile_lons in n_wise(tile_longitudes, 2):
            west_lon, east_lon = tile_lons[0], tile_lons[1]
            tile = Tiles(
                west_lon=west_lon,
                east_lon=east_lon,
                south_lat=south_lat,
                north_lat=north_lat,
            )
            session.add(tile)
    try:
        session.commit()
    except Exception:
        logger.exception('Exception when populating tiles table')
        session.rollback()
else:
    logger.info('tiles table is already populated')

num_tiles = Tiles.get_num_tiles(session)

# Initialize running stats object for each tile
stats_all = session.query(HistoricalStats.tile_id, HistoricalStats).order_by(HistoricalStats.tile_id, HistoricalStats.timestamp).all()
if stats_all:
    logger.info('Retrieved existing running stats counts per tile')
    # Populate running stats objects for each tile from table using the counts
    stats_counts = {i: [] for i in range(1, num_tiles + 1)}
    for i, stats in stats_all:
        stats_counts[i].append(stats.count)
    running_stats = {i: Statistics(stats_counts[i]) for i in range(1, num_tiles + 1)}
else:
    logger.info('Initializing new running stats objects')
    # Initialize new running stats objects
    running_stats = {i: Statistics() for i in range(1, num_tiles + 1)}

# Decide how long to collect new tweets for before assessing event
comparison = HistoricalStats.get_recent_stats(session)
if comparison:
    comparison_timestamp = comparison[0][1].timestamp.replace(tzinfo=pytz.UTC)
    # If there are no stats in the recent past, wait the full time
    if comparison_timestamp + timedelta(minutes=30) < datetime.utcnow().replace(tzinfo=pytz.UTC):
        comparison_timestamp = None
else:
    comparison_timestamp = None

if __name__ == '__main__':
    logger.info('Initializing tweet streamer...')
    stream = MyStreamer(
        running_stats=running_stats,
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
