from collections import Counter, namedtuple
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
from time import sleep
from typing import Dict, List

from dotenv import load_dotenv
import numpy as np
import pytz
from runstats import Statistics
from twython import Twython, TwythonStreamer
from twython import TwythonError, TwythonRateLimitError, TwythonAuthError

from utils.data_base import session_factory, Tiles, RecentTweets, HistoricalStats, Events
from utils.tweet_utils import check_tweet, date_string_to_datetime, get_tokens_to_tweet
from utils.data_utils import n_wise
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
TILE_SIZE = float(os.getenv("TILE_SIZE", default="0.01"))
EVENT_MIN_TWEETS = int(os.getenv("EVENT_MIN_TWEETS", default="9"))
TWEET_MAX_LENGTH = int(os.getenv("TWEET_MAX_LENGTH", default="280"))
TWEET_URL_LENGTH = int(os.getenv("TWEET_URL_LENGTH", default="23"))
HISTORICAL_STATS_DAYS_TO_KEEP = float(os.getenv("HISTORICAL_STATS_DAYS_TO_KEEP", default="1.0"))
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
IGNORE_MISSING_DAY_STATS = os.getenv("IGNORE_MISSING_DAY_STATS", default="False") == "True"

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
            tweet_info = self.get_tweet_info(status)
            tiles = Tiles.find_id_by_coords(session, tweet_info.longitude, tweet_info.latitude)

            if tiles:
                # Assume the first tile is correct (there should only be one)
                tile = tiles[0]

                self.log_tweet(tweet_info=tweet_info, tile_id=tile.id)

                if tweet_info.created_at - self.comparison_timestamp >= timedelta(hours=TEMPORAL_GRANULARITY_HOURS):
                    logger.info(f'current time: {tweet_info.created_at}')
                    logger.info(f'Been more than {TEMPORAL_GRANULARITY_HOURS} hours between oldest and current tweet')

                    # Get recent stats
                    tweet_counts, hs_hour, hs_day = self.get_stats(tweet_info.created_at)

                    # Initialize to state whether an event occurred
                    events = {i: False for i in range(1, num_tiles + 1)}

                    for tile_id, stats in self.running_stats.items():
                        # Update each tile's running stats with the current count
                        tweet_count = self.log_stats(tweet_counts, tile_id, stats, tweet_info.created_at)

                        # Decide whether an event occurred
                        events[tile_id] = self.compare_activity(tweet_count, tile_id, hs_hour, hs_day)

                    if any(events.values()):
                        last_hour_tweets = RecentTweets.get_recent_tweets(session, timestamp=tweet_info.created_at, hours=TEMPORAL_GRANULARITY_HOURS)

                        clusters = cluster_activity(session, activity=last_hour_tweets, min_samples=EVENT_MIN_TWEETS)

                        for cid, cluster_dict in clusters.items():
                            event_str = self.log_event_and_get_str(
                                event_tweets=cluster_dict['event_tweets'],
                                tile_id=cluster_dict['tile_id'],
                                timestamp=tweet_info.created_at,
                                token_count_min=TOKEN_COUNT_MIN,
                            )

                            if POST_EVENT:
                                try:
                                    status = twitter.update_status(status=event_str)
                                except TwythonAuthError:
                                    logger.exception('Authorization error, did you create read+write credentials?')
                                except TwythonRateLimitError:
                                    logger.exception('Rate limit error')
                                except TwythonError:
                                    logger.exception('Encountered some other error')
                            else:
                                logger.info('Not posting event due to environment variable settings')

                        # tiles_with_events = [tile_id for tile_id, event in events.items() if event]
                        # for tile_id in tiles_with_events:
                        #     event_tweets = RecentTweets.get_recent_tweets(session, timestamp=tweet_info.created_at, hours=TEMPORAL_GRANULARITY_HOURS, tile_id=tile_id)

                        #     event_str = self.log_event_and_get_str(
                        #         event_tweets=event_tweets,
                        #         tile_id=tile_id,
                        #         timestamp=tweet_info.created_at,
                        #         token_count_min=TOKEN_COUNT_MIN,
                        #     )

                        #     if POST_EVENT:
                        #         try:
                        #             status = twitter.update_status(status=event_str)
                        #         except TwythonAuthError:
                        #             logger.exception('Authorization error, did you create read+write credentials?')
                        #         except TwythonRateLimitError:
                        #             logger.exception('Rate limit error')
                        #         except TwythonError:
                        #             logger.exception('Encountered some other error')
                        #     else:
                        #         logger.info('Not posting event due to environment variable settings')

                    # Update the comparison tweet time
                    self.comparison_timestamp = tweet_info.created_at

                    # Delete old historical stats rows
                    logger.info('Deleting old historical stats')
                    HistoricalStats.delete_stats_older_than(session, timestamp=tweet_info.created_at, days=HISTORICAL_STATS_DAYS_TO_KEEP)

                    # Delete old recent tweets rows
                    logger.info('Deleting old recent tweets')
                    RecentTweets.delete_tweets_older_than(session, timestamp=tweet_info.created_at, days=RECENT_TWEETS_DAYS_TO_KEEP)

                    # Delete old events rows
                    logger.info('Deleting old events')
                    Events.delete_events_older_than(session, timestamp=tweet_info.created_at, days=EVENTS_DAYS_TO_KEEP)
            else:
                logger.info(f'Tweet {tweet_info.status_id_str} coordinates ({tweet_info.latitude}, {tweet_info.longitude}, {tweet_info.place_name}, {tweet_info.place_type}) matched incorrect number of tiles: {len(tiles)}')

    def get_tweet_info(self, status: Dict) -> Dict:
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

    def log_tweet(self, tweet_info: TweetInfo, tile_id: int):
        tile_name = Tiles.get_tile_name(session, tile_id=tile_id)[0][1]
        if LOG_TWEETS:
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
                logger.info(f'Logged tweet: {tweet_info.status_id_str} in Tile {tile_id} ({tile_name}), {tweet_info.place_name} ({tweet_info.place_type})')
            except Exception:
                logger.exception(f'Exception when logging tweet: {tweet_info.status_id_str} in Tile {tile_id} ({tile_name}), {tweet_info.place_name} ({tweet_info.place_type})')
                session.rollback()
        else:
            logger.info(f'Not logging tweet due to environment variable settings: {tweet_info.status_id_str} in Tile {tile_id} ({tile_name}), {tweet_info.place_name} ({tweet_info.place_type})')

    def get_stats(self, timestamp: datetime):
        # Get tweets from the most recent period
        tweet_counts = RecentTweets.count_tweets_per_tile(session, timestamp=timestamp, hours=TEMPORAL_GRANULARITY_HOURS)
        tweet_counts = {tile_id: count for tile_id, count in tweet_counts}

        # Get historical stats for the previous period
        hs_hour = HistoricalStats.get_recent_stats(session, timestamp=timestamp, hours=TEMPORAL_GRANULARITY_HOURS)
        hs_hour = {tile_id: row for tile_id, row in hs_hour}

        # Get historical stats for the same period on the previous day
        hs_day = HistoricalStats.get_recent_stats(session, timestamp=timestamp, days=1)
        hs_day = {tile_id: row for tile_id, row in hs_day}

        return tweet_counts, hs_hour, hs_day

    def log_stats(self, tweet_counts: Dict[int, int], tile_id: int, stats: Statistics, timestamp: datetime):
        tile_name = Tiles.get_tile_name(session, tile_id=tile_id)[0][1]
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
        if LOG_STATS:
            hs = HistoricalStats(
                tile_id=tile_id,
                timestamp=timestamp,
                count=tweet_count,
                mean=mean,
                variance=variance,
                stddev=stddev,
            )
            session.add(hs)

            try:
                session.commit()
                logger.debug(f'Logged historical stats: Tile {tile_id} ({tile_name}) {timestamp}')
            except Exception:
                logger.exception(f'Exception when logging historical stats: Tile {tile_id} ({tile_name}) {timestamp}')
                session.rollback()
        else:
            logger.info(f'Not logging stats due to environment variable settings: Tile {tile_id} ({tile_name}) {timestamp}')

        return tweet_count

    def compare_activity(self, tweet_count: int, tile_id: int, hs_hour: Dict, hs_day: Dict):
        event_hour = False
        event_day = False
        if tweet_count > 0:
            logger.info(f'tile_id: {tile_id}, tweet_count: {tweet_count}')
            if hs_hour:
                threshold_hour = hs_hour[tile_id].mean + (hs_hour[tile_id].stddev * 2)
                event_hour = (tweet_count >= EVENT_MIN_TWEETS) and (tweet_count > threshold_hour)
                # logger.info(f'Now vs hour: {event_hour}')
                # logger.info(f'    hour time: {hs_hour[tile_id].timestamp}, count: {hs_hour[tile_id].count}')
                logger.info(f'    hour threshold: {threshold_hour:.3f} = {hs_hour[tile_id].mean:.3f} + ({hs_hour[tile_id].stddev:.3f} * 2)')
            if hs_day:
                threshold_day = hs_day[tile_id].mean + (hs_day[tile_id].stddev * 2)
                event_day = (tweet_count >= EVENT_MIN_TWEETS) and (tweet_count > threshold_day)
                # logger.info(f'Now vs day: {event_day}')
                # logger.info(f'    day time: {hs_day[tile_id].timestamp}, count: {hs_day[tile_id].count}')
                logger.info(f'    day threshold: {threshold_day:.3f} = {hs_day[tile_id].mean:.3f} + ({hs_day[tile_id].stddev:.3f} * 2)')
            else:
                # If no stats for the previous day, only base event decision on previous hour
                if IGNORE_MISSING_DAY_STATS:
                    event_day = True

        found_event = False
        # Note that this tile had an event
        if event_hour and event_day:
            found_event = True

        return found_event

    def log_event_and_get_str(self, event_tweets, tile_id: int, timestamp: datetime, token_count_min: int = None):
        # Prepare the tweet text
        tokens_to_tweet = get_tokens_to_tweet(event_tweets, token_count_min=token_count_min)
        tokens_str = ' '.join(tokens_to_tweet)

        # Compute the average tweet location
        lons = [et.longitude for et in event_tweets]
        longitude = sum(lons) / len(lons)
        lats = [et.latitude for et in event_tweets]
        latitude = sum(lats) / len(lats)
        lat_long_str = f'{latitude:.4f}, {longitude:.4f}'

        # Get a label for this location if it is a neighborhood or point-of-interest, otherwise get tile name
        tile_identity = Tiles.get_tile_name(session, tile_id=tile_id)[0]
        tile_name, tile_name_type = tile_identity[1], tile_identity[2]
        # Get the most common place name from these tweets; only consider neighborhood or poi
        place_names = [
            et.place_name for et in event_tweets if et.place_type in ['neighborhood', 'poi']
        ]
        place_name = Counter(place_names).most_common(1)[0][0] if place_names else None
        # Get a larger granular name to include after a neighborhood or poi
        city_name = Tiles.get_tile_name(session, tile_id=tile_id, geo_granularity=['city', 'admin', 'country'])[0][1]
        # If the tweets didn't have a valid place name, fall back to the tile name
        if not place_name:
            place_name = tile_name
            # Don't include the city name if granularity is larger than neighborhood
            if tile_name_type != 'neighborhood':
                city_name = None

        # Construct the message to tweet
        event_str = "Something's happening"
        event_str = f'{event_str} in {place_name}' if place_name else event_str
        event_str = f'{event_str}, {city_name}' if city_name else event_str
        event_str = f'{event_str} ({lat_long_str}):'
        event_url = BASE_EVENT_URL + '&'.join([f'{i+1}={et.status_id_str}' for i, et in enumerate(event_tweets)])
        remaining_chars = TWEET_MAX_LENGTH - len(event_str) - 2 - TWEET_URL_LENGTH
        # Find the largest set of tokens allowed for the length of a tweet
        possible_token_sets = [' '.join(tokens_to_tweet[:i]) for i in range(1, len(tokens_to_tweet) + 1)[::-1]]
        mask = [len(x) <= remaining_chars for x in possible_token_sets]
        tokens = [t for t, m in zip(possible_token_sets, mask) if m][0]
        event_str = f'{event_str} {tokens} {event_url}'

        logger.info(f'{timestamp} Tile {tile_id} ({tile_name}) {timestamp}: Found event with {len(event_tweets)} tweets')
        logger.info(event_str)

        # Add to events table
        if LOG_EVENTS:
            ev = Events(
                tile_id=tile_id,
                timestamp=timestamp,
                count=len(event_tweets),
                longitude=longitude,
                latitude=latitude,
                place_name=place_name,
                description=tokens_str,
            )
            session.add(ev)

            try:
                session.commit()
                logger.info(f'Logged event: Tile {tile_id} ({tile_name}) {timestamp}: {tokens_str}')
            except Exception:
                logger.exception(f'Exception when logging event: Tile {tile_id} ({tile_name}) {timestamp}: {tokens_str}')
                session.rollback()
        else:
            logger.info(f'Not logging event due to environment variable settings: Tile {tile_id} ({tile_name}) {timestamp}: {tokens_str}')

        return event_str

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


def get_geo_info(bounding_box: List[float], tile_size: float):
    tile_longitudes = np.arange(bounding_box[0], bounding_box[2], tile_size).tolist()
    # np.arange does not include the end point; add the edge of bounding box
    tile_longitudes.append(bounding_box[2])
    tile_latitudes = np.arange(bounding_box[1], bounding_box[3], tile_size).tolist()
    # np.arange does not include the end point; add the edge of bounding box
    tile_latitudes.append(bounding_box[3])

    num_tiles = len(list(n_wise(tile_latitudes, 2))) * len(list(n_wise(tile_longitudes, 2)))
    # Assumes one stat per hour
    assert (num_tiles * HISTORICAL_STATS_DAYS_TO_KEEP * 24) <= MAX_ROWS_HISTORICAL_STATS

    geo_granularity = ['neighborhood', 'city', 'admin', 'country']

    # Initialize - index starts with 1
    geo_info = {
        i: {
            'west_lon': None,
            'east_lon': None,
            'south_lat': None,
            'north_lat': None,
            'neighborhood': None,
            'city': None,
            'admin': None,
            'country': None,
        }
        for i in range(1, num_tiles + 1)
    }

    logger.info('Reverse geocoding tiles to assign names')
    # Initialize tile index counter, start with 1
    i = 1
    for tile_lats in n_wise(tile_latitudes, 2):
        south_lat, north_lat = tile_lats[0], tile_lats[1]
        for tile_lons in n_wise(tile_longitudes, 2):
            west_lon, east_lon = tile_lons[0], tile_lons[1]
            if geo_info[i]['country'] is None:
                geo_info[i]['west_lon'] = west_lon
                geo_info[i]['east_lon'] = east_lon
                geo_info[i]['south_lat'] = south_lat
                geo_info[i]['north_lat'] = north_lat

                latitude = (south_lat + north_lat) / 2
                longitude = (west_lon + east_lon) / 2

                # Reverse geocode the tile's center latitude and longitude value to store tile names
                unsuccessful_tries = 0
                try_threshold = 10
                while unsuccessful_tries < try_threshold:
                    rev_geo = twitter.reverse_geocode(lat=latitude, long=longitude, granularity='neighborhood')
                    if 'result' in rev_geo:
                        unsuccessful_tries = try_threshold
                    else:
                        unsuccessful_tries += 1
                        logger.info('Sleeping for 10 seconds due to failed reverse geocode')
                        sleep(10)
                for gg in geo_granularity:
                    if 'result' in rev_geo:
                        tile_name = [x['name'] for x in rev_geo['result']['places'] if x['place_type'] == gg]
                    else:
                        tile_name = []
                    geo_info[i][gg] = tile_name[0] if tile_name else None

                logger.info(f"Got geo info for Tile {i} ({latitude:.4f}, {longitude:.4f}): {geo_info[i]['neighborhood']}, {geo_info[i]['city']}, {geo_info[i]['admin']}, {geo_info[i]['country']}")
                logger.info('Sleeping for 60 seconds due to reverse geocode rate limit')
                sleep(60)
            i += 1

    return geo_info


# Establish connection to Twitter
# Uses OAuth1 ("user auth") for authentication
twitter = Twython(
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
    geo_info = get_geo_info(bounding_box=BOUNDING_BOX, tile_size=TILE_SIZE)
    num_tiles = len(geo_info)

    logger.info('Populating tiles table')
    for i in range(1, num_tiles + 1):
        tile = Tiles(
            west_lon=geo_info[i]['west_lon'],
            east_lon=geo_info[i]['east_lon'],
            south_lat=geo_info[i]['south_lat'],
            north_lat=geo_info[i]['north_lat'],
            neighborhood=geo_info[i]['neighborhood'],
            city=geo_info[i]['city'],
            admin=geo_info[i]['admin'],
            country=geo_info[i]['country'],
        )
        session.add(tile)

        logger.info(f"Logged Tile {i}: {geo_info[i]['neighborhood']}, {geo_info[i]['city']}, {geo_info[i]['admin']}, {geo_info[i]['country']}")
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
