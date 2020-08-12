from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
# from pprint import pformat, pprint

from dotenv import load_dotenv
import numpy as np
import pytz
from runstats import Statistics
from twython import Twython
from twython import TwythonStreamer

from utils.data_base import session_factory, Tiles, RecentTweets, HistoricalStats
from utils.tweet_utils import check_tweet, date_string_to_datetime
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
    EVERY_N_SECONDS = 1
    TEMPORAL_GRANULARITY_HOURS = 1
    INITIAL_TIME = datetime(1970, 1, 1)
    ECHO = False
else:
    logger.setLevel(logging.INFO)
    # POST_EVENT = os.getenv("POST_EVENT", default="False") == "True"
    EVERY_N_SECONDS = int(os.getenv("EVERY_N_SECONDS", default="3600"))
    TEMPORAL_GRANULARITY_HOURS = int(os.getenv("TEMPORAL_GRANULARITY_HOURS", default="1"))
    # Wait half the rate limit time before making first post
    INITIAL_TIME = datetime.utcnow().replace(tzinfo=pytz.UTC) - timedelta(seconds=EVERY_N_SECONDS // 2)
    ECHO = False

APP_KEY = os.getenv("API_KEY", default=None)
APP_SECRET = os.getenv("API_SECRET", default=None)
OAUTH_TOKEN = os.getenv("ACCESS_TOKEN", default=None)
OAUTH_TOKEN_SECRET = os.getenv("ACCESS_TOKEN_SECRET", default=None)
DATABASE_URL = os.getenv("DATABASE_URL", default=None)
MY_SCREEN_NAME = os.getenv("MY_SCREEN_NAME", default="twitter")
LANGUAGE = os.getenv("LANGUAGE", default="en")

tile_size = 0.01

# Set the bounding box: longitude and latitude pairs for SW and NE corner (in that order)

# # San Francisco down to SFO
# bounding_box = [-122.52, 37.59, -122.36, 37.82]  # 432 tiles

# SF only
bounding_box = [-122.52, 37.71, -122.36, 37.82]  # 150 tiles

# # Bay Area including Oakland and Santa Cruz
# bounding_box = [-122.52, 36.94, -121.8, 38.0]

# # Bay Area including Santa Cruz, from Twitter's dev site
# bounding_box = [-122.75, 36.8, -121.75, 37.8]


class MyTwitterClient(Twython):
    '''Wrapper around the Twython Twitter client.
    Limits status update rate.
    '''
    def __init__(self, initial_time=None, *args, **kwargs):
        super(MyTwitterClient, self).__init__(*args, **kwargs)
        if initial_time is None:
            # Wait half the rate limit time before making first post
            initial_time = datetime.utcnow().replace(tzinfo=pytz.UTC) - timedelta(seconds=EVERY_N_SECONDS // 2)
        self.last_post_time = initial_time

    def update_status_check_rate(self, *args, **kwargs):
        current_time = datetime.utcnow().replace(tzinfo=pytz.UTC)
        logger.info(f'Current time: {current_time}')
        logger.info(f'Previous post time: {self.last_post_time}')
        logger.info(f'Difference: {current_time - self.last_post_time}')
        if (current_time - self.last_post_time).total_seconds() > EVERY_N_SECONDS:
            self.update_status(*args, **kwargs)
            self.last_post_time = current_time
            logger.info('Success')
            return True
        else:
            logger.info('Not posting haiku due to rate limit')
            return False


class MyStreamer(TwythonStreamer):
    def __init__(self, running_stats=None, comparison_tweet_time=None, *args, **kwargs):
        super(MyStreamer, self).__init__(*args, **kwargs)
        self.running_stats = running_stats if running_stats else {}
        if comparison_tweet_time is None:
            comparison_tweet_time = datetime.utcnow().replace(tzinfo=pytz.UTC)
        self.comparison_tweet_time = comparison_tweet_time

    def on_success(self, status):
        if check_tweet(status):
            status_id_str = status['id_str']
            user_screen_name = status['user']['screen_name']
            user_id_str = status['user']['id_str']
            created_at = date_string_to_datetime(status['created_at'])
            tweet_body = status['text']
            tweet_language = status['lang']
            longitude = status['coordinates']['coordinates'][0]
            latitude = status['coordinates']['coordinates'][1]
            place_name = status['place']['full_name']
            # Possible values: country, admin, city, neighborhood, poi; more?
            place_type = status['place']['place_type']
            tiles = Tiles.find_id_by_coords(session, longitude, latitude)

            if tiles:
                tile = tiles[0]
                tweet = RecentTweets(
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
                    tile_id=tile.id,
                )
                session.add(tweet)
                try:
                    session.commit()
                    logger.info(f'Logged tweet {status_id_str}')
                except Exception:
                    logger.exception(f'Exception when adding recent tweet {status_id_str}')
                    session.rollback()

                if created_at - self.comparison_tweet_time >= timedelta(hours=1):
                    logger.info(f'current time: {created_at}')
                    logger.info('Been more than 1 hour between oldest and current tweet')
                    # Get tweets from the last hour
                    tweet_counts = RecentTweets.count_tweets_per_tile(session, hours=1)
                    tweet_counts = {tile_id: count for tile_id, count in tweet_counts}

                    hs_hour = HistoricalStats.get_recent_stats(session, hours=1)
                    hs_hour = {tile_id: row for tile_id, row in hs_hour}

                    hs_day = HistoricalStats.get_recent_stats(session, days=1)
                    hs_day = {tile_id: row for tile_id, row in hs_day}

                    hs_week = HistoricalStats.get_recent_stats(session, weeks=1)
                    hs_week = {tile_id: row for tile_id, row in hs_week}

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
                            timestamp=created_at,
                            count=tweet_count,
                            mean=mean,
                            variance=variance,
                            stddev=stddev,
                        )
                        session.add(hs)

                        event_min_tweets = 4
                        event_hour = False
                        event_day = False
                        event_week = False
                        if tweet_count > 0:
                            logger.info(f'tile_id: {tile_id}, tweet_count: {tweet_count}')
                            if hs_hour:
                                threshold = hs_hour[tile_id].mean + (hs_hour[tile_id].stddev * 2)
                                event_hour = (tweet_count >= event_min_tweets) and (tweet_count > threshold)
                                logger.info(f'Now vs hour: {event_hour}')
                                logger.info(f'    hour time: {hs_hour[tile_id].timestamp}, count: {hs_hour[tile_id].count}')
                                logger.info(f'    hour threshold: {threshold} = {hs_hour[tile_id].mean} + ({hs_hour[tile_id].stddev} * 2)')
                            if hs_day:
                                threshold = hs_day[tile_id].mean + (hs_day[tile_id].stddev * 2)
                                event_day = (tweet_count >= event_min_tweets) and (tweet_count > threshold)
                                logger.info(f'Now vs day: {event_day}')
                                logger.info(f'    day time: {hs_day[tile_id].timestamp}, count: {hs_day[tile_id].count}')
                                logger.info(f'    day threshold: {threshold} = {hs_day[tile_id].mean} + ({hs_day[tile_id].stddev} * 2)')
                            if hs_week:
                                threshold = hs_week[tile_id].mean + (hs_week[tile_id].stddev * 2)
                                event_week = (tweet_count >= event_min_tweets) and (tweet_count > threshold)
                                logger.info(f'Now vs week: {event_week}')
                                logger.info(f'    week time: {hs_week[tile_id].timestamp}, count: {hs_week[tile_id].count}')
                                logger.info(f'    week threshold: {threshold} = {hs_week[tile_id].mean} + ({hs_week[tile_id].stddev} * 2)')

                        if any([event_hour, event_day, event_week]):
                            events[tile_id] = True

                    if any(events.values()):
                        tiles_with_events = [tile_id for tile_id, event in events.items() if event]
                        event_tweets = RecentTweets.get_recent_tweets(session, hours=1)
                        for tile_id in tiles_with_events:
                            tile_event_tweets = [et for et in event_tweets if et.tile_id == tile_id]
                            logger.info(f'Tile {tile_id} event: Found {len(tile_event_tweets)} tweets')

                    try:
                        session.commit()
                    except Exception:
                        logger.exception(f'Exception when adding historical stats for {created_at}')
                        session.rollback()

                    # Update the comparison tweet time
                    self.comparison_tweet_time = created_at

                    # Delete old historical stats rows
                    logger.info('Deleting old historical stats')
                    HistoricalStats.delete_older_than_weeks(session, weeks=1)

                    # # Delete old recent tweets rows
                    # logger.info('Deleting old recent tweets')
                    # RecentTweets.delete_older_than_hours(session, hours=1)
            else:
                logger.warning(f'Tweet {status_id_str} coordinates ({longitude}, {latitude}) matched incorrect number of tiles: {len(tiles)}')

    def on_error(self, status_code, status):
        logger.info(f'Error while streaming. status_code: {status_code}, status: {status}')


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
    tile_longitudes = np.arange(bounding_box[0], bounding_box[2], tile_size)
    tile_latitudes = np.arange(bounding_box[1], bounding_box[3], tile_size)

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
    # Populate running stats objects for each tile from table
    stats_counts = {i: [] for i in range(1, num_tiles + 1)}
    for i, stats in stats_all:
        stats_counts[i].append(stats.count)
    running_stats = {i: Statistics(stats_counts[i]) for i in range(1, num_tiles + 1)}
else:
    logger.info('Initializing new running stats objects')
    # Initialize new running stats objects
    running_stats = {i: Statistics() for i in range(1, num_tiles + 1)}

# comparison_tweet = RecentTweets.get_oldest_tweet(session)
comparison_tweet = RecentTweets.get_most_recent_tweet(session)
if comparison_tweet:
    comparison_tweet_time = comparison_tweet.created_at.replace(tzinfo=pytz.UTC)
else:
    comparison_tweet_time = None

if __name__ == '__main__':
    logger.info('Initializing tweet streamer...')
    stream = MyStreamer(
        running_stats=running_stats,
        comparison_tweet_time=comparison_tweet_time,
        app_key=APP_KEY,
        app_secret=APP_SECRET,
        oauth_token=OAUTH_TOKEN,
        oauth_token_secret=OAUTH_TOKEN_SECRET,
    )

    bounding_box_str = ','.join([str(x) for x in bounding_box])
    while True:
        try:
            stream.statuses.filter(locations=bounding_box_str)
        except Exception:
            logger.exception('Exception when streaming tweets')
            continue
