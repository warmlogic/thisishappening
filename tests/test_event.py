from thisishappening.utils.tweet_utils import get_event_info

from .data_event import clusters

twitter = None
TWEET_MAX_LENGTH = 280
TWEET_URL_LENGTH = 23
BASE_EVENT_URL = "https://mypage/thisishappening/?"
EVENT_TYPE = "moment"
EVENT_STR = None
TOKEN_COUNT_MIN = 2
REDUCE_TOKEN_COUNT_MIN = True
REMOVE_USERNAME_AT = True
TWEET_LAT_LON = False
SHOW_TWEETS_ON_EVENT = False


def test_get_event_info():
    event_tweets = clusters[1]["event_tweets"]

    event_info = get_event_info(
        twitter,
        event_tweets=event_tweets,
        tweet_max_length=TWEET_MAX_LENGTH,
        tweet_url_length=TWEET_URL_LENGTH,
        base_event_url=BASE_EVENT_URL,
        event_str=EVENT_STR,
        event_type=EVENT_TYPE,
        token_count_min=TOKEN_COUNT_MIN,
        reduce_token_count_min=REDUCE_TOKEN_COUNT_MIN,
        remove_username_at=REMOVE_USERNAME_AT,
        tweet_lat_lon=TWEET_LAT_LON,
        show_tweets_on_event=SHOW_TWEETS_ON_EVENT,
    )

    assert (
        event_info.tokens_str
        == "bostonmarathon boston marathon running amazing meet great coverage"
        + " congratulations 26.2 today day nice brutal"
        + " #marathon #running #bostonmarathon"
    )
