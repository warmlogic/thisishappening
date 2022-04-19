from thisishappening.utils.tweet_utils import get_event_info

twitter = None

cluster_tweets = []


def test_get_event_info():
    event_info = get_event_info(
        twitter,
        event_tweets=cluster_tweets,
        # tweet_max_length=TWEET_MAX_LENGTH,
        # tweet_url_length=TWEET_URL_LENGTH,
        # base_event_url=BASE_EVENT_URL,
        # event_str=event_str,
        # event_type=event_type,
        # token_count_min=token_count_min,
        # reduce_token_count_min=reduce_token_count_min,
        # remove_username_at=REMOVE_USERNAME_AT,
        # tweet_lat_lon=TWEET_LAT_LON,
        # show_tweets_on_event=SHOW_TWEETS_ON_EVENT,
    )

    print(event_info)
