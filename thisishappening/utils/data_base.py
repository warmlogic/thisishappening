import logging
from datetime import datetime, timedelta
from typing import List

import pytz
from sqlalchemy import (
    ARRAY,
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
    desc,
    func,
    or_,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from .data_utils import get_coords_min_max
from .tweet_utils import EventInfo, TweetInfo

logger = logging.getLogger("happeninglogger")

Base = declarative_base()


def session_factory(DATABASE_URL: str, echo: bool = False):
    engine = create_engine(DATABASE_URL, poolclass=NullPool, echo=echo)
    Base.metadata.create_all(engine)
    _SessionFactory = sessionmaker(bind=engine)
    return _SessionFactory()


class Events(Base):
    """To drop this table, run Events.metadata.drop_all(engine)"""

    __tablename__ = "events"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    count = Column(Integer, nullable=False)
    longitude = Column(Float(precision=8), nullable=False)
    latitude = Column(Float(precision=8), nullable=False)
    west_lon = Column(Float(precision=8), nullable=False)
    south_lat = Column(Float(precision=8), nullable=False)
    east_lon = Column(Float(precision=8), nullable=False)
    north_lat = Column(Float(precision=8), nullable=False)
    place_name = Column(String, nullable=True)
    description = Column(String, nullable=False)
    status_ids = Column(ARRAY(String), nullable=True)

    @classmethod
    def log_event(cls, session, event_info: EventInfo):
        # Add to events table
        event = cls(
            timestamp=event_info.timestamp,
            count=event_info.n,
            longitude=event_info.longitude,
            latitude=event_info.latitude,
            west_lon=event_info.west_lon,
            south_lat=event_info.south_lat,
            east_lon=event_info.east_lon,
            north_lat=event_info.north_lat,
            place_name=event_info.place_name,
            description=event_info.tokens_str,
            status_ids=event_info.status_ids,
        )
        session.add(event)
        try:
            session.commit()
            logger.info(
                f"Logged event: {event_info.timestamp} {event_info.place_name}:"
                + f" {event_info.tokens_str}"
            )
        except Exception as e:
            logger.warning(
                "Exception when logging event:"
                + f" {event_info.timestamp} {event_info.place_name}:"
                + f" {event_info.tokens_str}: {e}"
            )
            session.rollback()

        return event

    @classmethod
    def get_recent_events(cls, session, timestamp: datetime = None, hours: float = 1):
        if timestamp is None:
            timestamp = datetime.utcnow().replace(tzinfo=pytz.UTC)

        ts_start = timestamp - timedelta(hours=hours)

        q = (
            session.query(cls)
            .filter(cls.timestamp >= ts_start)
            .filter(cls.timestamp <= timestamp)
        )

        return q.order_by(desc(cls.timestamp)).all()

    @classmethod
    def get_most_recent_event(cls, session):
        q = session.query(cls)

        return q.order_by(desc(cls.timestamp)).first()

    @classmethod
    def get_event_tweets(cls, session, event_id: int, hours: float = 1):
        event = (
            session.query(cls)
            .filter(cls.id == event_id)
            .order_by(desc(cls.timestamp))
            .first()
        )
        if event is not None:
            timestamp = event.timestamp.replace(tzinfo=pytz.UTC)
            bounding_box = [
                event.west_lon,
                event.south_lat,
                event.east_lon,
                event.north_lat,
            ]
            event_tweets = RecentTweets.get_recent_tweets(
                session, timestamp=timestamp, hours=hours, bounding_box=bounding_box
            )
        else:
            logger.info(f"Event ID {event_id} not found")
            event_tweets = []

        return event_tweets

    def __repr__(self):
        return f"Event {self.id}"

    @classmethod
    def delete_events_older_than(
        cls,
        session,
        timestamp: datetime = None,
        hours: float = None,
        days: float = None,
        weeks: float = None,
    ):
        """Delete all records older than the specified time window,
        optionally relative to a timestamp"""
        hours = hours or 0
        days = days or 0
        weeks = weeks or 0

        if any([hours, days, weeks]):
            if timestamp is None:
                timestamp = datetime.utcnow().replace(tzinfo=pytz.UTC)

            ts_end = timestamp - timedelta(hours=hours, days=days, weeks=weeks)
            try:
                logger.info(
                    f"Deleting events older than {ts_end}:"
                    + f" {hours} hours {days} days {weeks} weeks"
                )
                delete_q = cls.__table__.delete().where(cls.timestamp < ts_end)
                session.execute(delete_q)
                session.commit()
            except Exception as e:
                logger.warning(
                    f"Exception when deleting events older than {ts_end}:"
                    + f" {hours} hours {days} days {weeks} weeks: {e}"
                )
                session.rollback()

    @classmethod
    def keep_events_n_rows(cls, session, n: int = None):
        """Keep the most recent n rows"""
        if n is not None:
            ids = session.query(cls.id).order_by(desc(cls.timestamp)).all()
            ids_to_delete = [x[0] for x in ids[n:]]

            if ids_to_delete:
                try:
                    logger.info(f"Keeping most recent {n} rows of events")
                    delete_q = cls.__table__.delete().where(cls.id.in_(ids_to_delete))

                    session.execute(delete_q)
                    session.commit()
                except Exception as e:
                    logger.warning(
                        f"Exception when keeping most recent {n} rows of events: {e}"
                    )
                    session.rollback()


class RecentTweets(Base):
    """To drop this table, run RecentTweets.metadata.drop_all(engine)"""

    __tablename__ = "recent_tweets"

    id = Column(Integer, primary_key=True)
    status_id_str = Column(String, nullable=False)
    user_screen_name = Column(String, nullable=False)
    user_id_str = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)
    tweet_body = Column(String, nullable=False)
    tweet_language = Column(String, nullable=True)
    has_coords = Column(Boolean, nullable=False, default=False)
    longitude = Column(Float(precision=8), nullable=False)
    latitude = Column(Float(precision=8), nullable=False)
    place_id = Column(String, nullable=True)
    place_name = Column(String, nullable=True)
    place_type = Column(String, nullable=True)

    @classmethod
    def log_tweet(cls, session, tweet_info: TweetInfo):
        tweet = cls(
            status_id_str=tweet_info.status_id_str,
            user_screen_name=tweet_info.user_screen_name,
            user_id_str=tweet_info.user_id_str,
            created_at=tweet_info.created_at,
            tweet_body=tweet_info.tweet_body,
            tweet_language=tweet_info.tweet_language,
            has_coords=tweet_info.has_coords,
            longitude=tweet_info.longitude,
            latitude=tweet_info.latitude,
            place_id=tweet_info.place_id,
            place_name=tweet_info.place_name,
            place_type=tweet_info.place_type,
        )
        session.add(tweet)
        try:
            session.commit()
            logger.info(
                f"Logged tweet: {tweet_info.status_id_str},"
                + f" coordinates: ({tweet_info.latitude}, {tweet_info.longitude}),"
                + f" {tweet_info.place_name} ({tweet_info.place_type})"
            )
        except Exception as e:
            logger.warning(
                f"Exception when logging tweet: {tweet_info.status_id_str},"
                + f" coordinates: ({tweet_info.latitude}, {tweet_info.longitude}),"
                + f" {tweet_info.place_name} ({tweet_info.place_type}): {e}"
            )
            session.rollback()

        return tweet

    @classmethod
    def count_tweets(
        cls,
        session,
        timestamp: datetime = None,
        hours: float = 0,
        bounding_box: List[float] = None,
    ):
        if timestamp is None:
            timestamp = datetime.utcnow().replace(tzinfo=pytz.UTC)

        ts_start = timestamp - timedelta(hours=hours)

        q = (
            session.query(cls, func.count(cls.status_id_str))
            .filter(cls.created_at >= ts_start)
            .filter(cls.created_at <= timestamp)
        )

        if bounding_box is not None:
            west_lon, east_lon, south_lat, north_lat = get_coords_min_max(
                bounding_box=bounding_box
            )
            q = (
                q.filter(cls.longitude >= west_lon)
                .filter(cls.longitude < east_lon)
                .filter(cls.latitude >= south_lat)
                .filter(cls.latitude < north_lat)
            )

        return q.all()

    @classmethod
    def get_recent_tweets(
        cls,
        session,
        timestamp: datetime = None,
        hours: float = 1,
        bounding_box: List[float] = None,
        place_type: List[str] = None,
        has_coords: bool = None,
        place_type_or_coords: bool = True,
    ):
        if timestamp is None:
            timestamp = datetime.utcnow().replace(tzinfo=pytz.UTC)

        ts_start = timestamp - timedelta(hours=hours)

        q = (
            session.query(cls)
            .filter(cls.created_at >= ts_start)
            .filter(cls.created_at <= timestamp)
        )

        if bounding_box is not None:
            west_lon, east_lon, south_lat, north_lat = get_coords_min_max(
                bounding_box=bounding_box
            )
            q = (
                q.filter(cls.longitude >= west_lon)
                .filter(cls.longitude < east_lon)
                .filter(cls.latitude >= south_lat)
                .filter(cls.latitude < north_lat)
            )

        if (
            place_type_or_coords
            and (place_type is not None)
            and (has_coords is not None)
        ):
            q = q.filter(
                or_(cls.place_type.in_(place_type), cls.has_coords.is_(has_coords))
            )
        else:
            if place_type is not None:
                q = q.filter(cls.place_type.in_(place_type))

            if has_coords is not None:
                q = q.filter(cls.has_coords.is_(has_coords))

        return q.order_by(desc(cls.created_at)).all()

    @classmethod
    def get_oldest_tweet(cls, session, bounding_box: List[float] = None):
        q = session.query(cls)

        if bounding_box is not None:
            west_lon, east_lon, south_lat, north_lat = get_coords_min_max(
                bounding_box=bounding_box
            )
            q = (
                q.filter(cls.longitude >= west_lon)
                .filter(cls.longitude < east_lon)
                .filter(cls.latitude >= south_lat)
                .filter(cls.latitude < north_lat)
            )

        return q.order_by(cls.created_at).first()

    @classmethod
    def get_most_recent_tweet(cls, session, bounding_box: List[float] = None):
        q = session.query(cls)

        if bounding_box is not None:
            west_lon, east_lon, south_lat, north_lat = get_coords_min_max(
                bounding_box=bounding_box
            )
            q = (
                q.filter(cls.longitude >= west_lon)
                .filter(cls.longitude < east_lon)
                .filter(cls.latitude >= south_lat)
                .filter(cls.latitude < north_lat)
            )

        return q.order_by(desc(cls.created_at)).first()

    @classmethod
    def delete_tweets_older_than(
        cls,
        session,
        timestamp: datetime = None,
        hours: float = None,
        days: float = None,
        weeks: float = None,
    ):
        """Delete all records older than the specified time window,
        optionally relative to a timestamp"""
        hours = hours or 0
        days = days or 0
        weeks = weeks or 0

        if any([hours, days, weeks]):
            if timestamp is None:
                timestamp = datetime.utcnow().replace(tzinfo=pytz.UTC)

            ts_end = timestamp - timedelta(hours=hours, days=days, weeks=weeks)
            try:
                logger.info(
                    f"Deleting tweets older than {ts_end}:"
                    + f" {hours} hours {days} days {weeks} weeks"
                )
                delete_q = cls.__table__.delete().where(cls.created_at < ts_end)
                session.execute(delete_q)
                session.commit()
            except Exception as e:
                logger.warning(
                    f"Exception when deleting tweets older than {ts_end}:"
                    + f" {hours} hours {days} days {weeks} weeks: {e}"
                )
                session.rollback()

    @classmethod
    def keep_tweets_n_rows(cls, session, n: int = None):
        """Keep the most recent n rows"""
        if n is not None:
            ids = session.query(cls.id).order_by(desc(cls.created_at)).all()
            ids_to_delete = [x[0] for x in ids[n:]]

            if ids_to_delete:
                try:
                    logger.info(f"Keeping most recent {n} rows of tweets")
                    delete_q = cls.__table__.delete().where(cls.id.in_(ids_to_delete))

                    session.execute(delete_q)
                    session.commit()
                except Exception as e:
                    logger.warning(
                        f"Exception when keeping most recent {n} rows of tweets: {e}"
                    )
                    session.rollback()
