from datetime import datetime, timedelta
import logging
from typing import List

import pytz
from sqlalchemy import create_engine, desc, func
from sqlalchemy import Column, String, Integer, DateTime, Float, ARRAY
from sqlalchemy.pool import NullPool
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger("happeninglogger")

Base = declarative_base()


def session_factory(DATABASE_URL: str, echo: bool = False):
    engine = create_engine(DATABASE_URL, poolclass=NullPool, echo=echo)
    Base.metadata.create_all(engine)
    _SessionFactory = sessionmaker(bind=engine)
    return _SessionFactory()


class Events(Base):
    '''To drop this table, run Events.metadata.drop_all(engine)
    '''
    __tablename__ = 'events'

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
    def get_recent_events(cls, session, timestamp: datetime = None, hours: float = 1):
        if timestamp is None:
            timestamp = datetime.utcnow().replace(tzinfo=pytz.UTC)

        ts_start = timestamp - timedelta(hours=hours)

        q = session.query(cls).filter(cls.timestamp >= ts_start).filter(cls.timestamp <= timestamp)

        return q.order_by(desc(cls.timestamp)).all()

    @classmethod
    def get_most_recent_event(cls, session):
        q = session.query(cls)

        return q.order_by(desc(cls.timestamp)).first()

    @classmethod
    def get_event_tweets(cls, session, event_id: int, hours: float = 1):
        event = session.query(cls).filter(cls.id == event_id).order_by(desc(cls.timestamp)).first()
        if event is not None:
            timestamp = event.timestamp.replace(tzinfo=pytz.UTC)
            bounding_box = [event.west_lon, event.south_lat, event.east_lon, event.north_lat]
            event_tweets = RecentTweets.get_recent_tweets(session, timestamp=timestamp, hours=hours, bounding_box=bounding_box)
        else:
            logger.info(f"Event ID {event_id} not found")
            event_tweets = []

        return event_tweets

    def __repr__(self):
        return f'Event {self.id}'

    @classmethod
    def delete_events_older_than(
        cls,
        session,
        timestamp: datetime = None,
        hours: float = None,
        days: float = None,
        weeks: float = None,
    ):
        '''Delete all records older than the specified time window, optionally relative to a timestamp
        '''
        hours = hours if hours else 0
        days = days if days else 0
        weeks = weeks if weeks else 0

        if any([hours, days, weeks]):
            if timestamp is None:
                timestamp = datetime.utcnow().replace(tzinfo=pytz.UTC)

            ts_end = timestamp - timedelta(hours=hours, days=days, weeks=weeks)
            try:
                logger.info(f'Deleting events older than {ts_end}: {hours} hours {days} days {weeks} weeks')
                delete_q = cls.__table__.delete().where(cls.timestamp < ts_end)
                session.execute(delete_q)
                session.commit()
            except Exception:
                logger.exception(f'Exception when deleting events older than {ts_end}: {hours} hours {days} days {weeks} weeks')
                session.rollback()

    @classmethod
    def keep_events_n_rows(cls, session, n: int = None):
        '''Keep the most recent n rows
        '''
        if n is not None:
            ids = session.query(cls.id).order_by(desc(cls.timestamp)).all()
            ids_to_delete = [x[0] for x in ids[n:]]

            if ids_to_delete:
                try:
                    logger.info(f'Keeping most recent {n} rows of events')
                    delete_q = cls.__table__.delete().where(cls.id.in_(ids_to_delete))

                    session.execute(delete_q)
                    session.commit()
                except Exception:
                    logger.exception(f'Exception when keeping most recent {n} rows of events')
                    session.rollback()


class RecentTweets(Base):
    '''To drop this table, run RecentTweets.metadata.drop_all(engine)
    '''
    __tablename__ = 'recent_tweets'

    id = Column(Integer, primary_key=True)
    status_id_str = Column(String, nullable=False)
    user_screen_name = Column(String, nullable=False)
    user_id_str = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)
    tweet_body = Column(String, nullable=False)
    tweet_language = Column(String, nullable=True)
    longitude = Column(Float(precision=8), nullable=False)
    latitude = Column(Float(precision=8), nullable=False)
    place_name = Column(String, nullable=True)
    place_type = Column(String, nullable=True)

    @classmethod
    def count_tweets(cls, session, timestamp: datetime = None, hours: float = 0, bounding_box: List[float] = None):
        if timestamp is None:
            timestamp = datetime.utcnow().replace(tzinfo=pytz.UTC)

        ts_start = timestamp - timedelta(hours=hours)

        q = session.query(
            cls, func.count(cls.status_id_str)).filter(
                cls.created_at >= ts_start).filter(cls.created_at <= timestamp)

        if bounding_box is not None:
            west_lon, south_lat, east_lon, north_lat = bounding_box
            q = q.filter(
                cls.longitude >= west_lon).filter(
                cls.longitude < east_lon).filter(
                cls.latitude >= south_lat).filter(
                cls.latitude < north_lat)

        return q.all()

    @classmethod
    def get_recent_tweets(cls, session, timestamp: datetime = None, hours: float = 1, bounding_box: List[float] = None):
        if timestamp is None:
            timestamp = datetime.utcnow().replace(tzinfo=pytz.UTC)

        ts_start = timestamp - timedelta(hours=hours)

        q = session.query(cls).filter(cls.created_at >= ts_start).filter(cls.created_at <= timestamp)

        if bounding_box is not None:
            west_lon, south_lat, east_lon, north_lat = bounding_box
            q = q.filter(
                cls.longitude >= west_lon).filter(
                cls.longitude < east_lon).filter(
                cls.latitude >= south_lat).filter(
                cls.latitude < north_lat)

        return q.order_by(desc(cls.created_at)).all()

    @classmethod
    def get_oldest_tweet(cls, session, bounding_box: List[float] = None):
        q = session.query(cls)

        if bounding_box is not None:
            west_lon, south_lat, east_lon, north_lat = bounding_box
            q = q.filter(
                cls.longitude >= west_lon).filter(
                cls.longitude < east_lon).filter(
                cls.latitude >= south_lat).filter(
                cls.latitude < north_lat)

        return q.order_by(cls.created_at).first()

    @classmethod
    def get_most_recent_tweet(cls, session, bounding_box: List[float] = None):
        q = session.query(cls)

        if bounding_box is not None:
            west_lon, south_lat, east_lon, north_lat = bounding_box
            q = q.filter(
                cls.longitude >= west_lon).filter(
                cls.longitude < east_lon).filter(
                cls.latitude >= south_lat).filter(
                cls.latitude < north_lat)

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
        '''Delete all records older than the specified time window, optionally relative to a timestamp
        '''
        hours = hours if hours else 0
        days = days if days else 0
        weeks = weeks if weeks else 0

        if any([hours, days, weeks]):
            if timestamp is None:
                timestamp = datetime.utcnow().replace(tzinfo=pytz.UTC)

            ts_end = timestamp - timedelta(hours=hours, days=days, weeks=weeks)
            try:
                logger.info(f'Deleting tweets older than {ts_end}: {hours} hours {days} days {weeks} weeks')
                delete_q = cls.__table__.delete().where(cls.created_at < ts_end)
                session.execute(delete_q)
                session.commit()
            except Exception:
                logger.exception(f'Exception when deleting tweets older than {ts_end}: {hours} hours {days} days {weeks} weeks')
                session.rollback()

    @classmethod
    def keep_tweets_n_rows(cls, session, n: int = None):
        '''Keep the most recent n rows
        '''
        if n is not None:
            ids = session.query(cls.id).order_by(desc(cls.created_at)).all()
            ids_to_delete = [x[0] for x in ids[n:]]

            if ids_to_delete:
                try:
                    logger.info(f'Keeping most recent {n} rows of tweets')
                    delete_q = cls.__table__.delete().where(cls.id.in_(ids_to_delete))

                    session.execute(delete_q)
                    session.commit()
                except Exception:
                    logger.exception(f'Exception when keeping most recent {n} rows of tweets')
                    session.rollback()
