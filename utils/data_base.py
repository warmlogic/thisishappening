from datetime import datetime, timedelta
import logging

import pytz
from sqlalchemy import create_engine, and_, desc, func, case
from sqlalchemy import Column, ForeignKey, String, Integer, DateTime, Float
from sqlalchemy.pool import NullPool
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

logger = logging.getLogger("happeninglogger")

Base = declarative_base()


def session_factory(DATABASE_URL: str, echo: bool = False):
    engine = create_engine(DATABASE_URL, poolclass=NullPool, echo=echo)
    Base.metadata.create_all(engine)
    _SessionFactory = sessionmaker(bind=engine)
    return _SessionFactory()


class Tiles(Base):
    '''To drop this table, run Tiles.metadata.drop_all(engine)
    '''
    __tablename__ = 'tiles'

    id = Column(Integer, primary_key=True)
    west_lon = Column(Float(precision=4), nullable=False)
    east_lon = Column(Float(precision=4), nullable=False)
    south_lat = Column(Float(precision=4), nullable=False)
    north_lat = Column(Float(precision=4), nullable=False)
    neighborhood = Column(String, nullable=True)
    city = Column(String, nullable=True)
    admin = Column(String, nullable=True)
    country = Column(String, nullable=True)
    recent_tweets = relationship('RecentTweets')
    historical_stats = relationship('HistoricalStats')
    events = relationship('Events')

    @classmethod
    def get_num_tiles(cls, session):
        return session.query(func.max(cls.id)).scalar()

    @classmethod
    def find_id_by_coords(cls, session, longitude, latitude):
        return session.query(cls).filter(
            longitude >= cls.west_lon).filter(
            longitude < cls.east_lon).filter(
            latitude >= cls.south_lat).filter(
            latitude < cls.north_lat).all()

    @classmethod
    def get_tile_center_lat_lon(cls, session, tile_id: int = None):
        q = session.query(
            cls.id,
            ((cls.north_lat + cls.south_lat) / 2).label('latitude'),
            ((cls.west_lon + cls.east_lon) / 2).label('longitude'),
        )

        if tile_id:
            q = q.filter(cls.id == tile_id)

        return q.group_by(cls.id).order_by(cls.id).all()

    @classmethod
    def get_tile_name(cls, session, tile_id: int = None):
        q = session.query(
            cls.id,
            case(
                [
                    (cls.neighborhood.isnot(None), cls.neighborhood),
                    (cls.city.isnot(None), cls.city),
                    (cls.admin.isnot(None), cls.admin),
                    (cls.country.isnot(None), cls.country),
                ],
                else_=None,
            )
        )

        if tile_id:
            q = q.filter(cls.id == tile_id)

        return q.order_by(cls.id).all()

    def __repr__(self):
        return f'Tile {self.id}'


class Events(Base):
    '''To drop this table, run Events.metadata.drop_all(engine)
    '''
    __tablename__ = 'events'

    id = Column(Integer, primary_key=True)
    tile_id = Column(Integer, ForeignKey('tiles.id'), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    count = Column(Integer, nullable=False)
    longitude = Column(Float(precision=8), nullable=False)
    latitude = Column(Float(precision=8), nullable=False)
    place_name = Column(String, nullable=True)
    description = Column(String, nullable=False)
    tile = relationship('Tiles')

    @classmethod
    def get_event_tweets(cls, session, event_id: int, hours: float = 1):

        event = session.query(cls).filter(cls.id == event_id).order_by(desc(cls.timestamp)).first()
        timestamp = event.timestamp.replace(tzinfo=pytz.UTC)
        tile_id = event.tile_id

        return RecentTweets.get_recent_tweets(session, timestamp=timestamp, hours=hours, tile_id=tile_id)

    def __repr__(self):
        return f'Event {self.id}'


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
    tile_id = Column(Integer, ForeignKey('tiles.id'), nullable=False)
    tile = relationship('Tiles')

    @classmethod
    def count_tweets_per_tile(cls, session, timestamp: datetime = None, hours: float = 0, tile_id: int = None):
        if timestamp is None:
            timestamp = datetime.utcnow().replace(tzinfo=pytz.UTC)

        filter_td = timestamp - timedelta(hours=hours)

        q = session.query(
            cls.tile_id, func.count(cls.status_id_str)).filter(
                cls.created_at >= filter_td).filter(cls.created_at <= timestamp)

        if tile_id:
            q = q.filter(cls.tile_id == tile_id)

        return q.group_by(cls.tile_id).order_by(cls.tile_id).all()

    @classmethod
    def get_recent_tweets(cls, session, timestamp: datetime = None, hours: float = 1, tile_id: int = None):
        if timestamp is None:
            timestamp = datetime.utcnow().replace(tzinfo=pytz.UTC)

        filter_td = timestamp - timedelta(hours=hours)

        q = session.query(cls).filter(cls.created_at >= filter_td).filter(cls.created_at <= timestamp)

        if tile_id:
            q = q.filter(cls.tile_id == tile_id)

        return q.order_by(desc(cls.created_at)).all()

    @classmethod
    def get_oldest_tweet(cls, session, tile_id: int = None):
        q = session.query(cls)

        if tile_id:
            q = q.filter(cls.tile_id == tile_id)

        return q.order_by(cls.created_at).first()

    @classmethod
    def get_most_recent_tweet(cls, session, tile_id: int = None):
        q = session.query(cls)

        if tile_id:
            q = q.filter(cls.tile_id == tile_id)

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

            filter_td = timestamp - timedelta(hours=hours, days=days, weeks=weeks)
            try:
                logger.info(f'Deleting tweets older than {filter_td}: {hours} hours {days} days {weeks} weeks')
                delete_q = cls.__table__.delete().where(
                    cls.created_at < filter_td)

                session.execute(delete_q)
                session.commit()
            except Exception:
                logger.exception(f'Exception when deleting tweets older than {filter_td}: {hours} hours {days} days {weeks} weeks')
                session.rollback()


class HistoricalStats(Base):
    '''To drop this table, run HistoricalStats.metadata.drop_all(engine)
    '''
    __tablename__ = 'historical_stats'

    id = Column(Integer, primary_key=True)
    tile_id = Column(Integer, ForeignKey('tiles.id'), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    count = Column(Integer, nullable=False)
    mean = Column(Float, nullable=False)
    variance = Column(Float, nullable=False)
    stddev = Column(Float, nullable=False)
    tile = relationship('Tiles')

    @classmethod
    def get_recent_stats(
        cls,
        session,
        timestamp: datetime = None,
        hours: float = 0,
        days: float = 0,
        weeks: float = 0,
    ):
        """Get the most recent stats for each tile.
        Can filter to only consider rows prior to an earlier point in time.
        Set all time values to 0 to return the most recent row per tile.
        returns a tuple (tile_id, row)
        """
        if timestamp is None:
            timestamp = datetime.utcnow().replace(tzinfo=pytz.UTC)

        filter_td = timestamp - timedelta(days=days, hours=hours, weeks=weeks)

        subq = session.query(
            cls.tile_id, func.max(cls.timestamp).label("maxtimestamp")).filter(
                cls.timestamp < filter_td).group_by(
                    cls.tile_id).subquery()
        return session.query(cls.tile_id, cls).join(
            subq, and_(cls.tile_id == subq.c.tile_id, cls.timestamp == subq.c.maxtimestamp)).order_by(cls.tile_id).all()

    @classmethod
    def delete_stats_older_than(
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

            filter_td = timestamp - timedelta(hours=hours, days=days, weeks=weeks)
            try:
                logger.info(f'Deleting historical stats older than {filter_td}: {hours} hours {days} days {weeks} weeks')
                delete_q = cls.__table__.delete().where(
                    cls.timestamp < filter_td)

                session.execute(delete_q)
                session.commit()
            except Exception:
                logger.exception(f'Exception when deleting historical stats older than {filter_td}: {hours} hours {days} days {weeks} weeks')
                session.rollback()
