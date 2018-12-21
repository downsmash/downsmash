from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, ForeignKey
from sqlalchemy.orm import relationship

Base = declarative_base()


class Tournament(Base):
    __tablename__ = "tournament"
    id = Column(Integer, primary_key=True)
    events = relationship("Event")
    attendees = relationship("Attendee")


class Event(Base):
    __tablename__ = "event"
    id = Column(Integer, primary_key=True)
    tournament_id = Column(Integer, ForeignKey("tournament.id"))
    phases = relationship("Phase")


class Phase(Base):
    __tablename__ = "phase"
    id = Column(Integer, primary_key=True)
    event_id = Column(Integer, ForeignKey("event.id"))
    brackets = relationship("Bracket")


class Bracket(Base):
    __tablename__ = "bracket"
    id = Column(Integer, primary_key=True)
    phase_id = Column(Integer, ForeignKey("phase.id"))
    sets = relationship("Set")


class Set(Base):
    __tablename__ = "set"
    id = Column(Integer, primary_key=True)
    bracket_id = Column(Integer, ForeignKey("bracket.id"))
    games = relationship("Game")


class Game(Base):
    __tablename__ = "game"
    id = Column(Integer, primary_key=True)
    set_id = Column(Integer, ForeignKey("set.id"))
    time_indices = relationship("TimeIndex")


class PortState(Base):
    __tablename__ = "portstate"
    id = Column(Integer, primary_key=True)


class TimeIndex(Base):
    __tablename__ = "timeindex"
    id = Column(Integer, primary_key=True)
    game_id = Column(Integer, ForeignKey("game.id"))
    port_states = Column(Integer, ForeignKey("portstate.id"))


class Attendee(Base):
    __tablename__ = "attendee"

    player_id = Column(Integer, ForeignKey("player.id"), primary_key=True)
    tournament_id = Column(Integer, ForeignKey("tournament.id"), primary_key=True)
    player = relationship("Player")


class GameSource(Base):
    __tablename__ = "gamesource"

    game_id = Column(Integer, ForeignKey("game.id"), primary_key=True)
    source_id = Column(Integer, ForeignKey("source.id"), primary_key=True)
    game = relationship("Game")


class Source(Base):
    __tablename__ = "source"

    id = Column(Integer, primary_key=True)
    game_id = Column(Integer, ForeignKey("game.id"))
