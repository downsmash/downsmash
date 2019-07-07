import enum
from glob import glob

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, ForeignKey, Enum, Numeric
from sqlalchemy.orm import relationship

Base = declarative_base()

icons_path = "assets/icons/"
icons_ext = ".png"
icons = glob(icons_path + "*" + icons_ext)
icons = [i[len(icons_path):-len(icons_ext)] for i in icons]
Characters = enum.Enum("Characters", sorted(icons))


class Set(Base):
    __tablename__ = "set"
    id = Column(Integer, primary_key=True)
    games = relationship("Game")


class Game(Base):
    __tablename__ = "game"
    id = Column(Integer, primary_key=True)
    set_id = Column(Integer, ForeignKey("set.id"))
    time_indices = relationship("TimeIndex")
    ports = relationship("Port")
    sources = relationship("GameSource")


class Source(Base):
    __tablename__ = "source"
    id = Column(Integer, primary_key=True)
    filename = Column(String)


class GameSource(Base):
    __tablename__ = "game_source"
    game_id = Column(Integer, ForeignKey("game.id"), primary_key=True)
    source_id = Column(Integer, ForeignKey("source.id"), primary_key=True)
    source = relationship("Source")


class Port(Base):
    __tablename__ = "port"
    game_id = Column(Integer, ForeignKey("game.id"), primary_key=True)
    port_number = Column(Integer, primary_key=True)
    portstates = relationship("PortState")


class PortState(Base):
    __tablename__ = "portstate"
    id = Column(Integer, primary_key=True)
    port_id = Column(Integer, ForeignKey("port.id"))

    char = Column(Enum(Characters))
    stocks = Column(Integer)
    pct = Column(Integer)


class TimeIndex(Base):
    __tablename__ = "timeindex"
    id = Column(Integer, primary_key=True)
    time = Column(Numeric)
    game_id = Column(Integer, ForeignKey("game.id"))
    port_states = Column(Integer, ForeignKey("portstate.id"))
