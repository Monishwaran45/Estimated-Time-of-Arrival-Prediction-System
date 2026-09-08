import os
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

logger = logging.getLogger("eta.database")
logging.basicConfig(level=logging.INFO)

# Configurable environment variables with defaults provided by user
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "root")
MYSQL_DB = os.getenv("MYSQL_DB", "eta_db")

# Defensive imports
try:
    import pymysql
    PYMYSQL_AVAILABLE = True
except ImportError:
    PYMYSQL_AVAILABLE = False
    logger.warning("PyMySQL is not installed. Will use fallback database mode.")

from sqlalchemy import (
    create_engine, Column, Integer, Float, String, Text, DateTime
)
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session

Base = declarative_base()

class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String(50), index=True, nullable=False)
    distance_km = Column(Float, nullable=False)
    weather = Column(String(50), nullable=False)
    traffic_level = Column(String(50), nullable=False)
    time_of_day = Column(String(50), nullable=False)
    vehicle_type = Column(String(50), nullable=False)
    preparation_time_min = Column(Float, nullable=False)
    courier_experience_yrs = Column(Float, nullable=False)
    predicted_time_min = Column(Float, nullable=False)
    formatted_eta = Column(String(50), nullable=False)
    lower_sla_min = Column(Float, nullable=False)
    upper_sla_min = Column(Float, nullable=False)
    risk_level = Column(String(50), nullable=False)
    risk_color = Column(String(20), nullable=False)
    risk_score = Column(Integer, nullable=False)
    kitchen_prep_min = Column(Float, nullable=True)
    base_transit_min = Column(Float, nullable=True)
    traffic_delay_min = Column(Float, nullable=True)
    weather_delay_min = Column(Float, nullable=True)
    courier_bonus_min = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.order_id,
            "db_id": self.id,
            "distance": self.distance_km,
            "weather": self.weather,
            "traffic": self.traffic_level,
            "time_of_day": self.time_of_day,
            "vehicle": self.vehicle_type,
            "prep_time": self.preparation_time_min,
            "experience": self.courier_experience_yrs,
            "predicted_eta": self.predicted_time_min,
            "formatted_eta": self.formatted_eta,
            "lower_sla": self.lower_sla_min,
            "upper_sla": self.upper_sla_min,
            "risk_level": self.risk_level,
            "risk_color": self.risk_color,
            "risk_score": self.risk_score,
            "breakdown": {
                "kitchen_prep_min": self.kitchen_prep_min,
                "base_transit_min": self.base_transit_min,
                "traffic_delay_min": self.traffic_delay_min,
                "weather_delay_min": self.weather_delay_min,
                "courier_tenure_bonus_min": self.courier_bonus_min
            },
            "time": self.created_at.strftime("%H:%M:%S") if self.created_at else ""
        }

class PresetScenario(Base):
    __tablename__ = "presets"

    id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=False)
    distance_km = Column(Float, nullable=False)
    weather = Column(String(50), nullable=False)
    traffic_level = Column(String(50), nullable=False)
    time_of_day = Column(String(50), nullable=False)
    vehicle_type = Column(String(50), nullable=False)
    preparation_time_min = Column(Float, nullable=False)
    courier_experience_yrs = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "data": {
                "Distance_km": self.distance_km,
                "Weather": self.weather,
                "Traffic_Level": self.traffic_level,
                "Time_of_Day": self.time_of_day,
                "Vehicle_Type": self.vehicle_type,
                "Preparation_Time_min": self.preparation_time_min,
                "Courier_Experience_yrs": self.courier_experience_yrs
            }
        }

class ModelMetricRecord(Base):
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False)
    test_mae = Column(Float, nullable=False)
    test_rmse = Column(Float, nullable=False)
    test_r2 = Column(Float, nullable=False)
    cv_r2 = Column(String(50), nullable=True)
    dataset_rows = Column(Integer, nullable=False)
    calculated_at = Column(DateTime, default=datetime.utcnow)

class DatabaseManager:
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self.is_connected = False
        self.db_type = "none"
        self.init_db()

    def init_db(self):
        # First attempt connecting to MySQL
        if PYMYSQL_AVAILABLE:
            try:
                conn = pymysql.connect(
                    host=MYSQL_HOST,
                    port=MYSQL_PORT,
                    user=MYSQL_USER,
                    password=MYSQL_PASSWORD,
                    charset="utf8mb4"
                )
                with conn.cursor() as cursor:
                    cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DB}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
                conn.commit()
                conn.close()
                logger.info(f"MySQL database '{MYSQL_DB}' ensured on {MYSQL_HOST}:{MYSQL_PORT}.")

                mysql_url = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?charset=utf8mb4"
                self.engine = create_engine(mysql_url, pool_recycle=3600, pool_pre_ping=True)
                Base.metadata.create_all(bind=self.engine)
                self.SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=self.engine))
                self.is_connected = True
                self.db_type = "MySQL"
                logger.info(f"Connected to MySQL database '{MYSQL_DB}' successfully.")
                self._seed_presets_if_empty()
                return
            except Exception as e:
                logger.warning(f"MySQL connection attempt failed: {e}. Falling back to SQLite.")

        # Graceful SQLite fallback
        try:
            sqlite_url = f"sqlite:///{os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eta_local.db')}"
            self.engine = create_engine(sqlite_url, connect_args={"check_same_thread": False})
            Base.metadata.create_all(bind=self.engine)
            self.SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=self.engine))
            self.is_connected = True
            self.db_type = "SQLite (Local Fallback)"
            logger.info("Using SQLite fallback storage.")
            self._seed_presets_if_empty()
        except Exception as e:
            logger.error(f"Failed to initialize database engine: {e}")

    def _seed_presets_if_empty(self):
        session = self.get_session()
        try:
            count = session.query(PresetScenario).count()
            if count == 0:
                defaults = [
                    PresetScenario(
                        id="quick-coffee",
                        name="☕ Morning Coffee Run",
                        description="Short distance, clear skies, low traffic on an electric scooter",
                        distance_km=2.5,
                        weather="Clear",
                        traffic_level="Low",
                        time_of_day="Morning",
                        vehicle_type="Scooter",
                        preparation_time_min=7.0,
                        courier_experience_yrs=4.5
                    ),
                    PresetScenario(
                        id="rainy-dinner-rush",
                        name="🌧️ Monsoon Dinner Rush",
                        description="Heavy rain, high traffic congestion during peak evening dinner hours",
                        distance_km=9.8,
                        weather="Rainy",
                        traffic_level="High",
                        time_of_day="Evening",
                        vehicle_type="Bike",
                        preparation_time_min=25.0,
                        courier_experience_yrs=1.5
                    ),
                    PresetScenario(
                        id="suburban-night-drive",
                        name="🚗 Midnight Long-Range Express",
                        description="Long highway distance, clear night weather by car with veteran courier",
                        distance_km=18.2,
                        weather="Clear",
                        traffic_level="Low",
                        time_of_day="Night",
                        vehicle_type="Car",
                        preparation_time_min=14.0,
                        courier_experience_yrs=8.0
                    ),
                    PresetScenario(
                        id="snowy-lunch-bottleneck",
                        name="❄️ Winter Storm Lunch Bottleneck",
                        description="Snowy roads, medium traffic, kitchen backlogged during afternoon peak",
                        distance_km=6.4,
                        weather="Snowy",
                        traffic_level="Medium",
                        time_of_day="Afternoon",
                        vehicle_type="Car",
                        preparation_time_min=32.0,
                        courier_experience_yrs=2.0
                    )
                ]
                session.add_all(defaults)
                session.commit()
                logger.info("Default presets seeded into database.")
        except Exception as e:
            session.rollback()
            logger.error(f"Error seeding presets: {e}")
        finally:
            session.close()

    def get_session(self):
        if not self.SessionLocal:
            self.init_db()
        return self.SessionLocal()

    def save_prediction(self, record_data: dict) -> PredictionRecord:
        session = self.get_session()
        try:
            breakdown = record_data.get("breakdown", {})
            rec = PredictionRecord(
                order_id=record_data["order_id"],
                distance_km=record_data["distance_km"],
                weather=record_data["weather"],
                traffic_level=record_data["traffic_level"],
                time_of_day=record_data["time_of_day"],
                vehicle_type=record_data["vehicle_type"],
                preparation_time_min=record_data["preparation_time_min"],
                courier_experience_yrs=record_data["courier_experience_yrs"],
                predicted_time_min=record_data["predicted_time_min"],
                formatted_eta=record_data["formatted_eta"],
                lower_sla_min=record_data["lower_sla_min"],
                upper_sla_min=record_data["upper_sla_min"],
                risk_level=record_data["risk_level"],
                risk_color=record_data["risk_color"],
                risk_score=record_data["risk_score"],
                kitchen_prep_min=breakdown.get("kitchen_prep_min"),
                base_transit_min=breakdown.get("base_transit_min"),
                traffic_delay_min=breakdown.get("traffic_delay_min"),
                weather_delay_min=breakdown.get("weather_delay_min"),
                courier_bonus_min=breakdown.get("courier_tenure_bonus_min")
            )
            session.add(rec)
            session.commit()
            session.refresh(rec)
            return rec
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving prediction to database: {e}")
            raise e
        finally:
            session.close()

    def get_recent_predictions(self, limit: int = 50) -> List[dict]:
        session = self.get_session()
        try:
            records = session.query(PredictionRecord).order_by(PredictionRecord.id.desc()).limit(limit).all()
            return [r.to_dict() for r in records]
        finally:
            session.close()

    def get_all_presets(self) -> List[dict]:
        session = self.get_session()
        try:
            presets = session.query(PresetScenario).all()
            return [p.to_dict() for p in presets]
        finally:
            session.close()

    def get_status(self) -> dict:
        session = self.get_session()
        try:
            pred_count = session.query(PredictionRecord).count()
            preset_count = session.query(PresetScenario).count()
            return {
                "status": "connected" if self.is_connected else "disconnected",
                "database_type": self.db_type,
                "host": MYSQL_HOST if "MySQL" in self.db_type else "Local File",
                "database_name": MYSQL_DB if "MySQL" in self.db_type else "eta_local.db",
                "user": MYSQL_USER if "MySQL" in self.db_type else "N/A",
                "total_predictions_stored": pred_count,
                "total_presets": preset_count
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "database_type": self.db_type
            }
        finally:
            session.close()

# Singleton instance
db_manager = DatabaseManager()
