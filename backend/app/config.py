from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "sqlite:///./detection_history.db"

    # Fraction of a plate box's area that must fall inside a vehicle box
    # for the plate to be attributed to that vehicle
    plate_match_threshold: float = 0.5

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
 