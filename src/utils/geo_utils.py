import json
import os
import time
from typing import Optional, Dict, Tuple, Any
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from .config_loader import get_config

class GeoMapper:
    def __init__(self) -> None:
        self.config: Dict[str, Any] = get_config()
        self.targets: Dict[str, int] = self.config["mappings"]["location_targets"]
        self.settings: Dict[str, Any] = self.config.get("location_settings", {"max_distance_km": 50})
        
        env_path = os.environ.get("SALARY_CACHE_FILE")
        if env_path:
            self.cache_file = os.path.abspath(os.path.expanduser(env_path))
        else:
            home_dir = os.path.expanduser("~")
            app_dir = os.path.join(home_dir, ".salary_forecast")
            self.cache_file = os.path.join(app_dir, "city_cache.json")
            
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        
        local_cache = "city_cache.json"
        if os.path.exists(local_cache) and not os.path.exists(self.cache_file):
            print(f"Migrating local cache from {local_cache} to {self.cache_file}...")
            try:
                with open(local_cache, "r") as src, open(self.cache_file, "w") as dst:
                    dst.write(src.read())
            except Exception as e:
                print(f"Failed to migrate cache: {e}")

        self.cache: Dict[str, Tuple[float, float]] = self._load_cache()
        self._init_geolocator()
        
        self.zone_cache: Dict[str, int] = {}
        self.target_coords: Dict[str, Tuple[float, float]] = {}
        self._init_targets()

    def _init_geolocator(self) -> None:
        self.geolocator = Nominatim(user_agent="salary_forecast_app_v1")

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        if 'geolocator' in state:
            del state['geolocator']
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._init_geolocator()

    def _load_cache(self) -> Dict[str, Tuple[float, float]]:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    data = json.load(f)
                    return {k: tuple(v) for k, v in data.items()}
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_cache(self) -> None:
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f, indent=4)

    def _get_coords(self, city: str) -> Optional[Tuple[float, float]]:
        """Get coordinates for a city. Args: city (str): City name. Returns: Optional[Tuple[float, float]]: (latitude, longitude) or None."""
        if city in self.cache:
            return tuple(self.cache[city])
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                location = self.geolocator.geocode(city, timeout=10)
                if location:
                    coords = (location.latitude, location.longitude)
                    self.cache[city] = coords
                    self._save_cache()
                    time.sleep(1)
                    return coords
                else:
                    print(f"City not found: {city}")
                    return None
            except Exception as e:
                print(f"Error geocoding {city} (Attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(2 * (attempt + 1))
            
        return None

    def _init_targets(self) -> None:
        """Initialize target cities. Returns: None."""
        print("Initializing target cities...")
        for city, zone in self.targets.items():
            coords = self._get_coords(city)
            if coords:
                self.target_coords[city] = coords
            else:
                print(f"Warning: Could not geocode target city {city}")

    def get_zone(self, input_city: Any) -> int:
        """Determine the cost zone for a given city based on proximity to targets. Args: input_city (Any): City name. Returns: int: Cost zone (defaults to 4)."""
        if not isinstance(input_city, str):
            return 4
            
        if input_city in self.zone_cache:
            return self.zone_cache[input_city]
            
        input_coords = self._get_coords(input_city)
        if not input_coords:
            self.zone_cache[input_city] = 4
            return 4
            
        nearest_city = None
        min_dist = float('inf')
        
        for target_city, target_coords in self.target_coords.items():
            dist = geodesic(input_coords, target_coords).kilometers
            if dist < min_dist:
                min_dist = dist
                nearest_city = target_city
                
        zone = 4
        if nearest_city and min_dist <= self.settings["max_distance_km"]:
            zone = self.targets[nearest_city]
            
        self.zone_cache[input_city] = zone
        return zone
