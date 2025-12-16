import time
from typing import Any, Dict, Optional, Tuple, cast

from geopy.distance import geodesic
from geopy.geocoders import Nominatim

from src.utils.cache_manager import get_cache_manager


class GeoMapper:
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize GeoMapper.

        Args:
            config (Optional[Dict[str, Any]]): Configuration dictionary. If None, uses defaults.
        """
        if config is None:
            config = {
                "mappings": {"location_targets": {}},
                "location_settings": {"max_distance_km": 50},
            }
        self.config: Dict[str, Any] = config
        self.targets: Dict[str, int] = self.config.get("mappings", {}).get("location_targets", {})
        self.settings: Dict[str, Any] = self.config.get(
            "location_settings", {"max_distance_km": 50}
        )

        self.cache_manager = get_cache_manager()
        self._init_geolocator()

        self.zone_cache: Dict[str, int] = {}
        self.target_coords: Dict[str, Tuple[float, float]] = {}
        self._init_targets()

    def _init_geolocator(self) -> None:
        """Initialize geocoder instance."""
        self.geolocator = Nominatim(user_agent="salary_forecast_app_v1")

    def __getstate__(self) -> Dict[str, Any]:
        """Get state for pickling.

        Returns:
            Dict[str, Any]: State dictionary without geolocator.
        """
        state = self.__dict__.copy()
        if "geolocator" in state:
            del state["geolocator"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore state from pickling.

        Args:
            state (Dict[str, Any]): State dictionary.
        """
        self.__dict__.update(state)
        self._init_geolocator()

    def _get_coords(self, city: str) -> Optional[Tuple[float, float]]:
        """Get coordinates for a city using in-memory cache.

        Args:
            city (str): City name.

        Returns:
            Optional[Tuple[float, float]]: (latitude, longitude) or None.
        """
        cached_coords = self.cache_manager.get("geo", city)
        if cached_coords is not None:
            return cast(Tuple[float, float], cached_coords)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                location = self.geolocator.geocode(city, timeout=10)
                if location:
                    coords = (location.latitude, location.longitude)
                    self.cache_manager.set("geo", city, coords)
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
        """Initialize target cities."""
        print("Initializing target cities...")
        for city, zone in self.targets.items():
            coords = self._get_coords(city)
            if coords:
                self.target_coords[city] = coords
            else:
                print(f"Warning: Could not geocode target city {city}")

    def get_zone(self, input_city: Any) -> int:
        """Determine the cost zone for a given city based on proximity to targets.

        Args:
            input_city (Any): City name.

        Returns:
            int: Cost zone (defaults to 4).
        """
        if not isinstance(input_city, str):
            return 4

        if input_city in self.zone_cache:
            return self.zone_cache[input_city]

        input_coords = self._get_coords(input_city)
        if not input_coords:
            self.zone_cache[input_city] = 4
            return 4

        nearest_city = None
        min_dist = float("inf")

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
