import json
import os
import time
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from .config_loader import get_config

class GeoMapper:
    def __init__(self):
        self.config = get_config()
        self.targets = self.config["mappings"]["location_targets"]
        self.settings = self.config.get("location_settings", {"max_distance_km": 50})
        self.cache_file = "city_cache.json"
        self.cache = self._load_cache()
        self._init_geolocator()
        
        # O(1) cache for zone lookups
        self.zone_cache = {}
        
        # Pre-fetch target coordinates if not in cache
        self.target_coords = {}
        self._init_targets()

    def _init_geolocator(self):
        self.geolocator = Nominatim(user_agent="salary_forecast_app_v1")

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove unpicklable entries.
        if 'geolocator' in state:
            del state['geolocator']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Restore unpicklable entries.
        self._init_geolocator()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_cache(self):
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f, indent=4)

    def _get_coords(self, city):
        # Check cache first
        if city in self.cache:
            return tuple(self.cache[city])
        
        # Geocode with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                location = self.geolocator.geocode(city, timeout=10) # Increased timeout
                if location:
                    coords = (location.latitude, location.longitude)
                    self.cache[city] = coords
                    self._save_cache()
                    time.sleep(1) # Respect rate limits
                    return coords
                else:
                    # If location is None, it's not a timeout, just not found.
                    print(f"City not found: {city}")
                    return None
            except Exception as e:
                print(f"Error geocoding {city} (Attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(2 * (attempt + 1)) # Backoff: 2s, 4s, 6s
            
        return None

    def _init_targets(self):
        print("Initializing target cities...")
        for city, zone in self.targets.items():
            coords = self._get_coords(city)
            if coords:
                self.target_coords[city] = coords
            else:
                print(f"Warning: Could not geocode target city {city}")

    def get_zone(self, input_city):
        if not isinstance(input_city, str):
            return 4 # Default zone
            
        # Check in-memory zone cache first (O(1))
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
            
        # Cache the result
        self.zone_cache[input_city] = zone
        return zone
