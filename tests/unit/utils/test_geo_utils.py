import pytest
from unittest.mock import patch, MagicMock, mock_open
from src.utils.geo_utils import GeoMapper

@pytest.fixture
def mock_config():
    return {
        "mappings": {
            "location_targets": {"New York, NY": 1, "San Francisco, CA": 2}
        },
        "location_settings": {"max_distance_km": 50}
    }

def test_geo_mapper_init(mock_config):
    with patch('src.utils.geo_utils.get_config', return_value=mock_config), \
         patch('src.utils.geo_utils.Nominatim'), \
         patch('builtins.open', mock_open(read_data='{}')), \
         patch('src.utils.geo_utils.GeoMapper._init_targets'): # Skip target init for basic test
        
        mapper = GeoMapper()
        assert mapper.targets == mock_config["mappings"]["location_targets"]
        assert mapper.settings["max_distance_km"] == 50

def test_get_zone_cached(mock_config):
    with patch('src.utils.geo_utils.get_config', return_value=mock_config), \
         patch('src.utils.geo_utils.Nominatim'), \
         patch('builtins.open', mock_open(read_data='{}')), \
         patch('src.utils.geo_utils.GeoMapper._init_targets'):
        
        mapper = GeoMapper()
        # Pre-populate zone cache
        mapper.zone_cache["Test City"] = 3
        
        assert mapper.get_zone("Test City") == 3

def test_get_zone_proximity(mock_config):
    with patch('src.utils.geo_utils.get_config', return_value=mock_config), \
         patch('src.utils.geo_utils.Nominatim') as MockNominatim, \
         patch('builtins.open', mock_open(read_data='{}')), \
         patch('src.utils.geo_utils.GeoMapper._init_targets'): # Skip init, manually set targets
        
        mapper = GeoMapper()
        # Manually set target coords
        mapper.target_coords = {
            "New York, NY": (40.7128, -74.0060)
        }
        
        # Mock geocode for input city (e.g., Newark, close to NY)
        mock_geolocator = MockNominatim.return_value
        mock_location = MagicMock()
        mock_location.latitude = 40.7357
        mock_location.longitude = -74.1724
        mock_geolocator.geocode.return_value = mock_location
        
        # Newark is close to NY, so should be Zone 1
        zone = mapper.get_zone("Newark, NJ")
        assert zone == 1
        
        # Check if it was cached
        assert mapper.zone_cache["Newark, NJ"] == 1

def test_get_zone_far(mock_config):
    with patch('src.utils.geo_utils.get_config', return_value=mock_config), \
         patch('src.utils.geo_utils.Nominatim') as MockNominatim, \
         patch('builtins.open', mock_open(read_data='{}')), \
         patch('src.utils.geo_utils.GeoMapper._init_targets'):
        
        mapper = GeoMapper()
        mapper.target_coords = {
            "New York, NY": (40.7128, -74.0060)
        }
        
        # Mock geocode for far city (e.g., London)
        mock_geolocator = MockNominatim.return_value
        mock_location = MagicMock()
        mock_location.latitude = 51.5074
        mock_location.longitude = -0.1278
        mock_geolocator.geocode.return_value = mock_location
        
        # London is far from NY, should be default zone 4
        zone = mapper.get_zone("London, UK")
        assert zone == 4
