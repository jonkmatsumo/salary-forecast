"""Tests for GeoMapper in-memory caching."""

from unittest.mock import MagicMock, patch

import pytest

from src.utils.cache_manager import get_cache_manager
from src.utils.geo_utils import GeoMapper


@pytest.fixture
def mock_config() -> dict:
    """Create mock config for GeoMapper."""
    return {
        "mappings": {"location_targets": {"New York, NY": 1, "San Francisco, CA": 2}},
        "location_settings": {"max_distance_km": 50},
    }


class TestGeoMapperCaching:
    """Test suite for GeoMapper in-memory caching."""

    def test_coords_cached_in_memory(self, mock_config: dict) -> None:
        """Test coordinates are cached in memory."""
        with patch("src.utils.geo_utils.Nominatim") as mock_nominatim:
            mock_geolocator = MagicMock()
            mock_nominatim.return_value = mock_geolocator

            mock_location = MagicMock()
            mock_location.latitude = 40.7128
            mock_location.longitude = -74.0060
            mock_geolocator.geocode.return_value = mock_location

            cache_manager = get_cache_manager()
            cache_manager.clear("geo")

            mapper = GeoMapper(config=mock_config)
            # _init_targets calls _get_coords for each target city during initialization
            # "New York, NY" is in the target cities, so it gets called during init
            # Now call it again - should use cache
            coords1 = mapper._get_coords("New York, NY")
            # Reset mock to check if second call uses cache
            mock_geolocator.geocode.reset_mock()
            coords2 = mapper._get_coords("New York, NY")

            assert coords1 == (40.7128, -74.0060)
            assert coords2 == (40.7128, -74.0060)
            # Second call should not make API call due to cache
            assert mock_geolocator.geocode.call_count == 0

    def test_no_file_io_operations(self, mock_config: dict) -> None:
        """Test that no file I/O operations are performed."""
        with (
            patch("src.utils.geo_utils.Nominatim") as mock_nominatim,
            patch("builtins.open") as mock_open,
            patch("os.path.exists") as mock_exists,
            patch("os.makedirs") as mock_makedirs,
        ):
            mock_geolocator = MagicMock()
            mock_nominatim.return_value = mock_geolocator

            mock_location = MagicMock()
            mock_location.latitude = 40.7128
            mock_location.longitude = -74.0060
            mock_geolocator.geocode.return_value = mock_location

            mapper = GeoMapper(config=mock_config)
            mapper._get_coords("New York, NY")

            assert not mock_open.called
            assert not mock_exists.called
            assert not mock_makedirs.called

    def test_cache_hit_returns_cached_coords(self, mock_config: dict) -> None:
        """Test cache hit returns cached coordinates."""
        with patch("src.utils.geo_utils.Nominatim") as mock_nominatim:
            mock_geolocator = MagicMock()
            mock_nominatim.return_value = mock_geolocator

            cache_manager = get_cache_manager()
            # Pre-cache target cities to avoid geocoder calls during _init_targets()
            cache_manager.set("geo", "New York, NY", (40.7128, -74.0060))
            cache_manager.set("geo", "San Francisco, CA", (37.7749, -122.4194))
            # Cache the test city
            cache_manager.set("geo", "Test City", (45.0, -120.0))

            mapper = GeoMapper(config=mock_config)
            # Reset mock after initialization (which uses cached target cities)
            mock_geolocator.geocode.reset_mock()
            coords = mapper._get_coords("Test City")

            assert coords == (45.0, -120.0)
            assert not mock_geolocator.geocode.called

    def test_cache_miss_calls_geocoder(self, mock_config: dict) -> None:
        """Test cache miss calls geocoder and stores result."""
        with patch("src.utils.geo_utils.Nominatim") as mock_nominatim:
            mock_geolocator = MagicMock()
            mock_nominatim.return_value = mock_geolocator

            mock_location = MagicMock()
            mock_location.latitude = 37.7749
            mock_location.longitude = -122.4194
            mock_geolocator.geocode.return_value = mock_location

            cache_manager = get_cache_manager()
            cache_manager.clear("geo")

            mapper = GeoMapper(config=mock_config)
            coords = mapper._get_coords("San Francisco, CA")

            assert coords == (37.7749, -122.4194)
            assert mock_geolocator.geocode.called
            cached = cache_manager.get("geo", "San Francisco, CA")
            assert cached == (37.7749, -122.4194)

    def test_zone_cache_still_works(self, mock_config: dict) -> None:
        """Test zone cache still works as before."""
        with patch("src.utils.geo_utils.Nominatim") as mock_nominatim:
            mock_geolocator = MagicMock()
            mock_nominatim.return_value = mock_geolocator

            mock_location = MagicMock()
            mock_location.latitude = 40.7128
            mock_location.longitude = -74.0060
            mock_geolocator.geocode.return_value = mock_location

            mapper = GeoMapper(config=mock_config)
            zone1 = mapper.get_zone("New York, NY")
            zone2 = mapper.get_zone("New York, NY")

            assert zone1 == zone2
            assert "New York, NY" in mapper.zone_cache

    def test_multiple_cities_cached(self, mock_config: dict) -> None:
        """Test multiple cities are cached separately."""
        with patch("src.utils.geo_utils.Nominatim") as mock_nominatim:
            mock_geolocator = MagicMock()
            mock_nominatim.return_value = mock_geolocator

            def geocode_side_effect(city: str, **kwargs: dict) -> MagicMock:
                mock_loc = MagicMock()
                if city == "City1":
                    mock_loc.latitude = 10.0
                    mock_loc.longitude = 20.0
                elif city == "City2":
                    mock_loc.latitude = 30.0
                    mock_loc.longitude = 40.0
                else:
                    # For target cities in _init_targets
                    mock_loc.latitude = 0.0
                    mock_loc.longitude = 0.0
                return mock_loc

            mock_geolocator.geocode.side_effect = geocode_side_effect

            cache_manager = get_cache_manager()
            cache_manager.clear("geo")

            mapper = GeoMapper(config=mock_config)
            # _init_targets calls _get_coords for each target city
            initial_call_count = mock_geolocator.geocode.call_count
            coords1 = mapper._get_coords("City1")
            coords2 = mapper._get_coords("City2")
            coords1_cached = mapper._get_coords("City1")

            assert coords1 == (10.0, 20.0)
            assert coords2 == (30.0, 40.0)
            assert coords1_cached == (10.0, 20.0)
            # Should be called twice for City1 and City2 (after init_targets)
            assert mock_geolocator.geocode.call_count == initial_call_count + 2
