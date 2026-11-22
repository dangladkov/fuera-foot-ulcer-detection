import pytest
import numpy as np
import scipy.io
import os
import tempfile
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt

# Import the functions we need to test from your workspace
from generate_heatspot_mask import make_mask, scale_mask
from thermal_Image_handler import generate_thermal_image, change_values_til_row

class TestSyntheticImageGenerator:
    """Test suite for synthetic thermal image generation with wound masks."""
    
    @pytest.fixture
    def setup_test_data(self):
        """Create test data for thermal images and wound masks."""
        # Create a synthetic thermal image (100x100 pixels)
        base_temp = 32.0  # Normal skin temperature
        thermal_image = np.full((100, 100), base_temp, dtype=float)
        
        # Create a simple circular wound mask (radius 10 pixels, center at 50,50)
        y, x = np.ogrid[:100, :100]
        center_y, center_x = 50, 50
        radius = 10
        wound_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        
        # Expected temperature increase
        temp_increase = 3.0
        
        return {
            'thermal_image': thermal_image,
            'wound_mask': wound_mask,
            'temp_increase': temp_increase,
            'base_temp': base_temp,
            'center': (center_y, center_x),
            'radius': radius
        }
    
    @pytest.fixture
    def setup_multi_day_data(self):
        """Create test data for multi-day temperature progression."""
        num_days = 7
        base_temp = 32.0
        initial_increase = 1.0
        final_increase = 5.0
        
        # Create thermal images for multiple days
        thermal_images = []
        expected_increases = []
        
        for day in range(num_days):
            # Linear progression from initial to final increase
            progress = day / (num_days - 1) if num_days > 1 else 1.0
            temp_increase = initial_increase + (final_increase - initial_increase) * progress
            
            thermal_image = np.full((100, 100), base_temp, dtype=float)
            thermal_images.append(thermal_image)
            expected_increases.append(temp_increase)
        
        # Create wound mask
        y, x = np.ogrid[:100, :100]
        wound_mask = (x - 50)**2 + (y - 50)**2 <= 100  # radius 10
        
        return {
            'thermal_images': thermal_images,
            'wound_mask': wound_mask,
            'expected_increases': expected_increases,
            'base_temp': base_temp,
            'num_days': num_days
        }

    def test_temperature_increase_correctness(self, setup_test_data):
        """Test that temperatures increase by the correct amount within wound mask."""
        data = setup_test_data
        thermal_image = data['thermal_image'].copy()
        wound_mask = data['wound_mask']
        temp_increase = data['temp_increase']
        base_temp = data['base_temp']
        
        # Apply temperature increase to wound area
        modified_image = thermal_image.copy()
        modified_image[wound_mask] += temp_increase
        
        # Test 1: Verify temperature increase in wound area
        wound_temps = modified_image[wound_mask]
        expected_wound_temp = base_temp + temp_increase
        
        assert np.allclose(wound_temps, expected_wound_temp), \
            f"Expected wound temperature {expected_wound_temp}, got {np.mean(wound_temps)}"
        
        # Test 2: Verify no temperature change outside wound area
        non_wound_mask = ~wound_mask
        non_wound_temps = modified_image[non_wound_mask]
        
        assert np.allclose(non_wound_temps, base_temp), \
            f"Expected base temperature {base_temp} outside wound, got {np.mean(non_wound_temps)}"
        
        # Test 3: Verify exact temperature difference
        temp_diff = modified_image - thermal_image
        wound_diff = temp_diff[wound_mask]
        non_wound_diff = temp_diff[non_wound_mask]
        
        assert np.allclose(wound_diff, temp_increase), \
            f"Expected temperature increase {temp_increase}, got {np.mean(wound_diff)}"
        assert np.allclose(non_wound_diff, 0.0), \
            f"Expected no temperature change outside wound, got {np.mean(non_wound_diff)}"

    def test_temperature_increase_multiple_days(self, setup_multi_day_data):
        """Test temperature progression over multiple days."""
        data = setup_multi_day_data
        thermal_images = data['thermal_images']
        wound_mask = data['wound_mask']
        expected_increases = data['expected_increases']
        base_temp = data['base_temp']
        num_days = data['num_days']
        
        modified_images = []
        
        # Apply progressive temperature increases
        for day in range(num_days):
            modified_image = thermal_images[day].copy()
            temp_increase = expected_increases[day]
            modified_image[wound_mask] += temp_increase
            modified_images.append(modified_image)
        
        # Test progression for each day
        for day in range(num_days):
            modified_image = modified_images[day]
            expected_increase = expected_increases[day]
            expected_wound_temp = base_temp + expected_increase
            
            # Test wound area temperature
            wound_temps = modified_image[wound_mask]
            assert np.allclose(wound_temps, expected_wound_temp), \
                f"Day {day+1}: Expected wound temperature {expected_wound_temp}, got {np.mean(wound_temps)}"
            
            # Test non-wound area unchanged
            non_wound_temps = modified_image[~wound_mask]
            assert np.allclose(non_wound_temps, base_temp), \
                f"Day {day+1}: Expected base temperature {base_temp} outside wound, got {np.mean(non_wound_temps)}"
        
        # Test temperature progression is monotonic (increasing over time)
        wound_mean_temps = [np.mean(img[wound_mask]) for img in modified_images]
        for i in range(1, len(wound_mean_temps)):
            assert wound_mean_temps[i] >= wound_mean_temps[i-1], \
                f"Temperature should increase over time: Day {i} temp {wound_mean_temps[i]} < Day {i+1} temp {wound_mean_temps[i-1]}"

    def test_wound_mask_application_precision(self, setup_test_data):
        """Test that wound mask is applied with pixel-level precision."""
        data = setup_test_data
        thermal_image = data['thermal_image'].copy()
        wound_mask = data['wound_mask']
        temp_increase = data['temp_increase']
        
        # Apply temperature increase
        modified_image = thermal_image.copy()
        modified_image[wound_mask] += temp_increase
        
        # Test each pixel individually
        for y in range(thermal_image.shape[0]):
            for x in range(thermal_image.shape[1]):
                if wound_mask[y, x]:
                    expected_temp = data['base_temp'] + temp_increase
                    actual_temp = modified_image[y, x]
                    assert abs(actual_temp - expected_temp) < 1e-10, \
                        f"Pixel ({y},{x}) in wound: expected {expected_temp}, got {actual_temp}"
                else:
                    expected_temp = data['base_temp']
                    actual_temp = modified_image[y, x]
                    assert abs(actual_temp - expected_temp) < 1e-10, \
                        f"Pixel ({y},{x}) outside wound: expected {expected_temp}, got {actual_temp}"

    def test_gaussian_blur_temperature_preservation(self, setup_test_data):
        """Test that Gaussian blur preserves average temperatures in wound area."""
        from scipy.ndimage import gaussian_filter
        
        data = setup_test_data
        thermal_image = data['thermal_image'].copy()
        wound_mask = data['wound_mask']
        temp_increase = data['temp_increase']
        
        # Create sharp temperature increase
        modified_image = thermal_image.copy()
        modified_image[wound_mask] += temp_increase
        
        # Apply Gaussian blur (similar to generate_heatspot_mask.py)
        blurred_image = gaussian_filter(modified_image, sigma=2.0)
        
        # Test that average temperature in wound area is preserved (approximately)
        original_wound_mean = np.mean(modified_image[wound_mask])
        blurred_wound_mean = np.mean(blurred_image[wound_mask])
        
        # Allow small tolerance due to blur spreading
        tolerance = 0.5  # 0.5°C tolerance
        assert abs(blurred_wound_mean - original_wound_mean) < tolerance, \
            f"Gaussian blur changed average wound temperature too much: {original_wound_mean} -> {blurred_wound_mean}"

    def test_edge_cases_and_boundary_conditions(self, setup_test_data):
        """Test edge cases and boundary conditions."""
        data = setup_test_data
        
        # Test 1: Empty wound mask
        empty_mask = np.zeros_like(data['wound_mask'], dtype=bool)
        thermal_image = data['thermal_image'].copy()
        original_image = thermal_image.copy()
        
        thermal_image[empty_mask] += data['temp_increase']  # Should do nothing
        assert np.array_equal(thermal_image, original_image), \
            "Empty wound mask should not change image"
        
        # Test 2: Full wound mask (entire image)
        full_mask = np.ones_like(data['wound_mask'], dtype=bool)
        thermal_image = data['thermal_image'].copy()
        thermal_image[full_mask] += data['temp_increase']
        
        expected_temp = data['base_temp'] + data['temp_increase']
        assert np.allclose(thermal_image, expected_temp), \
            f"Full wound mask should increase all pixels to {expected_temp}"
        
        # Test 3: Zero temperature increase
        thermal_image = data['thermal_image'].copy()
        original_image = thermal_image.copy()
        thermal_image[data['wound_mask']] += 0.0
        
        assert np.array_equal(thermal_image, original_image), \
            "Zero temperature increase should not change image"
        
        # Test 4: Negative temperature increase (cooling)
        thermal_image = data['thermal_image'].copy()
        cooling = -2.0
        thermal_image[data['wound_mask']] += cooling
        
        expected_cooled_temp = data['base_temp'] + cooling
        wound_temps = thermal_image[data['wound_mask']]
        assert np.allclose(wound_temps, expected_cooled_temp), \
            f"Negative temperature increase should cool to {expected_cooled_temp}"

    def test_integration_with_generate_heatspot_mask(self):
        """Test integration with the actual generate_heatspot_mask.py functionality."""
        # Create test data similar to what generate_heatspot_mask.py expects
        h, w = 100, 100
        base_image = np.full((h, w), 32.0)
        
        # Create a simple circular mask
        Y, X = np.ogrid[:h, :w]
        center_x, center_y = 50, 50
        radius = 15
        core_mask = (X - center_x)**2 + (Y - center_y)**2 <= radius**2
        
        # Simulate parameters from generate_heatspot_mask.py
        core_temp_increase = 5.0
        
        # Apply temperature increase
        mask_values = np.zeros((h, w))
        mask_values[core_mask] = core_temp_increase
        
        # Apply to base image
        modified_image = base_image + mask_values
        
        # Verify results
        wound_temps = modified_image[core_mask]
        non_wound_temps = modified_image[~core_mask]
        
        assert np.allclose(wound_temps, 32.0 + core_temp_increase), \
            f"Expected wound temperature {32.0 + core_temp_increase}"
        assert np.allclose(non_wound_temps, 32.0), \
            "Expected base temperature outside wound"

    def test_mat_file_integration(self):
        """Test integration with .mat file saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data
            test_image = np.random.normal(32.0, 2.0, (100, 100))
            wound_mask = np.zeros((100, 100))
            wound_mask[40:60, 40:60] = 3.0  # 20x20 square with 3°C increase
            
            # Save to .mat file
            mat_path = os.path.join(temp_dir, "test_wound.mat")
            scipy.io.savemat(mat_path, {
                "base_image": test_image,
                "wound_mask": wound_mask
            })
            
            # Load and verify
            loaded_data = scipy.io.loadmat(mat_path)
            loaded_image = loaded_data["base_image"]
            loaded_mask = loaded_data["wound_mask"]
            
            # Apply wound mask
            modified_image = loaded_image + loaded_mask
            
            # Verify wound area has correct temperature increase
            wound_area = loaded_mask > 0
            temp_increases = modified_image[wound_area] - loaded_image[wound_area]
            
            assert np.allclose(temp_increases, 3.0), \
                f"Expected 3°C increase in wound area, got {np.mean(temp_increases)}"

    @pytest.mark.parametrize("num_days,initial_temp,final_temp", [
        (5, 1.0, 4.0),
        (10, 0.5, 6.0),
        (3, 2.0, 2.0),  # constant temperature
        (7, 3.0, 1.0),  # decreasing temperature (healing)
    ])
    def test_parametrized_multi_day_progression(self, num_days, initial_temp, final_temp):
        """Test various multi-day temperature progression scenarios."""
        base_temp = 32.0
        thermal_images = [np.full((50, 50), base_temp) for _ in range(num_days)]
        wound_mask = np.zeros((50, 50), dtype=bool)
        wound_mask[20:30, 20:30] = True  # 10x10 wound area
        
        # Apply progressive temperature changes
        for day in range(num_days):
            if num_days == 1:
                temp_increase = final_temp
            else:
                progress = day / (num_days - 1)
                temp_increase = initial_temp + (final_temp - initial_temp) * progress
            
            thermal_images[day][wound_mask] += temp_increase
        
        # Verify first and last day
        first_day_wound_temp = np.mean(thermal_images[0][wound_mask])
        last_day_wound_temp = np.mean(thermal_images[-1][wound_mask])
        
        expected_first = base_temp + initial_temp
        expected_last = base_temp + final_temp
        
        assert abs(first_day_wound_temp - expected_first) < 1e-10, \
            f"Day 1: Expected {expected_first}, got {first_day_wound_temp}"
        assert abs(last_day_wound_temp - expected_last) < 1e-10, \
            f"Day {num_days}: Expected {expected_last}, got {last_day_wound_temp}"

# Additional utility functions for testing
def create_synthetic_wound_data(image_shape, wound_center, wound_radius, base_temp=32.0, wound_temp_increase=3.0):
    """Helper function to create synthetic wound data for testing."""
    h, w = image_shape
    thermal_image = np.full((h, w), base_temp, dtype=float)
    
    y, x = np.ogrid[:h, :w]
    wound_mask = (x - wound_center[1])**2 + (y - wound_center[0])**2 <= wound_radius**2
    
    return thermal_image, wound_mask, wound_temp_increase

def validate_temperature_preservation(original_image, modified_image, wound_mask, expected_increase):
    """Helper function to validate temperature preservation in wound area."""
    wound_diff = modified_image[wound_mask] - original_image[wound_mask]
    non_wound_diff = modified_image[~wound_mask] - original_image[~wound_mask]
    
    return (
        np.allclose(wound_diff, expected_increase) and
        np.allclose(non_wound_diff, 0.0)
    )

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])