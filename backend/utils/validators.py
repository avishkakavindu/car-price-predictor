"""
Input validation utilities
"""
from typing import Dict, List, Any


# Valid options for categorical features
VALID_BRANDS = [
    'TOYOTA', 'SUZUKI', 'NISSAN', 'HONDA', 'MITSUBISHI',
    'PERODUA', 'MICRO', 'HYUNDAI', 'MAZDA', 'MERCEDES-BENZ',
    'TATA', 'KIA', 'DAIHATSU', 'BMW', 'RENAULT', 'Other'
]

VALID_FUEL_TYPES = ['Petrol', 'Diesel', 'Hybrid', 'Electric']
VALID_GEAR_TYPES = ['Automatic', 'Manual']
VALID_CONDITIONS = ['NEW', 'USED']
VALID_LEASING = ['No Leasing', 'Ongoing Lease']
VALID_AVAILABILITY = ['Available', 'Not_Available']


def validate_car_data(data: Dict[str, Any]) -> List[str]:
    """
    Validate car input data

    Args:
        data: Dictionary with car features

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Check required fields
    required_fields = [
        'Brand', 'Model', 'YOM', 'Engine (cc)', 'Gear', 'Fuel Type',
        'Millage(KM)', 'Town', 'Date', 'Leasing', 'Condition',
        'AIR CONDITION', 'POWER STEERING', 'POWER MIRROR', 'POWER WINDOW'
    ]

    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")

    # If missing fields, return early
    if errors:
        return errors

    # Validate categorical fields
    if data.get('Brand') not in VALID_BRANDS:
        errors.append(f"Invalid Brand. Must be one of: {', '.join(VALID_BRANDS)}")

    if data.get('Fuel Type') not in VALID_FUEL_TYPES:
        errors.append(f"Invalid Fuel Type. Must be one of: {', '.join(VALID_FUEL_TYPES)}")

    if data.get('Gear') not in VALID_GEAR_TYPES:
        errors.append(f"Invalid Gear. Must be one of: {', '.join(VALID_GEAR_TYPES)}")

    if data.get('Condition') not in VALID_CONDITIONS:
        errors.append(f"Invalid Condition. Must be one of: {', '.join(VALID_CONDITIONS)}")

    if data.get('Leasing') not in VALID_LEASING:
        errors.append(f"Invalid Leasing. Must be one of: {', '.join(VALID_LEASING)}")

    # Validate numerical ranges
    try:
        yom = int(data.get('YOM', 0))
        if yom < 1990 or yom > 2026:
            errors.append("YOM (Year of Manufacture) must be between 1990 and 2026")
    except (ValueError, TypeError):
        errors.append("YOM must be a valid integer")

    try:
        engine = float(data.get('Engine (cc)', 0))
        if engine < 500 or engine > 5000:
            errors.append("Engine (cc) must be between 500 and 5000")
    except (ValueError, TypeError):
        errors.append("Engine (cc) must be a valid number")

    try:
        mileage = float(data.get('Millage(KM)', 0))
        if mileage < 0 or mileage > 500000:
            errors.append("Millage(KM) must be between 0 and 500,000")
    except (ValueError, TypeError):
        errors.append("Millage(KM) must be a valid number")

    return errors
