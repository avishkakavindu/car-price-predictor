/**
 * Client-side validation utilities
 */

import type { CarData, ValidationErrors } from '../types';
import {
  VALIDATION_RANGES,
  CAR_BRANDS,
  FUEL_TYPES,
  GEAR_TYPES,
} from './constants';

/**
 * Validate car input data
 * @param data - Car data to validate
 * @returns Object with field names as keys and error messages as values
 */
export const validateCarData = (data: Partial<CarData>): ValidationErrors => {
  const errors: ValidationErrors = {};

  // Brand validation
  if (!data.Brand || data.Brand.trim() === '') {
    errors.Brand = 'Brand is required';
  } else if (!CAR_BRANDS.includes(data.Brand as any)) {
    errors.Brand = 'Please select a valid brand';
  }

  // Model validation
  if (!data.Model || data.Model.trim() === '') {
    errors.Model = 'Model is required';
  }

  // Year of Manufacture validation
  if (!data.YOM) {
    errors.YOM = 'Year of Manufacture is required';
  } else if (
    data.YOM < VALIDATION_RANGES.YOM_MIN ||
    data.YOM > VALIDATION_RANGES.YOM_MAX
  ) {
    errors.YOM = `Year must be between ${VALIDATION_RANGES.YOM_MIN} and ${VALIDATION_RANGES.YOM_MAX}`;
  }

  // Engine capacity validation
  if (!data['Engine (cc)']) {
    errors['Engine (cc)'] = 'Engine capacity is required';
  } else if (
    data['Engine (cc)'] < VALIDATION_RANGES.ENGINE_MIN ||
    data['Engine (cc)'] > VALIDATION_RANGES.ENGINE_MAX
  ) {
    errors[
      'Engine (cc)'
    ] = `Engine capacity must be between ${VALIDATION_RANGES.ENGINE_MIN} and ${VALIDATION_RANGES.ENGINE_MAX} cc`;
  }

  // Gear validation
  if (!data.Gear) {
    errors.Gear = 'Transmission type is required';
  } else if (!GEAR_TYPES.includes(data.Gear as any)) {
    errors.Gear = 'Please select a valid transmission type';
  }

  // Fuel Type validation
  if (!data['Fuel Type']) {
    errors['Fuel Type'] = 'Fuel type is required';
  } else if (!FUEL_TYPES.includes(data['Fuel Type'] as any)) {
    errors['Fuel Type'] = 'Please select a valid fuel type';
  }

  // Mileage validation
  if (data['Millage(KM)'] === undefined || data['Millage(KM)'] === null) {
    errors['Millage(KM)'] = 'Mileage is required';
  } else if (
    data['Millage(KM)'] < VALIDATION_RANGES.MILEAGE_MIN ||
    data['Millage(KM)'] > VALIDATION_RANGES.MILEAGE_MAX
  ) {
    errors[
      'Millage(KM)'
    ] = `Mileage must be between ${VALIDATION_RANGES.MILEAGE_MIN.toLocaleString()} and ${VALIDATION_RANGES.MILEAGE_MAX.toLocaleString()} km`;
  }

  // Town validation
  if (!data.Town || data.Town.trim() === '') {
    errors.Town = 'Town is required';
  }

  // Condition validation
  if (!data.Condition) {
    errors.Condition = 'Vehicle condition is required';
  }

  // Leasing validation
  if (!data.Leasing) {
    errors.Leasing = 'Leasing status is required';
  }

  return errors;
};

/**
 * Check if there are any validation errors
 * @param errors - Validation errors object
 * @returns true if there are errors, false otherwise
 */
export const hasValidationErrors = (errors: ValidationErrors): boolean => {
  return Object.keys(errors).length > 0;
};

/**
 * Format number with thousand separators
 * @param value - Number to format
 * @returns Formatted string
 */
export const formatNumber = (value: number): string => {
  return value.toLocaleString();
};

/**
 * Format price in lakhs
 * @param lakhs - Price in lakhs
 * @returns Formatted price string
 */
export const formatPrice = (lakhs: number): string => {
  return `Rs. ${lakhs.toFixed(2)} Lakhs`;
};
