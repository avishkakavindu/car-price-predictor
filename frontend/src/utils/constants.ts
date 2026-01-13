/**
 * Constants for dropdown options and validation
 * These values must match what the backend expects
 */

// Car brands (Top 15 from Sri Lankan market + Other)
export const CAR_BRANDS = [
  'TOYOTA',
  'SUZUKI',
  'NISSAN',
  'HONDA',
  'MITSUBISHI',
  'PERODUA',
  'MICRO',
  'HYUNDAI',
  'MAZDA',
  'MERCEDES-BENZ',
  'TATA',
  'KIA',
  'DAIHATSU',
  'BMW',
  'RENAULT',
  'Other'
] as const;

// Popular car models by brand (Sri Lankan market)
export const CAR_MODELS: Record<string, string[]> = {
  'TOYOTA': ['Corolla', 'Prius', 'Aqua', 'Vitz', 'Land Cruiser', 'Camry', 'Yaris', 'Hilux', 'Fortuner', 'RAV4', 'Axio', 'Premio', 'Allion', 'Voxy', 'Noah', 'Other'],
  'SUZUKI': ['Alto', 'Wagon R', 'Swift', 'Celerio', 'Vitara', 'Ertiga', 'Dzire', 'Baleno', 'S-Presso', 'Jimny', 'Other'],
  'NISSAN': ['Sunny', 'March', 'Tiida', 'X-Trail', 'Kicks', 'Leaf', 'Navara', 'Caravan', 'Sylphy', 'Other'],
  'HONDA': ['Fit', 'Civic', 'Vezel', 'Grace', 'CR-V', 'Accord', 'City', 'Insight', 'Freed', 'Odyssey', 'Other'],
  'MITSUBISHI': ['Lancer', 'Montero', 'Pajero', 'Outlander', 'ASX', 'Mirage', 'L300', 'Triton', 'Other'],
  'PERODUA': ['Axia', 'Bezza', 'Myvi', 'Alza', 'Aruz', 'Other'],
  'MICRO': ['Panda', 'Other'],
  'HYUNDAI': ['i10', 'i20', 'Elantra', 'Accent', 'Santa Fe', 'Tucson', 'Creta', 'Venue', 'Verna', 'Other'],
  'MAZDA': ['Demio', 'Axela', 'Atenza', 'CX-3', 'CX-5', 'Premacy', 'Biante', 'Other'],
  'MERCEDES-BENZ': ['C-Class', 'E-Class', 'S-Class', 'A-Class', 'GLA', 'GLC', 'GLE', 'Other'],
  'TATA': ['Nano', 'Indica', 'Indigo', 'Safari', 'Nexon', 'Other'],
  'KIA': ['Picanto', 'Rio', 'Cerato', 'Sportage', 'Sorento', 'Seltos', 'Carnival', 'Other'],
  'DAIHATSU': ['Mira', 'Move', 'Terios', 'Cuore', 'Sirion', 'Other'],
  'BMW': ['3 Series', '5 Series', '7 Series', 'X1', 'X3', 'X5', 'X6', 'Other'],
  'RENAULT': ['Clio', 'Megane', 'Duster', 'Captur', 'Kwid', 'Other'],
  'Other': ['Other']
};

// Gear/Transmission types
export const GEAR_TYPES = ['Automatic', 'Manual'] as const;

// Fuel types
export const FUEL_TYPES = ['Petrol', 'Diesel', 'Hybrid', 'Electric'] as const;

// Sri Lankan towns/cities
export const TOWNS = [
  'Colombo',
  'Gampaha',
  'Kandy',
  'Negombo',
  'Nugegoda',
  'Dehiwala-Mount-Lavinia',
  'Battaramulla',
  'Matara',
  'Anuradhapura',
  'Ambalangoda',
  'Galle',
  'Jaffna',
  'Kurunegala',
  'Ratnapura'
] as const;

// Leasing options
export const LEASING_OPTIONS = ['No Leasing', 'Ongoing Lease'] as const;

// Vehicle conditions
export const CONDITIONS = ['NEW', 'USED'] as const;

// Feature availability
export const AVAILABLE_OPTIONS = ['Available', 'Not_Available'] as const;

// Validation ranges
export const VALIDATION_RANGES = {
  YOM_MIN: 1990,
  YOM_MAX: new Date().getFullYear(),
  ENGINE_MIN: 500,
  ENGINE_MAX: 5000,
  MILEAGE_MIN: 0,
  MILEAGE_MAX: 500000
} as const;

// API configuration
export const API_CONFIG = {
  BASE_URL: import.meta.env.VITE_API_URL || '/api',
  TIMEOUT: 30000, // 30 seconds
  SHAP_POLL_INTERVAL: 2000 // Poll every 2 seconds
} as const;

// Type exports for TypeScript
export type Brand = typeof CAR_BRANDS[number];
export type GearType = typeof GEAR_TYPES[number];
export type FuelType = typeof FUEL_TYPES[number];
export type Town = typeof TOWNS[number];
export type LeasingOption = typeof LEASING_OPTIONS[number];
export type VehicleCondition = typeof CONDITIONS[number];
export type FeatureAvailability = typeof AVAILABLE_OPTIONS[number];
