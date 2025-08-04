import requests
import json
import numpy as np
import math
import random

def find_nearby_hospitals(address, radius_miles=15):
    """
    Find hospitals with neurosurgery departments near a given address.
    
    In a real implementation, this would use Google Maps or a similar API.
    For this demo, we'll generate simulated data to demonstrate functionality.
    
    Args:
        address: User's address or city
        radius_miles: Search radius in miles
        
    Returns:
        List of dictionaries containing hospital information
    """
    # In a real app, this would use geocoding to get coordinates from the address
    # And then search for nearby hospitals using Places API or similar
    
    # For demonstration, generate simulated coordinates based on the address string
    # This is just to make the demo work - in production, use real geocoding
    latitude, longitude = get_simulated_coordinates(address)
    
    # Get simulated hospitals
    hospitals = get_simulated_hospitals(latitude, longitude, radius_miles)
    
    return hospitals

def get_simulated_coordinates(address):
    """
    Generate simulated coordinates based on the input address.
    In a real app, this would use Google's Geocoding API or similar.
    
    Args:
        address: User's address or city
        
    Returns:
        Tuple of (latitude, longitude)
    """
    # Use hash of address to generate "random" but consistent coordinates
    address_hash = hash(address) % 10000
    
    # Base coordinates (roughly middle of continental USA)
    base_lat = 39.8283
    base_lng = -98.5795
    
    # Adjust based on hash
    lat_offset = (address_hash % 100) / 10.0
    lng_offset = ((address_hash // 100) % 100) / 10.0
    
    # Cities tend to be within ±25 degrees of the base
    latitude = base_lat + (lat_offset - 5)
    longitude = base_lng + (lng_offset - 5)
    
    return latitude, longitude

def get_simulated_hospitals(latitude, longitude, radius_miles):
    """
    Generate a list of simulated hospitals near the given coordinates.
    
    Args:
        latitude: User's latitude
        longitude: User's longitude
        radius_miles: Search radius in miles
        
    Returns:
        List of dictionaries containing hospital information
    """
    # Generate between 3 and 8 hospitals
    num_hospitals = random.randint(3, 8)
    
    hospitals = []
    
    # Common hospital name parts
    prefixes = ["Memorial", "University", "Regional", "Community", "General", "St.", "Methodist", "Baptist"]
    suffixes = ["Hospital", "Medical Center", "Health Center", "Health System", "Hospital Center"]
    specialties = ["Neurosurgical", "Neurological", "Brain & Spine", "Neuroscience"]
    
    for i in range(num_hospitals):
        # Generate random coordinates within the radius
        hospital_lat, hospital_lng = get_random_nearby_coordinates(latitude, longitude, radius_miles)
        
        # Calculate distance
        distance = calculate_distance(latitude, longitude, hospital_lat, hospital_lng)
        
        # Generate a hospital name
        if random.random() < 0.3:
            name = f"{random.choice(prefixes)} {random.choice(specialties)} {random.choice(suffixes)}"
        else:
            name = f"{random.choice(prefixes)} {random.choice(suffixes)}"
        
        # Generate simulated street address
        street_number = random.randint(100, 9999)
        streets = ["Main St", "Oak Ave", "Maple Rd", "Park Blvd", "Washington St", "Jefferson Ave", 
                   "Medical Dr", "Health Way", "Hospital Blvd", "University Ave"]
        street = random.choice(streets)
        
        # Generate city name based on coordinates
        cities = ["Springfield", "Franklin", "Greenville", "Bristol", "Madison", "Georgetown", 
                  "Salem", "Oxford", "Kingston", "Burlington"]
        city = random.choice(cities)
        
        # Generate state abbreviation
        states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID"]
        state = random.choice(states)
        
        # Generate ZIP code
        zip_code = random.randint(10000, 99999)
        
        # Generate phone number
        area_code = random.randint(200, 999)
        prefix = random.randint(200, 999)
        line = random.randint(1000, 9999)
        phone = f"({area_code}) {prefix}-{line}"
        
        # Create hospital object
        hospital = {
            "name": name,
            "lat": hospital_lat,
            "lng": hospital_lng,
            "distance": round(distance, 1),
            "address": f"{street_number} {street}, {city}, {state} {zip_code}",
            "phone": phone,
            "specialties": ["Neurosurgery", "Neurology", "Radiology", "Oncology"]
        }
        
        hospitals.append(hospital)
    
    # Sort by distance
    hospitals.sort(key=lambda x: x["distance"])
    
    return hospitals

def get_random_nearby_coordinates(lat, lng, radius_miles):
    """
    Generate random coordinates within a specified radius.
    
    Args:
        lat: Center latitude
        lng: Center longitude
        radius_miles: Radius in miles
        
    Returns:
        Tuple of (latitude, longitude)
    """
    # Convert radius from miles to degrees (approximate)
    radius_lat = (radius_miles / 69.0) * random.random()
    radius_lng = (radius_miles / (69.0 * math.cos(math.radians(lat)))) * random.random()
    
    # Random angle
    angle = random.random() * 2 * math.pi
    
    # Calculate new coordinates
    new_lat = lat + (radius_lat * math.sin(angle))
    new_lng = lng + (radius_lng * math.cos(angle))
    
    return new_lat, new_lng

def calculate_distance(lat1, lng1, lat2, lng2):
    """
    Calculate the distance between two coordinate points using the Haversine formula.
    
    Args:
        lat1: Latitude of first point
        lng1: Longitude of first point
        lat2: Latitude of second point
        lng2: Longitude of second point
        
    Returns:
        Distance in miles
    """
    # Earth radius in miles
    radius = 3958.8
    
    # Convert to radians
    lat1 = math.radians(lat1)
    lng1 = math.radians(lng1)
    lat2 = math.radians(lat2)
    lng2 = math.radians(lng2)
    
    # Haversine formula
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = radius * c
    
    return distance
