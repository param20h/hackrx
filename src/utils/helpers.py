"""
Utility functions for the Document Intelligence System.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import re

logger = logging.getLogger(__name__)


def format_currency(amount: float, currency: str = "INR") -> str:
    """
    Format amount as currency string.
    
    Args:
        amount: Amount to format
        currency: Currency code
        
    Returns:
        Formatted currency string
    """
    if currency == "INR":
        return f"₹{amount:,.2f}"
    else:
        return f"{currency} {amount:,.2f}"


def parse_age_from_text(text: str) -> Optional[int]:
    """
    Parse age from text using various patterns.
    
    Args:
        text: Text containing age information
        
    Returns:
        Parsed age or None if not found
    """
    patterns = [
        r'(\d+)[-\s]?year[-\s]?old',
        r'(\d+)[-\s]?yo',
        r'age[-\s]?(\d+)',
        r'(\d+)y',
        r'(\d+)m',  # for "46M" format
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                continue
    
    return None


def parse_gender_from_text(text: str) -> Optional[str]:
    """
    Parse gender from text.
    
    Args:
        text: Text containing gender information
        
    Returns:
        Parsed gender (male/female) or None if not found
    """
    text_lower = text.lower()
    
    # Direct gender mentions
    if re.search(r'\bmale\b', text_lower):
        return 'male'
    elif re.search(r'\bfemale\b', text_lower):
        return 'female'
    
    # Gender with age patterns (e.g., "46M", "35F")
    pattern = r'(\d+)(m|f)\b'
    match = re.search(pattern, text_lower)
    if match:
        gender_char = match.group(2)
        return 'male' if gender_char == 'm' else 'female'
    
    return None


def extract_indian_cities(text: str) -> List[str]:
    """
    Extract Indian city names from text.
    
    Args:
        text: Text to search for cities
        
    Returns:
        List of found cities
    """
    indian_cities = {
        'mumbai', 'delhi', 'bangalore', 'bengaluru', 'hyderabad', 'pune', 
        'chennai', 'kolkata', 'ahmedabad', 'surat', 'jaipur', 'lucknow',
        'kanpur', 'nagpur', 'indore', 'thane', 'bhopal', 'visakhapatnam',
        'pimpri-chinchwad', 'patna', 'vadodara', 'ghaziabad', 'ludhiana',
        'agra', 'nashik', 'faridabad', 'meerut', 'rajkot', 'kalyan-dombivli',
        'chandigarh', 'guwahati', 'coimbatore', 'madurai', 'trichy',
        'kochi', 'kozhikode', 'thiruvananthapuram', 'mysore', 'mangalore'
    }
    
    found_cities = []
    text_lower = text.lower()
    
    for city in indian_cities:
        if city in text_lower:
            found_cities.append(city.title())
    
    return found_cities


def calculate_policy_age(start_date: datetime, current_date: datetime = None) -> Dict[str, Any]:
    """
    Calculate policy age and related information.
    
    Args:
        start_date: Policy start date
        current_date: Current date (defaults to now)
        
    Returns:
        Dictionary with policy age information
    """
    if current_date is None:
        current_date = datetime.now()
    
    age_delta = current_date - start_date
    
    years = age_delta.days // 365
    months = (age_delta.days % 365) // 30
    days = age_delta.days % 30
    
    return {
        'total_days': age_delta.days,
        'years': years,
        'months': months + (years * 12),
        'days': days,
        'is_new_policy': age_delta.days < 180,  # Less than 6 months
        'waiting_periods': {
            'initial_30_days': age_delta.days >= 30,
            'pre_existing_24_months': age_delta.days >= (24 * 30),
            'maternity_9_months': age_delta.days >= (9 * 30),
            'knee_surgery_12_months': age_delta.days >= (12 * 30),
            'cardiac_24_months': age_delta.days >= (24 * 30)
        }
    }


def validate_medical_procedure(procedure: str) -> Dict[str, Any]:
    """
    Validate and categorize medical procedure.
    
    Args:
        procedure: Medical procedure name
        
    Returns:
        Dictionary with procedure validation and categorization
    """
    procedure_lower = procedure.lower()
    
    # Define procedure categories
    categories = {
        'orthopedic': [
            'knee surgery', 'hip replacement', 'arthroscopy', 'joint replacement',
            'fracture repair', 'spine surgery', 'shoulder surgery'
        ],
        'cardiac': [
            'bypass surgery', 'angioplasty', 'heart surgery', 'cardiac',
            'pacemaker', 'valve replacement'
        ],
        'ophthalmology': [
            'cataract surgery', 'retinal surgery', 'glaucoma surgery',
            'eye surgery', 'lasik'
        ],
        'general_surgery': [
            'appendectomy', 'gallbladder surgery', 'hernia repair',
            'cholecystectomy', 'laparoscopy'
        ],
        'emergency': [
            'emergency surgery', 'trauma surgery', 'accident',
            'life threatening'
        ]
    }
    
    # Find matching category
    matched_category = None
    for category, procedures in categories.items():
        if any(proc in procedure_lower for proc in procedures):
            matched_category = category
            break
    
    # Estimate coverage based on procedure type
    coverage_estimates = {
        'orthopedic': {'min': 150000, 'max': 300000},
        'cardiac': {'min': 300000, 'max': 500000},
        'ophthalmology': {'min': 30000, 'max': 80000},
        'general_surgery': {'min': 50000, 'max': 150000},
        'emergency': {'min': 100000, 'max': 200000}
    }
    
    return {
        'procedure': procedure,
        'category': matched_category or 'general',
        'is_recognized': matched_category is not None,
        'estimated_coverage': coverage_estimates.get(matched_category, {'min': 50000, 'max': 100000}),
        'complexity': 'high' if matched_category in ['cardiac', 'orthopedic'] else 'medium'
    }


def create_response_summary(response) -> str:
    """
    Create a human-readable summary of the system response.
    
    Args:
        response: SystemResponse object
        
    Returns:
        Formatted summary string
    """
    summary_parts = []
    
    # Decision summary
    decision = response.decision
    summary_parts.append(f"Decision: {decision.decision_type.value.upper()}")
    
    if decision.amount:
        amount_str = format_currency(decision.amount, decision.currency)
        summary_parts.append(f"Amount: {amount_str}")
    
    summary_parts.append(f"Confidence: {decision.confidence:.0%}")
    
    # Add key entities
    entities = response.structured_query.entities
    if entities:
        entity_summary = []
        for entity in entities[:3]:  # Top 3 entities
            entity_summary.append(f"{entity.entity_type}: {entity.value}")
        summary_parts.append(f"Key Details: {', '.join(entity_summary)}")
    
    # Add processing time
    summary_parts.append(f"Processed in {response.processing_time:.2f}s")
    
    return " | ".join(summary_parts)


def save_response_log(response, log_file: str = "query_log.json"):
    """
    Save system response to log file.
    
    Args:
        response: SystemResponse object
        log_file: Log file path
    """
    try:
        # Convert response to dict
        log_entry = {
            'timestamp': response.timestamp.isoformat(),
            'query_id': response.query_id,
            'query': response.original_query,
            'decision': response.decision.decision_type.value,
            'amount': response.decision.amount,
            'confidence': response.decision.confidence,
            'processing_time': response.processing_time,
            'entities': [
                {
                    'type': e.entity_type,
                    'value': e.value,
                    'confidence': e.confidence
                }
                for e in response.structured_query.entities
            ]
        }
        
        # Read existing log
        log_entries = []
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    log_entries = json.load(f)
            except (json.JSONDecodeError, IOError):
                log_entries = []
        
        # Add new entry
        log_entries.append(log_entry)
        
        # Keep only last 1000 entries
        if len(log_entries) > 1000:
            log_entries = log_entries[-1000:]
        
        # Save updated log
        with open(log_file, 'w') as f:
            json.dump(log_entries, f, indent=2)
            
    except Exception as e:
        logger.error(f"Error saving response log: {e}")


def generate_test_queries() -> List[Dict[str, str]]:
    """
    Generate test queries for system validation.
    
    Returns:
        List of test query dictionaries
    """
    return [
        {
            'query': '46-year-old male, knee surgery in Pune, 3-month-old insurance policy',
            'description': 'Basic insurance claim with all key entities',
            'expected_entities': ['age', 'gender', 'medical_procedure', 'location', 'policy_duration']
        },
        {
            'query': '35F cataract surgery Mumbai 2 year policy',
            'description': 'Compact format with abbreviated gender',
            'expected_entities': ['age', 'gender', 'medical_procedure', 'location', 'policy_duration']
        },
        {
            'query': 'Is hip replacement covered for 65-year-old in Delhi?',
            'description': 'Coverage inquiry question format',
            'expected_entities': ['medical_procedure', 'age', 'location']
        },
        {
            'query': 'Claim for appendectomy, patient age 28, policy 1 year old',
            'description': 'Formal claim submission format',
            'expected_entities': ['medical_procedure', 'age', 'policy_duration']
        },
        {
            'query': 'What is the coverage amount for bypass surgery in Chennai?',
            'description': 'Amount inquiry for cardiac procedure',
            'expected_entities': ['medical_procedure', 'location']
        },
        {
            'query': '52 year old female gallbladder surgery Bangalore',
            'description': 'No policy duration mentioned',
            'expected_entities': ['age', 'gender', 'medical_procedure', 'location']
        },
        {
            'query': 'Emergency hernia repair for 40M in Hyderabad, policy active for 6 months',
            'description': 'Emergency procedure with policy info',
            'expected_entities': ['medical_procedure', 'age', 'gender', 'location', 'policy_duration']
        }
    ]
