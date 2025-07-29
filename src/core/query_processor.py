"""
Query processing module for parsing and structuring natural language queries.
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime


# Gemini API imports (Google Generative AI)
try:
    import google.generativeai as genai
except ImportError:
    genai = None

from models.schemas import QueryInput, StructuredQuery, ExtractedEntity, QueryType
from core.config import get_settings

logger = logging.getLogger(__name__)


class QueryProcessor:
    """Handles parsing and structuring of natural language queries."""
    
    def __init__(self):
        """Initialize the query processor."""
        self.settings = get_settings()
        self.client = None
        if genai and self.settings.gemini_api_key:
            genai.configure(api_key=self.settings.gemini_api_key)
            self.client = genai
    
    def process_query(self, query_input: QueryInput) -> StructuredQuery:
        """
        Process a natural language query into structured format.
        
        Args:
            query_input: Input query object
            
        Returns:
            Structured query with extracted entities and intent
        """
        logger.info(f"Processing query: {query_input.query}")
        
        # Extract entities using rule-based and LLM-based methods
        entities = self._extract_entities(query_input.query)
        
        # Determine intent
        intent = self._determine_intent(query_input.query, entities)
        
        # Classify query type
        query_type = self._classify_query_type(query_input.query, intent)
        
        structured_query = StructuredQuery(
            original_query=query_input.query,
            entities=entities,
            intent=intent,
            query_type=query_type
        )
        
        logger.info(f"Structured query: {structured_query.dict()}")
        return structured_query
    
    def _extract_entities(self, query: str) -> List[ExtractedEntity]:
        """
        Extract entities from the query using multiple methods.
        
        Args:
            query: Natural language query
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        # Rule-based entity extraction
        rule_based_entities = self._extract_entities_rule_based(query)
        entities.extend(rule_based_entities)
        
        # LLM-based entity extraction (if available)
        if self.client:
            llm_entities = self._extract_entities_llm(query)
            entities.extend(llm_entities)
        
        # Remove duplicates and merge similar entities
        entities = self._merge_entities(entities)
        
        return entities
    
    def _extract_entities_rule_based(self, query: str) -> List[ExtractedEntity]:
        """Extract entities using rule-based patterns."""
        entities = []
        query_lower = query.lower()
        
        # Age extraction
        age_patterns = [
            r'(\d+)[-\\s]?year[-\\s]?old',
            r'(\d+)[-\\s]?yo',
            r'age[-\\s]?(\d+)',
            r'(\d+)y',
            r'(\d+)m',  # for "46M" format
        ]
        
        for pattern in age_patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                entities.append(ExtractedEntity(
                    entity_type="age",
                    value=match,
                    confidence=0.9
                ))
        
        # Gender extraction
        gender_patterns = [
            r'\\b(male|female|m|f)\\b',
            r'(\d+)(m|f)\\b',  # for "46M" format
        ]
        
        for pattern in gender_patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                gender_value = match[1] if isinstance(match, tuple) else match
                if gender_value in ['m', 'male']:
                    gender_value = 'male'
                elif gender_value in ['f', 'female']:
                    gender_value = 'female'
                
                entities.append(ExtractedEntity(
                    entity_type="gender",
                    value=gender_value,
                    confidence=0.8
                ))
        
        # Location extraction (Indian cities)
        indian_cities = [
            'mumbai', 'delhi', 'bangalore', 'bengaluru', 'hyderabad', 'pune', 
            'chennai', 'kolkata', 'ahmedabad', 'surat', 'jaipur', 'lucknow',
            'kanpur', 'nagpur', 'indore', 'thane', 'bhopal', 'visakhapatnam',
            'pimpri-chinchwad', 'patna', 'vadodara', 'ghaziabad', 'ludhiana',
            'agra', 'nashik', 'faridabad', 'meerut', 'rajkot', 'kalyan-dombivli'
        ]
        
        for city in indian_cities:
            if city in query_lower:
                entities.append(ExtractedEntity(
                    entity_type="location",
                    value=city.title(),
                    confidence=0.9
                ))
        
        # Medical procedure extraction
        medical_procedures = [
            'knee surgery', 'hip replacement', 'cataract surgery', 'appendectomy',
            'bypass surgery', 'angioplasty', 'gallbladder surgery', 'hernia repair',
            'tonsillectomy', 'arthroscopy', 'mastectomy', 'colonoscopy'
        ]
        
        for procedure in medical_procedures:
            if procedure in query_lower:
                entities.append(ExtractedEntity(
                    entity_type="medical_procedure",
                    value=procedure.title(),
                    confidence=0.9
                ))
        
        # Policy duration extraction
        duration_patterns = [
            r'(\d+)[-\\s]?month[-\\s]?old policy',
            r'(\d+)[-\\s]?year[-\\s]?old policy',
            r'policy[-\\s]?(\d+)[-\\s]?months?',
            r'policy[-\\s]?(\d+)[-\\s]?years?'
        ]
        
        for pattern in duration_patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                entities.append(ExtractedEntity(
                    entity_type="policy_duration",
                    value=match,
                    confidence=0.8
                ))
        
        return entities
    
    def _extract_entities_llm(self, query: str) -> List[ExtractedEntity]:
        """Extract entities using Gemini LLM."""
        if not self.client:
            return []
        try:
            prompt = f"""
            Extract structured information from the following insurance-related query.
            Return a JSON list of entities with entity_type, value, and confidence (0-1).
            Query: "{query}"
            Focus on these entity types:
            - age: person's age in years
            - gender: male/female
            - location: city or place
            - medical_procedure: type of surgery or medical treatment
            - policy_duration: how long the insurance policy has been active
            - amount: any monetary amounts mentioned
            - insurance_type: type of insurance (health, life, etc.)
            Return only valid JSON array, no other text:
            """
            model = self.client.GenerativeModel(self.settings.model_name)
            response = model.generate_content(prompt)
            content = response.text.strip()
            # Parse JSON response
            try:
                entities_data = json.loads(content)
                entities = []
                for entity_data in entities_data:
                    if all(key in entity_data for key in ['entity_type', 'value', 'confidence']):
                        entities.append(ExtractedEntity(**entity_data))
                return entities
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse Gemini entity extraction response: {content}")
                return []
        except Exception as e:
            logger.error(f"Error in Gemini entity extraction: {str(e)}")
            return []
    
    def _determine_intent(self, query: str, entities: List[ExtractedEntity]) -> str:
        """
        Determine the intent of the query.
        
        Args:
            query: Original query
            entities: Extracted entities
            
        Returns:
            Identified intent
        """
        query_lower = query.lower()
        
        # Check for coverage-related keywords
        coverage_keywords = ['covered', 'coverage', 'eligible', 'claim', 'approve', 'payout']
        if any(keyword in query_lower for keyword in coverage_keywords):
            return "check_coverage"
        
        # Check for policy-related keywords
        policy_keywords = ['policy', 'premium', 'terms', 'conditions']
        if any(keyword in query_lower for keyword in policy_keywords):
            return "policy_inquiry"
        
        # Check for claim-related keywords
        claim_keywords = ['claim', 'submit', 'process', 'reimbursement']
        if any(keyword in query_lower for keyword in claim_keywords):
            return "process_claim"
        
        # Default intent based on entities
        entity_types = [entity.entity_type for entity in entities]
        if 'medical_procedure' in entity_types and 'age' in entity_types:
            return "check_coverage"
        
        return "general_inquiry"
    
    def _classify_query_type(self, query: str, intent: str) -> QueryType:
        """
        Classify the type of query.
        
        Args:
            query: Original query
            intent: Determined intent
            
        Returns:
            Query type classification
        """
        query_lower = query.lower()
        
        # Insurance-related keywords
        insurance_keywords = ['insurance', 'policy', 'claim', 'coverage', 'premium', 'medical']
        if any(keyword in query_lower for keyword in insurance_keywords):
            return QueryType.INSURANCE_CLAIM
        
        # Contract-related keywords
        contract_keywords = ['contract', 'agreement', 'terms', 'conditions', 'clause']
        if any(keyword in query_lower for keyword in contract_keywords):
            return QueryType.CONTRACT_ANALYSIS
        
        # Policy-related keywords
        policy_keywords = ['policy', 'rule', 'regulation', 'guideline']
        if any(keyword in query_lower for keyword in policy_keywords):
            return QueryType.POLICY_CHECK
        
        return QueryType.GENERAL
    
    def _merge_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """
        Merge duplicate or similar entities.
        
        Args:
            entities: List of entities to merge
            
        Returns:
            Merged list of entities
        """
        merged = {}
        
        for entity in entities:
            key = f"{entity.entity_type}_{entity.value.lower()}"
            
            if key in merged:
                # Keep the entity with higher confidence
                if entity.confidence > merged[key].confidence:
                    merged[key] = entity
            else:
                merged[key] = entity
        
        return list(merged.values())
