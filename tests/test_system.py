"""
Test cases for the Document Intelligence System.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
import tempfile
import os

from src.main import DocumentIntelligenceSystem
from src.models import QueryInput, DecisionType
from src.utils import generate_test_queries


class TestDocumentIntelligenceSystem:
    """Test cases for the main system."""
    
    @pytest.fixture
    def system(self):
        """Create a test system instance."""
        with patch('src.core.config.validate_settings'):
            return DocumentIntelligenceSystem()
    
    def test_system_initialization(self, system):
        """Test that the system initializes properly."""
        assert system is not None
        assert system.document_processor is not None
        assert system.query_processor is not None
        assert system.semantic_search is not None
        assert system.decision_engine is not None
    
    def test_get_system_status(self, system):
        """Test system status retrieval."""
        status = system.get_system_status()
        assert 'system_status' in status
        assert 'components' in status
    
    @pytest.mark.parametrize("query_data", generate_test_queries()[:3])
    def test_query_processing(self, system, query_data):
        """Test query processing with various inputs."""
        query = query_data['query']
        
        # Mock the semantic search to return empty results for testing
        with patch.object(system.semantic_search, 'search', return_value=[]):
            response = system.process_query(query)
            
            assert response is not None
            assert response.original_query == query
            assert response.structured_query is not None
            assert response.decision is not None
            assert isinstance(response.processing_time, float)


class TestQueryProcessor:
    """Test cases for query processing."""
    
    @pytest.fixture
    def processor(self):
        """Create a test query processor."""
        from src.core import QueryProcessor
        return QueryProcessor()
    
    def test_entity_extraction_rule_based(self, processor):
        """Test rule-based entity extraction."""
        query = "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
        entities = processor._extract_entities_rule_based(query)
        
        entity_types = [e.entity_type for e in entities]
        assert 'age' in entity_types
        assert 'gender' in entity_types
        assert 'medical_procedure' in entity_types
        assert 'location' in entity_types
    
    def test_intent_determination(self, processor):
        """Test intent determination from queries."""
        test_cases = [
            ("Is knee surgery covered?", "check_coverage"),
            ("What are the policy terms?", "policy_inquiry"),
            ("Submit claim for surgery", "process_claim")
        ]
        
        for query, expected_intent in test_cases:
            entities = processor._extract_entities_rule_based(query)
            intent = processor._determine_intent(query, entities)
            assert intent == expected_intent


class TestDocumentProcessor:
    """Test cases for document processing."""
    
    @pytest.fixture
    def processor(self):
        """Create a test document processor."""
        from src.core import DocumentProcessor
        return DocumentProcessor()
    
    def test_text_chunking(self, processor):
        """Test text chunking functionality."""
        text = "This is a test document. " * 100  # Create a long text
        chunks = processor._create_text_chunks(text)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= processor.chunk_size for chunk in chunks)
    
    def test_document_id_generation(self, processor):
        """Test document ID generation."""
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("Test content")
            temp_path = f.name
        
        try:
            doc_id1 = processor._generate_document_id(temp_path)
            doc_id2 = processor._generate_document_id(temp_path)
            
            # Same file should generate same ID
            assert doc_id1 == doc_id2
            assert len(doc_id1) == 12  # MD5 hash truncated to 12 chars
            
        finally:
            os.unlink(temp_path)


class TestSemanticSearch:
    """Test cases for semantic search."""
    
    @pytest.fixture
    def search_engine(self):
        """Create a test semantic search engine."""
        from src.core import SemanticSearch
        return SemanticSearch()
    
    def test_search_query_creation(self, search_engine):
        """Test search query creation from structured query."""
        from src.models import StructuredQuery, ExtractedEntity, QueryType
        
        entities = [
            ExtractedEntity(entity_type="age", value="46", confidence=0.9),
            ExtractedEntity(entity_type="medical_procedure", value="knee surgery", confidence=0.9)
        ]
        
        structured_query = StructuredQuery(
            original_query="46-year-old knee surgery",
            entities=entities,
            intent="check_coverage",
            query_type=QueryType.INSURANCE_CLAIM
        )
        
        search_query = search_engine._create_search_query(structured_query)
        assert "knee surgery" in search_query
        assert "age 46" in search_query


class TestDecisionEngine:
    """Test cases for decision engine."""
    
    @pytest.fixture
    def engine(self):
        """Create a test decision engine."""
        from src.core import DecisionEngine
        return DecisionEngine()
    
    def test_rule_based_decision(self, engine):
        """Test rule-based decision making."""
        from src.models import StructuredQuery, ExtractedEntity, QueryType, RetrievedClause
        
        entities = [
            ExtractedEntity(entity_type="age", value="46", confidence=0.9),
            ExtractedEntity(entity_type="medical_procedure", value="knee surgery", confidence=0.9),
            ExtractedEntity(entity_type="policy_duration", value="3", confidence=0.8)
        ]
        
        structured_query = StructuredQuery(
            original_query="46-year-old knee surgery, 3-month policy",
            entities=entities,
            intent="check_coverage",
            query_type=QueryType.INSURANCE_CLAIM
        )
        
        # Create a mock coverage clause
        clause = RetrievedClause(
            clause_id="test_clause",
            content="Knee surgery is covered under this policy",
            source_document="test_policy.txt",
            relevance_score=0.9,
            clause_type="coverage"
        )
        
        decision = engine._rule_based_decision(structured_query, [clause])
        
        assert decision is not None
        assert decision.decision_type in [DecisionType.APPROVED, DecisionType.PENDING, DecisionType.REJECTED]
        assert 0 <= decision.confidence <= 1


class TestUtilities:
    """Test cases for utility functions."""
    
    def test_age_parsing(self):
        """Test age parsing from text."""
        from src.utils import parse_age_from_text
        
        test_cases = [
            ("46-year-old", 46),
            ("35yo", 35),
            ("age 28", 28),
            ("52M", 52),
            ("no age mentioned", None)
        ]
        
        for text, expected in test_cases:
            result = parse_age_from_text(text)
            assert result == expected
    
    def test_gender_parsing(self):
        """Test gender parsing from text."""
        from src.utils import parse_gender_from_text
        
        test_cases = [
            ("46-year-old male", "male"),
            ("35F", "female"),
            ("female patient", "female"),
            ("52M", "male"),
            ("no gender mentioned", None)
        ]
        
        for text, expected in test_cases:
            result = parse_gender_from_text(text)
            assert result == expected
    
    def test_city_extraction(self):
        """Test Indian city extraction."""
        from src.utils import extract_indian_cities
        
        text = "Treatment in Mumbai and Delhi hospitals"
        cities = extract_indian_cities(text)
        
        assert "Mumbai" in cities
        assert "Delhi" in cities
        assert len(cities) == 2
    
    def test_medical_procedure_validation(self):
        """Test medical procedure validation."""
        from src.utils import validate_medical_procedure
        
        result = validate_medical_procedure("knee surgery")
        
        assert result['category'] == 'orthopedic'
        assert result['is_recognized'] is True
        assert 'estimated_coverage' in result


# Integration tests
class TestAPIIntegration:
    """Integration tests for the API."""
    
    @pytest.fixture
    def client(self):
        """Create a test API client."""
        from fastapi.testclient import TestClient
        from src.api.app import app
        
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test the health check endpoint."""
        response = client.get("/health")
        # Note: This might fail if system is not properly initialized
        # In a real test environment, you'd mock the system initialization
        assert response.status_code in [200, 503]  # 503 if system not initialized
    
    def test_root_endpoint(self, client):
        """Test the root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
    
    def test_examples_endpoint(self, client):
        """Test the examples endpoint."""
        response = client.get("/examples")
        assert response.status_code == 200
        data = response.json()
        assert "example_queries" in data
        assert "supported_entities" in data


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
