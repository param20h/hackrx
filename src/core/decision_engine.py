"""
Decision engine module for making decisions based on retrieved clauses and query context.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime


# Gemini API imports (Google Generative AI)
try:
    import google.generativeai as genai
except ImportError:
    genai = None

from models.schemas import (
    StructuredQuery, RetrievedClause, Decision, ClauseReference, 
    SystemResponse, DecisionType
)
from core.config import get_settings

logger = logging.getLogger(__name__)


class DecisionEngine:
    """Makes decisions based on retrieved clauses and query context."""
    
    def __init__(self):
        """Initialize the decision engine."""
        self.settings = get_settings()
        self.client = None
        if genai and self.settings.gemini_api_key:
            genai.configure(api_key=self.settings.gemini_api_key)
            self.client = genai
    
    def make_decision(
        self, 
        structured_query: StructuredQuery, 
        retrieved_clauses: List[RetrievedClause]
    ) -> SystemResponse:
        """
        Make a decision based on the query and retrieved clauses.
        
        Args:
            structured_query: The processed query
            retrieved_clauses: Relevant clauses from documents
            
        Returns:
            Complete system response with decision and justification
        """
        start_time = datetime.now()
        
        logger.info(f"Making decision for query: {structured_query.original_query}")
        
        # Analyze clauses and make decision
        decision = self._analyze_and_decide(structured_query, retrieved_clauses)
        
        # Create clause references
        supporting_clauses = self._create_clause_references(retrieved_clauses, decision)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Generate response ID
        query_id = f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        response = SystemResponse(
            query_id=query_id,
            original_query=structured_query.original_query,
            structured_query=structured_query,
            decision=decision,
            supporting_clauses=supporting_clauses,
            processing_time=processing_time
        )
        
        logger.info(f"Decision made: {decision.decision_type} with confidence {decision.confidence}")
        return response
    
    def _analyze_and_decide(
        self, 
        structured_query: StructuredQuery, 
        retrieved_clauses: List[RetrievedClause]
    ) -> Decision:
        """
        Analyze the context and make a decision.
        
        Args:
            structured_query: Structured query
            retrieved_clauses: Retrieved relevant clauses
            
        Returns:
            Decision object
        """
        # If no clauses found, return a default decision
        if not retrieved_clauses:
            return Decision(
                decision_type=DecisionType.REQUIRES_REVIEW,
                confidence=0.1,
                reasoning="No relevant clauses found in the policy documents to make a determination."
            )
        
        # Use LLM for decision making if available
        if self.client:
            return self._llm_decision(structured_query, retrieved_clauses)
        else:
            return self._rule_based_decision(structured_query, retrieved_clauses)
    
    def _llm_decision(
        self, 
        structured_query: StructuredQuery, 
        retrieved_clauses: List[RetrievedClause]
    ) -> Decision:
        """
        Use Gemini LLM to make a decision based on the context.
        """
        if not self.client:
            return self._rule_based_decision(structured_query, retrieved_clauses)
        try:
            # Prepare context for LLM
            context = self._prepare_llm_context(structured_query, retrieved_clauses)
            prompt = f"""
You are an expert insurance claims analyst. Based on the provided policy clauses and customer query, 
make a decision about the insurance claim or coverage inquiry.

Customer Query: {structured_query.original_query}

Extracted Information:
{self._format_entities(structured_query.entities)}

Relevant Policy Clauses:
{context}

Instructions:
1. Analyze if the requested procedure/treatment is covered under the policy
2. Check for any exclusions or waiting periods that might apply
3. Consider the customer's age, location, and policy duration
4. Make a clear decision: APPROVED, REJECTED, PENDING, or REQUIRES_REVIEW
5. If approved, estimate a coverage amount in INR
6. Provide detailed reasoning with specific clause references

Return your response in the following JSON format:
{{
    "decision_type": "APPROVED|REJECTED|PENDING|REQUIRES_REVIEW",
    "amount": <amount_in_inr_or_null>,
    "currency": "INR",
    "confidence": <confidence_0_to_1>,
    "reasoning": "<detailed_reasoning_with_clause_references>"
}}

Be thorough in your analysis and reference specific clauses in your reasoning.
            """
            model = self.client.GenerativeModel(self.settings.model_name)
            response = model.generate_content(prompt)
            content = response.text.strip()
            # Parse JSON response
            try:
                decision_data = json.loads(content)
                # Map decision type
                decision_type_map = {
                    "APPROVED": DecisionType.APPROVED,
                    "REJECTED": DecisionType.REJECTED,
                    "PENDING": DecisionType.PENDING,
                    "REQUIRES_REVIEW": DecisionType.REQUIRES_REVIEW
                }
                decision_type = decision_type_map.get(
                    decision_data.get("decision_type", "REQUIRES_REVIEW"),
                    DecisionType.REQUIRES_REVIEW
                )
                return Decision(
                    decision_type=decision_type,
                    amount=decision_data.get("amount"),
                    currency=decision_data.get("currency", "INR"),
                    confidence=min(1.0, max(0.0, decision_data.get("confidence", 0.5))),
                    reasoning=decision_data.get("reasoning", "Decision made based on policy analysis.")
                )
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse Gemini decision response: {content}")
                # Extract decision from text if JSON parsing fails
                return self._extract_decision_from_text(content)
        except Exception as e:
            logger.error(f"Error in Gemini decision making: {e}")
            return self._rule_based_decision(structured_query, retrieved_clauses)
    
    def _rule_based_decision(
        self, 
        structured_query: StructuredQuery, 
        retrieved_clauses: List[RetrievedClause]
    ) -> Decision:
        """
        Make a decision using rule-based logic.
        
        Args:
            structured_query: Structured query
            retrieved_clauses: Retrieved clauses
            
        Returns:
            Rule-based decision
        """
        # Extract entities for rule-based logic
        entities = {entity.entity_type: entity.value for entity in structured_query.entities}
        
        # Default decision
        decision_type = DecisionType.REQUIRES_REVIEW
        amount = None
        confidence = 0.5
        reasoning_parts = []
        
        # Check for coverage clauses
        coverage_clauses = [c for c in retrieved_clauses if c.clause_type == "coverage"]
        exclusion_clauses = [c for c in retrieved_clauses if c.clause_type == "exclusion"]
        
        # Basic coverage check
        if coverage_clauses:
            decision_type = DecisionType.APPROVED
            confidence = 0.7
            reasoning_parts.append("Found relevant coverage clauses in the policy.")
            
            # Basic amount estimation for medical procedures
            if "medical_procedure" in entities:
                procedure = entities["medical_procedure"].lower()
                if "knee surgery" in procedure:
                    amount = 150000  # Estimated coverage for knee surgery
                elif "surgery" in procedure:
                    amount = 100000  # Generic surgery coverage
                else:
                    amount = 50000   # Basic medical procedure
        
        # Check for exclusions
        if exclusion_clauses:
            decision_type = DecisionType.REJECTED
            confidence = 0.8
            reasoning_parts.append("Found exclusion clauses that may apply to this case.")
        
        # Check waiting period for new policies
        if "policy_duration" in entities:
            try:
                duration = int(entities["policy_duration"])
                if duration < 6:  # Less than 6 months
                    decision_type = DecisionType.PENDING
                    confidence = 0.6
                    reasoning_parts.append(f"Policy is only {duration} months old, waiting period may apply.")
            except ValueError:
                pass
        
        # Age-based adjustments
        if "age" in entities:
            try:
                age = int(entities["age"])
                if age > 60:
                    confidence *= 0.9  # Slightly lower confidence for senior citizens
                    reasoning_parts.append("Senior citizen category may have specific terms.")
            except ValueError:
                pass
        
        reasoning = " ".join(reasoning_parts) if reasoning_parts else "Rule-based analysis completed."
        
        return Decision(
            decision_type=decision_type,
            amount=amount,
            currency="INR",
            confidence=min(1.0, confidence),
            reasoning=reasoning
        )
    
    def _prepare_llm_context(
        self, 
        structured_query: StructuredQuery, 
        retrieved_clauses: List[RetrievedClause]
    ) -> str:
        """Prepare context string for LLM analysis."""
        context_parts = []
        
        for i, clause in enumerate(retrieved_clauses[:5], 1):  # Top 5 clauses
            context_parts.append(
                f"Clause {i} (Relevance: {clause.relevance_score:.2f}):\\n"
                f"Source: {clause.source_document}\\n"
                f"Content: {clause.content}\\n"
            )
        
        return "\\n".join(context_parts)
    
    def _format_entities(self, entities) -> str:
        """Format entities for LLM prompt."""
        if not entities:
            return "No specific entities extracted."
        
        formatted = []
        for entity in entities:
            formatted.append(f"- {entity.entity_type.title()}: {entity.value}")
        
        return "\\n".join(formatted)
    
    def _extract_decision_from_text(self, text: str) -> Decision:
        """Extract decision from unstructured text response."""
        text_lower = text.lower()
        
        # Determine decision type from text
        if "approved" in text_lower or "covered" in text_lower:
            decision_type = DecisionType.APPROVED
        elif "rejected" in text_lower or "not covered" in text_lower:
            decision_type = DecisionType.REJECTED
        elif "pending" in text_lower:
            decision_type = DecisionType.PENDING
        else:
            decision_type = DecisionType.REQUIRES_REVIEW
        
        # Extract amount if mentioned
        import re
        amount_matches = re.findall(r'₹?\\s?(\\d+(?:,\\d+)*(?:\\.\\d+)?)', text)
        amount = None
        if amount_matches:
            try:
                amount = float(amount_matches[0].replace(',', ''))
            except ValueError:
                pass
        
        return Decision(
            decision_type=decision_type,
            amount=amount,
            currency="INR",
            confidence=0.6,
            reasoning=text[:500] + "..." if len(text) > 500 else text
        )
    
    def _create_clause_references(
        self, 
        retrieved_clauses: List[RetrievedClause], 
        decision: Decision
    ) -> List[ClauseReference]:
        """
        Create clause references that support the decision.
        
        Args:
            retrieved_clauses: Retrieved clauses
            decision: Made decision
            
        Returns:
            List of clause references
        """
        references = []
        
        for i, clause in enumerate(retrieved_clauses[:3]):  # Top 3 most relevant
            # Calculate influence weight based on relevance and decision type
            influence_weight = clause.relevance_score
            
            # Adjust weight based on clause type and decision
            if decision.decision_type == DecisionType.APPROVED and clause.clause_type == "coverage":
                influence_weight *= 1.2
            elif decision.decision_type == DecisionType.REJECTED and clause.clause_type == "exclusion":
                influence_weight *= 1.2
            
            influence_weight = min(1.0, influence_weight)
            
            reference = ClauseReference(
                clause_id=clause.clause_id,
                clause_text=clause.content[:200] + "..." if len(clause.content) > 200 else clause.content,
                source_document=clause.source_document,
                influence_weight=influence_weight
            )
            references.append(reference)
        
        return references
