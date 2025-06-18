import re
from typing import Dict, Any
import json

class PICOTTExtractor:
    """Extract PICOTT elements with confidence scoring"""
    
    def __init__(self):
        self.components = [
            "population",
            "intervention", 
            "comparator",
            "outcome",
            "time",
            "type_of_study"
        ]
    
    def extract(self, text: str) -> Dict[str, Dict[str, Any]]:
        """Extract PICOTT components from text"""
        results = {}
        
        # Define patterns for each component
        patterns = {
            "population": [
                r"(?i)(patients?|participants?|subjects?|population).*?(?:with|diagnosed|having)\s+([^.]+)",
                r"(?i)included\s+(\d+)\s+(patients?|participants?|subjects?)",
                r"(?i)eligibility.*?criteria.*?([^.]+)"
            ],
            "intervention": [
                r"(?i)(intervention|treatment|therapy).*?(?:was|consisted|included)\s+([^.]+)",
                r"(?i)received\s+([^.]+)",
                r"(?i)experimental.*?group.*?([^.]+)"
            ],
            "comparator": [
                r"(?i)(control|comparator|comparison).*?(?:group|arm).*?([^.]+)",
                r"(?i)compared.*?(?:to|with)\s+([^.]+)",
                r"(?i)placebo.*?([^.]+)"
            ],
            "outcome": [
                r"(?i)primary.*?outcome.*?(?:was|included)\s+([^.]+)",
                r"(?i)secondary.*?outcome.*?([^.]+)",
                r"(?i)endpoint.*?([^.]+)"
            ],
            "time": [
                r"(?i)follow.?up.*?(\d+\s*(?:days?|weeks?|months?|years?))",
                r"(?i)duration.*?(\d+\s*(?:days?|weeks?|months?|years?))",
                r"(?i)(?:from|between).*?(\d{4}).*?(?:to|and).*?(\d{4})"
            ],
            "type_of_study": [
                r"(?i)(randomized|RCT|cohort|case.?control|cross.?sectional|retrospective|prospective)",
                r"(?i)study.*?design.*?([^.]+)",
                r"(?i)this.*?(randomized|observational|experimental).*?study"
            ]
        }
        
        # Extract each component
        for component in self.components:
            extracted_text = "Not found"
            confidence = 0.0
            
            # Try each pattern
            for pattern in patterns.get(component, []):
                matches = re.findall(pattern, text)
                if matches:
                    # Get the most relevant match
                    if isinstance(matches[0], tuple):
                        extracted_text = " ".join(matches[0]).strip()
                    else:
                        extracted_text = matches[0].strip()
                    
                    # Calculate confidence based on match quality
                    confidence = self._calculate_confidence(extracted_text, component)
                    break
            
            results[component] = {
                "text": extracted_text[:200] if len(extracted_text) > 200 else extracted_text,
                "confidence": confidence
            }
        
        return results
    
    def _calculate_confidence(self, text: str, component: str) -> float:
        """Calculate confidence score for extracted text"""
        if text == "Not found":
            return 0.0
        
        # Base confidence
        confidence = 0.5
        
        # Increase confidence for specific indicators
        if component == "population" and any(word in text.lower() for word in ["patients", "participants", "subjects"]):
            confidence += 0.2
        elif component == "intervention" and any(word in text.lower() for word in ["treatment", "therapy", "intervention"]):
            confidence += 0.2
        elif component == "type_of_study" and any(word in text.lower() for word in ["randomized", "rct", "cohort"]):
            confidence += 0.3
        
        # Increase confidence for numeric data
        if re.search(r'\d+', text):
            confidence += 0.1
        
        # Cap at 0.95
        return min(confidence, 0.95)


class BiasAssessmentExtractor:
    """Extract risk of bias assessments"""
    
    def __init__(self):
        self.domains = [
            "random_sequence_generation",
            "allocation_concealment",
            "blinding_participants",
            "blinding_assessment",
            "incomplete_outcome",
            "selective_reporting",
            "other_bias"
        ]
    
    def extract(self, text: str) -> Dict[str, Dict[str, Any]]:
        """Extract bias assessment from text"""
        results = {}
        
        # Keywords for each domain
        keywords = {
            "random_sequence_generation": ["randomization", "random sequence", "randomized"],
            "allocation_concealment": ["allocation", "concealment", "sealed envelopes"],
            "blinding_participants": ["blinding", "blind", "masked", "double-blind"],
            "blinding_assessment": ["outcome assessment", "assessor blinding", "evaluator blind"],
            "incomplete_outcome": ["lost to follow-up", "attrition", "dropout", "missing data"],
            "selective_reporting": ["protocol", "pre-registered", "all outcomes reported"],
            "other_bias": ["baseline", "imbalance", "conflict of interest", "funding"]
        }
        
        for domain in self.domains:
            # Search for relevant text
            risk_level = "Unclear risk"
            justification = "No clear information found"
            confidence = 0.3
            
            for keyword in keywords.get(domain, []):
                if keyword.lower() in text.lower():
                    # Found relevant text, analyze context
                    context = self._extract_context(text, keyword)
                    risk_level, justification = self._assess_risk(context, domain)
                    confidence = 0.7
                    break
            
            results[domain] = {
                "risk_level": risk_level,
                "justification": justification,
                "confidence": confidence
            }
        
        return results
    
    def _extract_context(self, text: str, keyword: str, window: int = 100) -> str:
        """Extract context around keyword"""
        keyword_pos = text.lower().find(keyword.lower())
        if keyword_pos == -1:
            return ""
        
        start = max(0, keyword_pos - window)
        end = min(len(text), keyword_pos + len(keyword) + window)
        return text[start:end]
    
    def _assess_risk(self, context: str, domain: str) -> tuple:
        """Assess risk level based on context"""
        context_lower = context.lower()
        
        # Positive indicators (low risk)
        if any(word in context_lower for word in ["properly", "adequate", "appropriate", "successful"]):
            return "Low risk", f"Adequate {domain.replace('_', ' ')} described"
        
        # Negative indicators (high risk)
        if any(word in context_lower for word in ["not", "inadequate", "failed", "unclear"]):
            return "High risk", f"Inadequate {domain.replace('_', ' ')}"
        
        return "Unclear risk", "Insufficient information to assess"