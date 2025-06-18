from langchain_community.llms import Ollama
from langchain.schema import HumanMessage
import re
import json
from typing import Dict, List, Any

class OllamaClient:
    def __init__(self):
        self.models = {
            "deepseek-coder": "deepseek-coder:latest",
            "mistral": "mistral:latest",
            "llama3": "llama3:latest"
        }
    
    def generate(self, query: str, context: str, model: str = "deepseek-coder") -> str:
        """Generate answer using Ollama"""
        llm = Ollama(model=self.models.get(model, model))
        
        prompt = f"""You are a medical research assistant analyzing scientific papers.
        
Context from the document:
{context}

Question: {query}

Please provide a detailed, accurate answer based on the context provided."""
        
        response = llm.invoke(prompt)
        return response
    
    def generate_with_confidence(self, query: str, context: List[Any], model: str = "deepseek-coder") -> Dict:
        """Generate answer with confidence scoring"""
        llm = Ollama(model=self.models.get(model, model))
        
        # Prepare context
        context_text = "\n\n".join([chunk.text for chunk in context[:5]])
        
        prompt = f"""You are a medical research assistant analyzing scientific papers.
        
Context from the document:
{context_text}

Question: {query}

Instructions:
1. Provide a detailed answer based ONLY on the provided context
2. After your answer, rate your confidence (0.0 to 1.0) based on:
   - How well the context addresses the question
   - Clarity of the information
   - Completeness of the answer

Format your response as:
ANSWER: [your detailed answer]
CONFIDENCE: [0.0-1.0]"""
        
        response = llm.invoke(prompt)
        
        # Parse response
        answer = response
        confidence = 0.8  # Default
        
        if "CONFIDENCE:" in response:
            parts = response.split("CONFIDENCE:")
            answer = parts[0].replace("ANSWER:", "").strip()
            try:
                confidence = float(parts[1].strip())
            except:
                confidence = 0.8
        
        return {
            "answer": answer,
            "confidence": confidence,
            "model": model
        }
    
    def extract_structured(self, text: str, schema: Dict, model: str = "deepseek-coder") -> Dict:
        """Extract structured information according to schema"""
        llm = Ollama(model=self.models.get(model, model))
        
        prompt = f"""Extract the following information from the text.
        
Text:
{text}

Extract these fields:
{json.dumps(schema, indent=2)}

Return a JSON object with the extracted information and confidence scores (0.0-1.0) for each field.
Format: {{"field_name": {{"text": "extracted text", "confidence": 0.9}}}}

If a field cannot be found, use {{"text": "Not found", "confidence": 0.0}}"""
        
        response = llm.invoke(prompt)
        
        # Try to parse JSON from response
        try:
            # Find JSON in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback
                return {field: {"text": "Extraction failed", "confidence": 0.0} for field in schema}
        except:
            return {field: {"text": "Extraction failed", "confidence": 0.0} for field in schema}