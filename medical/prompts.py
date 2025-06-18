class MedicalPrompts:
    """Medical research-specific prompts for systematic reviews"""
    
    def get_picott_prompt(self):
        return """Extract the PICOTT (Population, Intervention, Comparator, Outcome, Time, Type of Study) from this research paper.
        
        For each component, provide:
        - Population: Patient characteristics, inclusion/exclusion criteria
        - Intervention: Treatment or exposure being studied
        - Comparator: Control group or comparison treatment
        - Outcome: Primary and secondary outcomes measured
        - Time: Follow-up period, study duration
        - Type: Study design (RCT, cohort, case-control, etc.)
        
        Be specific and include sample sizes where mentioned."""
    
    def get_bias_prompt(self):
        return """Assess the risk of bias in this study according to the Cochrane Risk of Bias tool.
        
        Evaluate:
        1. Random sequence generation (selection bias)
        2. Allocation concealment (selection bias)
        3. Blinding of participants and personnel (performance bias)
        4. Blinding of outcome assessment (detection bias)
        5. Incomplete outcome data (attrition bias)
        6. Selective reporting (reporting bias)
        7. Other bias
        
        For each domain, indicate: Low risk, High risk, or Unclear risk, with justification."""
    
    def get_outcomes_prompt(self):
        return """Extract all primary and secondary outcomes from this study.
        
        For each outcome, identify:
        - Outcome name and definition
        - How it was measured
        - Time points of measurement
        - Statistical results (effect size, confidence intervals, p-values)
        - Clinical significance
        
        Organize by primary vs secondary outcomes."""
    
    def get_adverse_prompt(self):
        return """Extract all adverse events and safety data from this study.
        
        Include:
        - Types of adverse events reported
        - Frequency/incidence in each group
        - Severity grading
        - Serious adverse events
        - Discontinuations due to adverse events
        - Deaths or life-threatening events
        
        Focus only on the results and safety sections."""
    
    def get_statistics_prompt(self):
        return """Extract key statistical information:
        
        - Sample size calculation and power analysis
        - Statistical tests used
        - Primary outcome results with confidence intervals
        - P-values for main comparisons
        - Subgroup analyses
        - Sensitivity analyses
        - Missing data handling
        
        Include actual numbers, not just descriptions."""
    
    def get_inclusion_exclusion_prompt(self):
        return """Extract the complete inclusion and exclusion criteria.
        
        List separately:
        
        INCLUSION CRITERIA:
        - Age requirements
        - Diagnosis/condition requirements
        - Other inclusion criteria
        
        EXCLUSION CRITERIA:
        - Medical conditions
        - Medications
        - Other exclusions
        
        Be comprehensive and specific."""
    
    def get_custom_prompt(self, template: str, variables: dict):
        """Create custom prompt from template"""
        return template.format(**variables)