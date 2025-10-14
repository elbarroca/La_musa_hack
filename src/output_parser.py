"""
Output Parser Module
Extracts structured data from Symphony agent outputs for visualization and analysis
"""

import re
from typing import Dict, List, Optional, TypedDict
from enum import Enum


class RiskSeverity(Enum):
    """Risk severity levels"""
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class ParsedThreat(TypedDict):
    """Structured threat data"""
    threat: str
    evidence: str
    impact: str
    severity: str
    agent: str


class ParsedRecommendation(TypedDict):
    """Structured recommendation data"""
    title: str
    description: str
    agent: str
    priority: str


class AgentOutputParser:
    """
    Parses Symphony agent outputs into structured data for dashboard visualization.
    Extracts threats, evidence, impacts, and recommendations from formatted agent responses.
    """

    @staticmethod
    def parse_shrek_output(output: str) -> Dict:
        """
        Parse Shrek's output - OPPORTUNITIES (not threats).
        """
        opportunities = []  # Changed from threats
        action_items = []
        
        # Extract Opportunities (new format)
        major_opp_pattern = r'🟢\s*\*\*Major Opportunity \d+:\*\*\s*(.+?)(?=🟡|🔵|💡|\Z)'
        hidden_adv_pattern = r'🟡\s*\*\*Hidden Advantage \d+:\*\*\s*(.+?)(?=🔵|💡|\Z)'
        what_if_pattern = r'🔵\s*\*\*"What If" Possibility \d+:\*\*\s*(.+?)(?=💡|\Z)'
        
        for pattern, priority in [(major_opp_pattern, "High"), (hidden_adv_pattern, "Medium"), (what_if_pattern, "Low")]:
            matches = re.finditer(pattern, output, re.DOTALL | re.IGNORECASE)
            for match in matches:
                opp_text = match.group(1).strip()
                
                # Extract evidence from the opportunity text
                evidence_match = re.search(r'📄\s*\*\*Evidence from (.+?):\*\*\s*["\'](.+?)["\']', opp_text, re.DOTALL)
                evidence = evidence_match.group(2) if evidence_match else "N/A"
                source = evidence_match.group(1) if evidence_match else "N/A"
                
                opportunities.append(ParsedThreat(  # Reusing threat structure for opportunities
                    threat=opp_text.split('📄')[0].strip() if '📄' in opp_text else opp_text[:200],
                    evidence=f"[{source}] {evidence}" if evidence != "N/A" else "N/A",
                    impact="Opportunity",
                    severity=priority,
                    agent="Shrek"
                ))
        
        # Extract ACTION ITEMS
        action_pattern = r'📋\s*\*\*ACTION ITEMS\*\*.*?\n\s*(\d+)\.\s*(.+?)(?=\n\s*\d+\.|📋|🎯|\Z)'
        action_matches = re.finditer(action_pattern, output, re.DOTALL | re.IGNORECASE)
        
        for match in action_matches:
            action_items.append(ParsedRecommendation(
                title=f"Opportunity Action {match.group(1)}",
                description=match.group(2).strip(),
                agent="Shrek",
                priority="High"
            ))
        
        return {
            "threats": opportunities,  # Storing as threats for compatibility
            "action_items": action_items,
            "agent": "Shrek",
            "type": "opportunities"
        }

    @staticmethod
    def parse_sonic_output(output: str) -> Dict:
        """
        Parse Sonic's output - SPEED/LEAN focus (not UX).
        """
        speed_killers = []
        action_items = []
        
        # Extract speed killers (new format)
        bloat_pattern = r'🔴\s*\*\*Major Bloat \d+:\*\*\s*(.+?)(?=🟠|🟡|⚡|\Z)'
        waste_pattern = r'🟠\s*\*\*Waste Alert \d+:\*\*\s*(.+?)(?=🟡|⚡|\Z)'
        overthinking_pattern = r'🟡\s*\*\*Overthinking Zone \d+:\*\*\s*(.+?)(?=⚡|\Z)'
        
        for pattern, severity in [(bloat_pattern, "Critical"), (waste_pattern, "High"), (overthinking_pattern, "Medium")]:
            matches = re.finditer(pattern, output, re.DOTALL | re.IGNORECASE)
            for match in matches:
                threat_text = match.group(1).strip()
                
                # Extract evidence
                evidence_match = re.search(r'📄\s*\*\*Evidence from (.+?):\*\*\s*["\'](.+?)["\']', threat_text, re.DOTALL)
                evidence = evidence_match.group(2) if evidence_match else "N/A"
                source = evidence_match.group(1) if evidence_match else "N/A"
                
                speed_killers.append(ParsedThreat(
                    threat=threat_text.split('📄')[0].strip() if '📄' in threat_text else threat_text[:200],
                    evidence=f"[{source}] {evidence}" if evidence != "N/A" else "N/A",
                    impact="Speed/Lean Issue",
                    severity=severity,
                    agent="Sonic"
                ))
        
        # Extract ACTION ITEMS
        action_pattern = r'📋\s*\*\*ACTION ITEMS\*\*.*?\n\s*(\d+)\.\s*(.+?)(?=\n\s*\d+\.|📋|🎯|\Z)'
        action_matches = re.finditer(action_pattern, output, re.DOTALL | re.IGNORECASE)
        
        for match in action_matches:
            action_items.append(ParsedRecommendation(
                title=f"Speed Action {match.group(1)}",
                description=match.group(2).strip(),
                agent="Sonic",
                priority="High"
            ))
        
        return {
            "threats": speed_killers,
            "action_items": action_items,
            "agent": "Sonic",
            "type": "speed_lean"
        }

    @staticmethod
    def parse_hulk_output(output: str) -> Dict:
        """
        Parse Hulk's output - USER REALITY/ASSUMPTIONS focus (not competition).
        """
        user_flaws = []
        action_items = []
        
        # Extract user reality flaws (new format)
        assumption_pattern = r'🔴\s*\*\*Fatal Assumption \d+:\*\*\s*(.+?)(?=🟠|🟡|💪|\Z)'
        gap_pattern = r'🟠\s*\*\*Reality Gap \d+:\*\*\s*(.+?)(?=🟡|💪|\Z)'
        wishful_pattern = r'🟡\s*\*\*Wishful Thinking \d+:\*\*\s*(.+?)(?=💪|\Z)'
        
        for pattern, severity in [(assumption_pattern, "Critical"), (gap_pattern, "High"), (wishful_pattern, "Medium")]:
            matches = re.finditer(pattern, output, re.DOTALL | re.IGNORECASE)
            for match in matches:
                threat_text = match.group(1).strip()
                
                # Extract evidence
                evidence_match = re.search(r'📄\s*\*\*Evidence from (.+?):\*\*\s*["\'](.+?)["\']', threat_text, re.DOTALL)
                evidence = evidence_match.group(2) if evidence_match else "N/A"
                source = evidence_match.group(1) if evidence_match else "N/A"
                
                user_flaws.append(ParsedThreat(
                    threat=threat_text.split('📄')[0].strip() if '📄' in threat_text else threat_text[:200],
                    evidence=f"[{source}] {evidence}" if evidence != "N/A" else "N/A",
                    impact="User Reality Flaw",
                    severity=severity,
                    agent="Hulk"
                ))
        
        # Extract ACTION ITEMS
        action_pattern = r'📋\s*\*\*ACTION ITEMS\*\*.*?\n\s*(\d+)\.\s*(.+?)(?=\n\s*\d+\.|📋|🎯|\Z)'
        action_matches = re.finditer(action_pattern, output, re.DOTALL | re.IGNORECASE)
        
        for match in action_matches:
            action_items.append(ParsedRecommendation(
                title=f"User Reality Action {match.group(1)}",
                description=match.group(2).strip(),
                agent="Hulk",
                priority="High"
            ))
        
        return {
            "threats": user_flaws,
            "action_items": action_items,
            "agent": "Hulk",
            "type": "user_reality"
        }

    @staticmethod
    def parse_trevor_output(output: str) -> Dict:
        """
        Parse Trevor's output - updated for new CONSOLIDATED ACTION PLAN with new categories.
        
        """
        recommendations = []
        
        # Extract consolidated action items by category
        # Opportunity Capture (Shrek's Domain)
        opp_pattern = r'🌟\s*Opportunity Capture.*?\n((?:\d+\.\s*.+?\n?)+)'
        opp_match = re.search(opp_pattern, output, re.DOTALL | re.IGNORECASE)
        if opp_match:
            action_items = re.findall(r'(\d+)\.\s*(.+?)(?=\n\d+\.|\Z)', opp_match.group(1), re.DOTALL)
            for num, action in action_items:
                recommendations.append(ParsedRecommendation(
                    title=f"Opportunity Action {num}",
                    description=action.strip(),
                    agent="Trevor",
                    priority="High"
                ))
        
        # Speed & Execution (Sonic's Domain)
        speed_pattern = r'⚡\s*Speed & Execution.*?\n((?:\d+\.\s*.+?\n?)+)'
        speed_match = re.search(speed_pattern, output, re.DOTALL | re.IGNORECASE)
        if speed_match:
            action_items = re.findall(r'(\d+)\.\s*(.+?)(?=\n\d+\.|\Z)', speed_match.group(1), re.DOTALL)
            for num, action in action_items:
                recommendations.append(ParsedRecommendation(
                    title=f"Speed Action {num}",
                    description=action.strip(),
                    agent="Trevor",
                    priority="High"
                ))
        
        # User Reality Alignment (Hulk's Domain)
        user_pattern = r'💪\s*User Reality Alignment.*?\n((?:\d+\.\s*.+?\n?)+)'
        user_match = re.search(user_pattern, output, re.DOTALL | re.IGNORECASE)
        if user_match:
            action_items = re.findall(r'(\d+)\.\s*(.+?)(?=\n\d+\.|\Z)', user_match.group(1), re.DOTALL)
            for num, action in action_items:
                recommendations.append(ParsedRecommendation(
                    title=f"User Reality Action {num}",
                    description=action.strip(),
                    agent="Trevor",
                    priority="High"
                ))
        
        # Extract the final call/decision
        decision_match = re.search(r'The Final Call:\s*(.+?)(?=📋|The Integrated Reality:|\Z)', output, re.DOTALL)
        decision = decision_match.group(1).strip() if decision_match else "UNKNOWN"
        
        return {
            "recommendations": recommendations,
            "decision": decision,
            "justification": "",
            "agent": "Trevor",
            "type": "synthesis"
        }

    @staticmethod
    def parse_evaluator_output(output: str) -> Dict:
        """
        Parse Evaluator's output (QUALITY ASSESSMENT).
        Extracts quality scores and metrics for each agent.
        """
        scores = {}
        
        # Extract agent scores
        score_pattern = r'\*\*(\w+):\*\*.*?Clarity:\s*(\d+(?:\.\d+)?)/10.*?Evidence:\s*(\d+(?:\.\d+)?)/10.*?Actionability:\s*(\d+(?:\.\d+)?)/10'
        matches = re.finditer(score_pattern, output, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            agent = match.group(1)
            clarity = float(match.group(2))
            evidence = float(match.group(3))
            actionability = float(match.group(4))
            
            scores[agent] = {
                "clarity": clarity,
                "evidence": evidence,
                "actionability": actionability,
                "overall": round((clarity + evidence + actionability) / 3, 1)
            }
        
        return {
            "scores": scores,
            "agent": "Evaluator",
            "type": "quality_assessment"
        }

    @staticmethod
    def _determine_severity(text: str) -> RiskSeverity:
        """Determine risk severity based on keywords in the text."""
        text_lower = text.lower()
        
        critical_keywords = ['critical', 'catastrophic', 'fatal', 'collapse', 'severe', 'emergency']
        high_keywords = ['major', 'significant', 'serious', 'substantial', 'heavy']
        medium_keywords = ['moderate', 'considerable', 'notable', 'important']
        
        if any(kw in text_lower for kw in critical_keywords):
            return RiskSeverity.CRITICAL
        elif any(kw in text_lower for kw in high_keywords):
            return RiskSeverity.HIGH
        elif any(kw in text_lower for kw in medium_keywords):
            return RiskSeverity.MEDIUM
        else:
            return RiskSeverity.LOW

    @staticmethod
    def parse_all_outputs(conversation_history: List[str]) -> Dict:
        """
        Parse all agent outputs from conversation history.
        
        Args:
            conversation_history: List of agent outputs
            
        Returns:
            Dictionary with structured data for all agents
        """
        all_threats = []
        all_recommendations = []
        parsed_data = {
            "threats": [],
            "recommendations": [],
            "decision": None,
            "quality_scores": {},
            "summary": {}
        }
        
        for output in conversation_history:
            if "Shrek" in output:
                shrek_data = AgentOutputParser.parse_shrek_output(output)
                all_threats.extend(shrek_data.get("threats", []))
                all_recommendations.extend(shrek_data.get("action_items", []))
            
            elif "Sonic" in output:
                sonic_data = AgentOutputParser.parse_sonic_output(output)
                all_threats.extend(sonic_data.get("threats", []))
                all_recommendations.extend(sonic_data.get("action_items", []))
            
            elif "Hulk" in output:
                hulk_data = AgentOutputParser.parse_hulk_output(output)
                all_threats.extend(hulk_data.get("threats", []))
                all_recommendations.extend(hulk_data.get("action_items", []))
            
            elif "Trevor" in output:
                trevor_data = AgentOutputParser.parse_trevor_output(output)
                all_recommendations.extend(trevor_data["recommendations"])
                parsed_data["decision"] = trevor_data["decision"]
                parsed_data["justification"] = trevor_data.get("justification", "")
            
            elif "Evaluator" in output:
                eval_data = AgentOutputParser.parse_evaluator_output(output)
                parsed_data["quality_scores"] = eval_data.get("scores", {})
        
        parsed_data["threats"] = all_threats
        parsed_data["recommendations"] = all_recommendations
        
        # Calculate summary statistics
        if all_threats:
            severity_counts = {}
            for threat in all_threats:
                severity = threat["severity"]
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            parsed_data["summary"] = {
                "total_threats": len(all_threats),
                "severity_breakdown": severity_counts,
                "total_recommendations": len(all_recommendations)
            }
        
        return parsed_data

    @staticmethod
    def validate_output_format(output: str, agent_name: str) -> Dict[str, bool]:
        """
        Validate that an agent's output follows the proper format.
        
        Args:
            output: The agent's output text
            agent_name: Name of the agent (Shrek, Sonic, Hulk, Trevor)
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            "has_header": False,
            "follows_format": False,
            "has_evidence": False,
            "is_valid": False
        }
        
        # Check for proper headers (updated for new agent roles)
        header_patterns = {
            "Shrek": r"SHREK'S OPPORTUNITY MAP",
            "Sonic": r"SONIC'S LEAN EXECUTION AUDIT",
            "Hulk": r"HULK SMASH ASSUMPTIONS",
            "Trevor": r"TREVOR'S STRATEGIC SYNTHESIS",
            "Evaluator": r"EVALUATOR'S QUALITY ASSESSMENT"
        }
        
        if agent_name in header_patterns:
            validation["has_header"] = bool(re.search(header_patterns[agent_name], output, re.IGNORECASE))
        
        # Check for evidence citations (new format requirement)
        validation["has_evidence"] = bool(re.search(r'📄\s*\*\*Evidence from', output, re.IGNORECASE))
        
        # Check format compliance based on agent (updated for new roles)
        if agent_name == "Shrek":
            validation["follows_format"] = bool(re.search(r'(🟢.*Major Opportunity|🟡.*Hidden Advantage|📋.*ACTION ITEMS)', output, re.DOTALL | re.IGNORECASE))
        elif agent_name == "Sonic":
            validation["follows_format"] = bool(re.search(r'(🔴.*Major Bloat|🟠.*Waste Alert|📋.*ACTION ITEMS)', output, re.DOTALL | re.IGNORECASE))
        elif agent_name == "Hulk":
            validation["follows_format"] = bool(re.search(r'(🔴.*Fatal Assumption|🟠.*Reality Gap|📋.*ACTION ITEMS)', output, re.DOTALL | re.IGNORECASE))
        elif agent_name == "Trevor":
            validation["follows_format"] = bool(re.search(r'(CONSOLIDATED ACTION PLAN|🌟.*Opportunity|⚡.*Speed|💪.*User Reality)', output, re.DOTALL | re.IGNORECASE))
        elif agent_name == "Evaluator":
            validation["follows_format"] = bool(re.search(r'Individual Agent Scores.*Team Summary', output, re.DOTALL | re.IGNORECASE))
        
        validation["is_valid"] = validation["has_header"] and validation["follows_format"]
        
        return validation
