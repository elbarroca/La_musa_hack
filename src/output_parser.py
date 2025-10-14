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
        major_opp_pattern = r'🟢\s*Major Opportunity \d+:\s*(.+?)(?=🟡|🔵|💡|\Z)'
        hidden_adv_pattern = r'🟡\s*Hidden Advantage \d+:\s*(.+?)(?=🔵|💡|\Z)'
        what_if_pattern = r'🔵\s*"What If" Possibility \d+:\s*(.+?)(?=💡|\Z)'
        
        for pattern, priority in [(major_opp_pattern, "High"), (hidden_adv_pattern, "Medium"), (what_if_pattern, "Low")]:
            matches = re.finditer(pattern, output, re.DOTALL | re.IGNORECASE)
            for match in matches:
                opp_text = match.group(1).strip()
                
                # Extract evidence from the opportunity text
                evidence_match = re.search(r'📄\s*Evidence from (.+?):\s*["\'](.+?)["\']', opp_text, re.DOTALL)
                evidence = evidence_match.group(2) if evidence_match else "N/A"
                source = evidence_match.group(1) if evidence_match else "N/A"
                
                opportunities.append(ParsedThreat(  # Reusing threat structure for opportunities
                    threat=opp_text.split('📄')[0].strip() if '📄' in opp_text else opp_text[:200],
                    evidence=f"[{source}] {evidence}" if evidence != "N/A" else "N/A",
                    impact="Opportunity",
                    severity=priority,
                    agent="Shrek"
                ))
        
        # Extract ACTION ITEMS - more flexible pattern
        action_pattern = r'📋\s*ACTION ITEMS[^:]*:(.*?)(?:>|$)'
        action_match = re.search(action_pattern, output, re.DOTALL | re.IGNORECASE)
        action_items_text = action_match.group(1) if action_match else ""
        
        # If still no match, try broader pattern
        if not action_items_text:
            action_pattern_alt = r'ACTION ITEMS.*?\n(.*?)(?:\n\n>|\Z)'
            action_match_alt = re.search(action_pattern_alt, output, re.DOTALL | re.IGNORECASE)
            action_items_text = action_match_alt.group(1) if action_match_alt else ""

        # Extract numbered action items (1. 2. 3.)
        numbered_items = re.findall(r'\d+\.\s*(.+?)(?=\d+\.|$)', action_items_text, re.DOTALL)
        
        for i, item_text in enumerate(numbered_items):
            # Clean up the text
            cleaned = item_text.strip()
            if cleaned:
                action_items.append(ParsedRecommendation(
                    title=f"Opportunity Action {i+1}",
                    description=cleaned,
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
        bloat_pattern = r'🔴\s*Major Bloat \d+:\s*(.+?)(?=🟠|🟡|⚡|\Z)'
        waste_pattern = r'🟠\s*Waste Alert \d+:\s*(.+?)(?=🟡|⚡|\Z)'
        overthinking_pattern = r'🟡\s*Overthinking Zone \d+:\s*(.+?)(?=⚡|\Z)'
        
        for pattern, severity in [(bloat_pattern, "Critical"), (waste_pattern, "High"), (overthinking_pattern, "Medium")]:
            matches = re.finditer(pattern, output, re.DOTALL | re.IGNORECASE)
            for match in matches:
                threat_text = match.group(1).strip()
                
                # Extract evidence
                evidence_match = re.search(r'📄\s*Evidence from (.+?):\s*["\'](.+?)["\']', threat_text, re.DOTALL)
                evidence = evidence_match.group(2) if evidence_match else "N/A"
                source = evidence_match.group(1) if evidence_match else "N/A"
                
                speed_killers.append(ParsedThreat(
                    threat=threat_text.split('📄')[0].strip() if '📄' in threat_text else threat_text[:200],
                    evidence=f"[{source}] {evidence}" if evidence != "N/A" else "N/A",
                    impact="Speed/Lean Issue",
                    severity=severity,
                    agent="Sonic"
                ))
        
        # Extract ACTION ITEMS - more flexible pattern
        action_pattern = r'📋\s*ACTION ITEMS[^:]*:(.*?)(?:>|$)'
        action_match = re.search(action_pattern, output, re.DOTALL | re.IGNORECASE)
        action_items_text = action_match.group(1) if action_match else ""
        
        # If still no match, try broader pattern
        if not action_items_text:
            action_pattern_alt = r'ACTION ITEMS.*?\n(.*?)(?:\n\n>|\Z)'
            action_match_alt = re.search(action_pattern_alt, output, re.DOTALL | re.IGNORECASE)
            action_items_text = action_match_alt.group(1) if action_match_alt else ""

        # Extract numbered action items (1. 2. 3.)
        numbered_items = re.findall(r'\d+\.\s*(.+?)(?=\d+\.|$)', action_items_text, re.DOTALL)
        
        for i, item_text in enumerate(numbered_items):
            # Clean up the text
            cleaned = item_text.strip()
            if cleaned:
                action_items.append(ParsedRecommendation(
                    title=f"Speed Action {i+1}",
                    description=cleaned,
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
        assumption_pattern = r'🔴\s*Fatal Assumption \d+:\s*(.+?)(?=🟠|🟡|💪|\Z)'
        gap_pattern = r'🟠\s*Reality Gap \d+:\s*(.+?)(?=🟡|💪|\Z)'
        wishful_pattern = r'🟡\s*Wishful Thinking \d+:\s*(.+?)(?=💪|\Z)'
        
        for pattern, severity in [(assumption_pattern, "Critical"), (gap_pattern, "High"), (wishful_pattern, "Medium")]:
            matches = re.finditer(pattern, output, re.DOTALL | re.IGNORECASE)
            for match in matches:
                threat_text = match.group(1).strip()
                
                # Extract evidence
                evidence_match = re.search(r'📄\s*Evidence from (.+?):\s*["\'](.+?)["\']', threat_text, re.DOTALL)
                evidence = evidence_match.group(2) if evidence_match else "N/A"
                source = evidence_match.group(1) if evidence_match else "N/A"
                
                user_flaws.append(ParsedThreat(
                    threat=threat_text.split('📄')[0].strip() if '📄' in threat_text else threat_text[:200],
                    evidence=f"[{source}] {evidence}" if evidence != "N/A" else "N/A",
                    impact="User Reality Flaw",
                    severity=severity,
                    agent="Hulk"
                ))
        
        # Extract ACTION ITEMS - more flexible pattern
        action_pattern = r'📋\s*ACTION ITEMS[^:]*:(.*?)(?:>|$)'
        action_match = re.search(action_pattern, output, re.DOTALL | re.IGNORECASE)
        action_items_text = action_match.group(1) if action_match else ""
        
        # If still no match, try broader pattern
        if not action_items_text:
            action_pattern_alt = r'ACTION ITEMS.*?\n(.*?)(?:\n\n>|\Z)'
            action_match_alt = re.search(action_pattern_alt, output, re.DOTALL | re.IGNORECASE)
            action_items_text = action_match_alt.group(1) if action_match_alt else ""

        # Extract numbered action items (1. 2. 3.)
        numbered_items = re.findall(r'\d+\.\s*(.+?)(?=\d+\.|$)', action_items_text, re.DOTALL)
        
        for i, item_text in enumerate(numbered_items):
            # Clean up the text
            cleaned = item_text.strip()
            if cleaned:
                action_items.append(ParsedRecommendation(
                    title=f"User Reality Action {i+1}",
                    description=cleaned,
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
        
        # Extract all consolidated action items
        # Find all items in the CONSOLIDATED ACTION PLAN section
        action_plan_pattern = r'📋\s*CONSOLIDATED ACTION PLAN\s*(.*?)(?=The Integrated Reality:|🔍|\Z)'
        action_plan_match = re.search(action_plan_pattern, output, re.DOTALL | re.IGNORECASE)

        if action_plan_match:
            action_plan_text = action_plan_match.group(1)

            # Extract all items from each domain section
            domain_sections = [
                (r'🌟\s*Opportunity Capture.*?\n(.*?)(?=⚡|\Z)', "Opportunity"),
                (r'⚡\s*Speed & Execution.*?\n(.*?)(?=💪|\Z)', "Speed"),
                (r'💪\s*User Reality Alignment.*?\n(.*?)(?=\Z)', "User")
            ]

            for section_pattern, domain in domain_sections:
                section_match = re.search(section_pattern, action_plan_text, re.DOTALL | re.IGNORECASE)
                if section_match:
                    section_text = section_match.group(1)
                    # Extract individual items (lines that start with a dash or contain " - Owner:")
                    items = re.findall(r'(.+?)\s*-\s*Owner:', section_text)

                    for i, item in enumerate(items):
                        title_prefix = {
                            "Opportunity": "Opportunity Action",
                            "Speed": "Speed Action",
                            "User": "User Reality Action"
                        }.get(domain, "Action")

                        recommendations.append(ParsedRecommendation(
                            title=f"{title_prefix} {i+1}",
                            description=item.strip(),
                            agent="Trevor",
                            priority="High"
                        ))
        
        # Extract the final call/decision - multiple patterns
        decision = None
        justification = ""
        
        # Pattern 1: **Decision:** PROCEED/PIVOT/ABORT
        decision_match = re.search(r'\*\*Decision:\*\*\s*(PROCEED|PIVOT|ABORT)', output, re.IGNORECASE)
        if decision_match:
            decision = decision_match.group(1).upper()
        
        # Pattern 2: Look for decision in "The Final Call" section
        if not decision:
            final_call_match = re.search(r'The Final Call:.*?(?=📋|The Complete|$)', output, re.DOTALL | re.IGNORECASE)
            if final_call_match:
                final_call_text = final_call_match.group(0)
                if re.search(r'\b(PROCEED|GO AHEAD|MOVE FORWARD)\b', final_call_text, re.IGNORECASE):
                    decision = "PROCEED"
                elif re.search(r'\b(PIVOT|ADJUST|MODIFY)\b', final_call_text, re.IGNORECASE):
                    decision = "PIVOT"
                elif re.search(r'\b(ABORT|STOP|HALT)\b', final_call_text, re.IGNORECASE):
                    decision = "ABORT"
        
        # Pattern 3: Scan entire output for decision keywords
        if not decision:
            if re.search(r'(Decision|Recommendation|Final Call).*?(PROCEED|GO)', output, re.DOTALL | re.IGNORECASE):
                decision = "PROCEED"
            elif re.search(r'(Decision|Recommendation|Final Call).*?(PIVOT)', output, re.DOTALL | re.IGNORECASE):
                decision = "PIVOT"
            elif re.search(r'(Decision|Recommendation|Final Call).*?(ABORT)', output, re.DOTALL | re.IGNORECASE):
                decision = "ABORT"
        
        # Extract justification
        justification_match = re.search(r'\*\*Justification:\*\*\s*(.+?)(?=\*\*|📋|The Complete|\Z)', output, re.DOTALL)
        if justification_match:
            justification = justification_match.group(1).strip()
        elif decision:
            # Try to extract context around decision
            decision_context = re.search(rf'{decision}[^.]*\.[^.]*\.', output, re.IGNORECASE)
            if decision_context:
                justification = decision_context.group(0).strip()
        
        return {
            "recommendations": recommendations,
            "decision": decision,
            "justification": justification,
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
        score_pattern = r'(\w+)\'s Analysis:.*?Clarity:\s*(\d+(?:\.\d+)?)/10.*?Evidence:\s*(\d+(?:\.\d+)?)/10.*?Actionability:\s*(\d+(?:\.\d+)?)/10'
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
        print("\n=== STARTING OUTPUT PARSING ===")
        print(f"Total outputs to parse: {len(conversation_history)}")
        
        all_threats = []
        all_recommendations = []
        parsed_data = {
            "threats": [],
            "recommendations": [],
            "decision": None,
            "justification": "",
            "quality_scores": {},
            "summary": {}
        }
        
        for i, output in enumerate(conversation_history, 1):
            print(f"\n--- Parsing output {i} ---")
            output_preview = output[:200].replace('\n', ' ')
            print(f"Preview: {output_preview}...")
            
            try:
                # Use more robust agent identification
                if "🎯 Shrek" in output or "SHREK'S OPPORTUNITY MAP" in output or "Shrek" in output:
                    print("  → Identified as SHREK output")
                    shrek_data = AgentOutputParser.parse_shrek_output(output)
                    threats_found = len(shrek_data.get("threats", []))
                    actions_found = len(shrek_data.get("action_items", []))
                    print(f"  → Extracted: {threats_found} opportunities, {actions_found} actions")
                    if threats_found > 0:
                        print(f"     Sample opportunity: {shrek_data['threats'][0]['threat'][:80]}...")
                    if actions_found > 0:
                        print(f"     Sample action: {shrek_data['action_items'][0]['description'][:80]}...")
                    all_threats.extend(shrek_data.get("threats", []))
                    all_recommendations.extend(shrek_data.get("action_items", []))

                elif "🎯 Sonic" in output or "SONIC'S LEAN EXECUTION AUDIT" in output or "Sonic" in output:
                    print("  → Identified as SONIC output")
                    sonic_data = AgentOutputParser.parse_sonic_output(output)
                    threats_found = len(sonic_data.get("threats", []))
                    actions_found = len(sonic_data.get("action_items", []))
                    print(f"  → Extracted: {threats_found} speed issues, {actions_found} actions")
                    if threats_found > 0:
                        print(f"     Sample issue: {sonic_data['threats'][0]['threat'][:80]}...")
                    if actions_found > 0:
                        print(f"     Sample action: {sonic_data['action_items'][0]['description'][:80]}...")
                    all_threats.extend(sonic_data.get("threats", []))
                    all_recommendations.extend(sonic_data.get("action_items", []))

                elif "🎯 Hulk" in output or "HULK SMASH ASSUMPTIONS" in output or "Hulk" in output:
                    print("  → Identified as HULK output")
                    hulk_data = AgentOutputParser.parse_hulk_output(output)
                    threats_found = len(hulk_data.get("threats", []))
                    actions_found = len(hulk_data.get("action_items", []))
                    print(f"  → Extracted: {threats_found} assumption flaws, {actions_found} actions")
                    if threats_found > 0:
                        print(f"     Sample flaw: {hulk_data['threats'][0]['threat'][:80]}...")
                    if actions_found > 0:
                        print(f"     Sample action: {hulk_data['action_items'][0]['description'][:80]}...")
                    all_threats.extend(hulk_data.get("threats", []))
                    all_recommendations.extend(hulk_data.get("action_items", []))

                elif "🎯 Trevor" in output or "TREVOR'S STRATEGIC SYNTHESIS" in output or "Trevor" in output:
                    print("  → Identified as TREVOR output")
                    trevor_data = AgentOutputParser.parse_trevor_output(output)
                    recs_found = len(trevor_data.get("recommendations", []))
                    decision = trevor_data.get("decision", "UNKNOWN")
                    print(f"  → Extracted: {recs_found} recommendations, decision: {decision}")
                    if recs_found > 0:
                        print(f"     Sample recommendation: {trevor_data['recommendations'][0]['description'][:80]}...")
                    if decision and decision != "UNKNOWN":
                        print(f"     Decision justification: {trevor_data.get('justification', 'N/A')[:100]}...")
                    all_recommendations.extend(trevor_data.get("recommendations", []))
                    if trevor_data.get("decision"):
                        parsed_data["decision"] = trevor_data["decision"]
                        parsed_data["justification"] = trevor_data.get("justification", "")

                elif "🎯 Evaluator" in output or "EVALUATOR'S QUALITY ASSESSMENT" in output:
                    print("  → Identified as EVALUATOR output")
                    eval_data = AgentOutputParser.parse_evaluator_output(output)
                    scores_found = len(eval_data.get("scores", {}))
                    print(f"  → Extracted: {scores_found} agent quality scores")
                    parsed_data["quality_scores"] = eval_data.get("scores", {})
                else:
                    print("  → WARNING: Could not identify agent type")
                    
            except Exception as e:
                print(f"  → ERROR parsing output: {e}")
                import traceback
                traceback.print_exc()
        
        parsed_data["threats"] = all_threats
        parsed_data["recommendations"] = all_recommendations
        
        print(f"\n=== PARSING COMPLETE ===")
        print(f"Total threats: {len(all_threats)}")
        print(f"Total recommendations: {len(all_recommendations)}")
        print(f"Decision: {parsed_data.get('decision', 'None')}")
        print(f"Quality scores: {len(parsed_data.get('quality_scores', {}))}")
        print("===========================\n")
        
        # Calculate summary statistics
        if all_threats:
            severity_counts = {}
            for threat in all_threats:
                severity = threat.get("severity", "Unknown")
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
        validation["has_evidence"] = bool(re.search(r'📄\s*Evidence from', output, re.IGNORECASE))
        
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
