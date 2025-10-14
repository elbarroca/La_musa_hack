# app.py
"""
Symphony Enterprise - Multi-Agent Strategic Analysis Platform
Enhanced with RAG, LangChain, LangGraph, and Structured Output Parsing
"""

import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
import tempfile
import os
from dotenv import load_dotenv

from src.orchestrator import Orchestrator
from src.output_parser import AgentOutputParser

# --- Page & App Setup ---
load_dotenv()
st.set_page_config(
    page_title="Symphony Enterprise",
    page_icon="🎼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for better styling ---
st.markdown("""
<style>
    .stAlert {
        margin-top: 1rem;
    }
    .upload-section {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .threat-card {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid;
    }
    .critical { border-left-color: #dc3545; background-color: #f8d7da; }
    .high { border-left-color: #fd7e14; background-color: #ffe5d0; }
    .medium { border-left-color: #ffc107; background-color: #fff3cd; }
    .low { border-left-color: #28a745; background-color: #d4edda; }
    .recommendation-card {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid #0066cc;
    }
    .quality-score {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
    }
    .score-excellent { color: #28a745; }
    .score-good { color: #17a2b8; }
    .score-fair { color: #ffc107; }
    .score-poor { color: #dc3545; }
</style>
""", unsafe_allow_html=True)


# --- Initialize Session State ---
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'parsed_data' not in st.session_state:
    st.session_state.parsed_data = None
if 'uploaded_files_count' not in st.session_state:
    st.session_state.uploaded_files_count = 0


# --- Helper Functions ---
def get_severity_emoji(severity: str) -> str:
    """Get emoji for risk severity level."""
    severity_map = {
        "Critical": "🔴",
        "High": "🟠",
        "Medium": "🟡",
        "Low": "🟢"
    }
    return severity_map.get(severity, "⚪")


def get_score_class(score: float) -> str:
    """Get CSS class for quality score."""
    if score >= 8.5:
        return "score-excellent"
    elif score >= 7.0:
        return "score-good"
    elif score >= 5.0:
        return "score-fair"
    else:
        return "score-poor"


def render_executive_summary(parsed_data: dict):
    """Render the executive summary section."""
    st.header("📋 Executive Summary")
    
    # Decision banner
    decision = parsed_data.get("decision", "UNKNOWN")
    justification = parsed_data.get("justification", "")
    
    decision_colors = {
        "PROCEED": "🟢",
        "PIVOT": "🟡",
        "ABORT": "🔴"
    }
    
    st.markdown(f"### {decision_colors.get(decision, '⚪')} Strategic Decision: **{decision}**")
    if justification:
        st.markdown(f"> {justification}")
    
    st.markdown("---")
    
    # Summary metrics
    summary = parsed_data.get("summary", {})
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_threats = summary.get("total_threats", 0)
        st.metric("Total Threats Identified", total_threats)
    
    with col2:
        total_recs = summary.get("total_recommendations", 0)
        st.metric("Action Items", total_recs)
    
    with col3:
        severity_breakdown = summary.get("severity_breakdown", {})
        critical_count = severity_breakdown.get("Critical", 0)
        st.metric("Critical Risks", critical_count, delta="High Priority" if critical_count > 0 else None)
    
    with col4:
        quality_scores = parsed_data.get("quality_scores", {})
        if quality_scores:
            avg_score = sum(agent["overall"] for agent in quality_scores.values()) / len(quality_scores)
            st.metric("Team Quality Score", f"{avg_score:.1f}/10")
        else:
            st.metric("Team Quality Score", "N/A")


def render_threats_analysis(parsed_data: dict):
    """Render threats and risks analysis."""
    st.header("❌ Threats & Risks Analysis")
    
    threats = parsed_data.get("threats", [])
    
    if not threats:
        st.info("No threats identified in the analysis.")
        return
    
    # Severity breakdown
    severity_counts = {}
    for threat in threats:
        severity = threat["severity"]
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
    
    # Display severity summary
    cols = st.columns(4)
    for i, (severity, count) in enumerate([("Critical", 0), ("High", 0), ("Medium", 0), ("Low", 0)]):
        actual_count = severity_counts.get(severity, count)
        with cols[i]:
            st.markdown(f"{get_severity_emoji(severity)} **{severity}**: {actual_count}")
    
    st.markdown("---")
    
    # Display threats grouped by agent
    for agent_name in ["Shrek", "Sonic", "Hulk"]:
        agent_threats = [t for t in threats if t["agent"] == agent_name]
        if agent_threats:
            st.subheader(f"🎯 {agent_name}'s Findings")
            
            for threat in agent_threats:
                severity_class = threat["severity"].lower()
                st.markdown(f"""
                <div class="threat-card {severity_class}">
                    <strong>{get_severity_emoji(threat["severity"])} {threat["threat"]}</strong><br>
                    <em>Evidence:</em> {threat["evidence"]}<br>
                    <em>Impact:</em> {threat["impact"]}
                </div>
                """, unsafe_allow_html=True)


def render_recommendations(parsed_data: dict):
    """Render action items and recommendations."""
    st.header("📋 Action Items & Recommendations")
    
    recommendations = parsed_data.get("recommendations", [])
    
    if not recommendations:
        st.info("No specific recommendations provided.")
        return
    
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"""
        <div class="recommendation-card">
            <strong>#{i}: {rec["title"]}</strong><br>
            {rec["description"]}
        </div>
        """, unsafe_allow_html=True)


def render_quality_dashboard(parsed_data: dict):
    """Render quality metrics dashboard."""
    st.header("🏆 Quality Assessment Dashboard")
    
    quality_scores = parsed_data.get("quality_scores", {})
    
    if not quality_scores:
        st.info("Quality assessment not available. Ensure the Evaluator agent is running.")
        return
    
    # Overall team metrics
    avg_clarity = sum(agent["clarity"] for agent in quality_scores.values()) / len(quality_scores)
    avg_evidence = sum(agent["evidence"] for agent in quality_scores.values()) / len(quality_scores)
    avg_actionability = sum(agent["actionability"] for agent in quality_scores.values()) / len(quality_scores)
    team_avg = sum(agent["overall"] for agent in quality_scores.values()) / len(quality_scores)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'<div class="quality-score {get_score_class(team_avg)}">{team_avg:.1f}</div>', unsafe_allow_html=True)
        st.caption("Team Average")
    
    with col2:
        st.markdown(f'<div class="quality-score {get_score_class(avg_clarity)}">{avg_clarity:.1f}</div>', unsafe_allow_html=True)
        st.caption("Clarity")
    
    with col3:
        st.markdown(f'<div class="quality-score {get_score_class(avg_evidence)}">{avg_evidence:.1f}</div>', unsafe_allow_html=True)
        st.caption("Evidence")
    
    with col4:
        st.markdown(f'<div class="quality-score {get_score_class(avg_actionability)}">{avg_actionability:.1f}</div>', unsafe_allow_html=True)
        st.caption("Actionability")
    
    st.markdown("---")
    
    # Individual agent scores
    st.subheader("Agent Performance Breakdown")
    
    # Create DataFrame for visualization
    scores_data = []
    for agent_name, scores in quality_scores.items():
        scores_data.append({
            "Agent": agent_name,
            "Clarity": scores["clarity"],
            "Evidence": scores["evidence"],
            "Actionability": scores["actionability"],
            "Overall": scores["overall"]
        })
    
    if scores_data:
        df = pd.DataFrame(scores_data)
        
        # Melt for better visualization
        df_melted = df.melt(id_vars=["Agent"], 
                           value_vars=["Clarity", "Evidence", "Actionability"],
                           var_name="Metric", value_name="Score")
        
        # Create grouped bar chart
        chart = alt.Chart(df_melted).mark_bar().encode(
            x=alt.X('Agent:N', title='Agent'),
            y=alt.Y('Score:Q', scale=alt.Scale(domain=[0, 10]), title='Score (0-10)'),
            color=alt.Color('Metric:N', legend=alt.Legend(title="Quality Metric")),
            xOffset='Metric:N'
        ).properties(
            height=300,
            title="Agent Quality Metrics"
        )
        
        st.altair_chart(chart, use_container_width=True)
        
        # Best performer
        best_agent = max(quality_scores.items(), key=lambda x: x[1]["overall"])
        st.success(f"🏆 **Top Performer**: {best_agent[0]} with overall score of {best_agent[1]['overall']:.1f}/10")




# --- Sidebar: Knowledge Base Management ---
with st.sidebar:
    st.title("🗂️ Knowledge Base")
    st.markdown("Upload company documents to ground the analysis in your context.")
    st.markdown("---")

    # File Upload Section
    st.subheader("📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or DOCX files",
        type=['pdf', 'txt', 'docx'],
        accept_multiple_files=True,
        help="Upload company documents, policies, or research to inform the analysis"
    )

    if uploaded_files and st.button("🔄 Process Documents", type="primary"):
        with st.spinner("Processing uploaded documents..."):
            try:
                # Initialize orchestrator if needed
                if st.session_state.orchestrator is None:
                    st.session_state.orchestrator = Orchestrator(
                        agent_config_path="config/agents.yaml"
                    )

                # Save uploaded files temporarily
                temp_files = []
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        temp_files.append(tmp.name)

                # Add to knowledge base
                chunks_added = st.session_state.orchestrator.add_documents_to_kb(temp_files)

                # Clean up temp files
                for tmp_file in temp_files:
                    os.unlink(tmp_file)

                st.session_state.uploaded_files_count = len(uploaded_files)
                st.success(f"✅ Processed {len(uploaded_files)} documents ({chunks_added} chunks)")

            except Exception as e:
                st.error(f"❌ Error processing documents: {e}")

    # Show KB status
    if st.session_state.orchestrator is not None:
        kb = st.session_state.orchestrator.get_knowledge_base()
        try:
            doc_count = kb.get_document_count()
            st.info(f"📊 Knowledge Base: {doc_count} chunks indexed")
        except:
            pass

    # Clear KB button
    if st.button("🗑️ Clear Knowledge Base"):
        if st.session_state.orchestrator is not None:
            st.session_state.orchestrator.clear_knowledge_base()
            st.session_state.uploaded_files_count = 0
            st.success("Knowledge base cleared")
            st.rerun()

    st.markdown("---")

    # API Configuration Status
    st.subheader("⚙️ Configuration")
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and api_key != "your-openai-api-key-here":
        st.success("✅ OpenAI API Key configured")
    else:
        st.error("❌ OpenAI API Key not set")
        st.caption("Set OPENAI_API_KEY in your .env file")


# --- Main UI ---
st.title("🎼 Symphony Enterprise v2.0")
st.markdown("""
**Your AI strategy team, powered by LangChain & LangGraph**  
Get a comprehensive pre-mortem analysis from five specialized AI agents with structured insights.
""")

# --- User Input Section ---
st.markdown("### 📝 Strategic Plan Input")
product_idea = st.text_area(
    "Enter your product idea or strategic plan for analysis:",
    height=150,
    placeholder="e.g., Launch a new multi-tenant SaaS offering on Azure for supply chain logistics, using Cosmos DB, AKS, and Entra ID B2C.",
    help="Provide as much detail as possible for a thorough analysis"
)

# --- Analysis Button ---
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    analyze_button = st.button(
        "🚀 Run Pre-Mortem Analysis",
        type="primary",
        use_container_width=True,
        disabled=not product_idea.strip()
    )

# --- Main Analysis Logic ---
if analyze_button:
    if not product_idea.strip():
        st.warning("⚠️ Please enter a product idea before running the analysis.")
    else:
        try:
            # Initialize orchestrator
            if st.session_state.orchestrator is None:
                with st.spinner("Initializing Symphony with production RAG..."):
                    st.session_state.orchestrator = Orchestrator(
                        agent_config_path="config/agents.yaml"
                    )

            st.session_state.analysis_complete = False
            st.session_state.conversation_history = []
            st.session_state.parsed_data = None

            # Create tabs for output
            tab1, tab2 = st.tabs([
                "📊 Executive Dashboard",
                "💬 Full Analysis"
            ])

            # --- Tab 2: Full Analysis (Real-time) ---
            with tab2:
                st.header("🎙️ Multi-Agent Analysis")
                st.markdown("Watch as each agent provides their analysis in real-time...")
                st.markdown("---")

                # Container for streaming output
                output_container = st.container()

                with output_container:
                    # Stream the analysis
                    current_agent = None
                    agent_placeholder = None
                    accumulated_text = ""

                    for agent_name, chunk in st.session_state.orchestrator.run_analysis_streaming(product_idea, max_retries=2):
                        if agent_name != current_agent:
                            current_agent = agent_name
                            agent_placeholder = st.empty()
                            accumulated_text = ""

                        accumulated_text += chunk
                        agent_placeholder.markdown(accumulated_text)

                        # Store for history
                        if agent_name not in ["System", "Error"]:
                            # Update conversation history
                            # Use try-except to handle orchestrator instances without get_agent_count method
                            try:
                                agent_count = st.session_state.orchestrator.get_agent_count()
                            except AttributeError:
                                agent_count = 5  # Default to 5 agents (Shrek, Sonic, Hulk, Trevor, Evaluator)
                            
                            if len(st.session_state.conversation_history) < agent_count:
                                st.session_state.conversation_history.append(accumulated_text)
                            else:
                                st.session_state.conversation_history[-1] = accumulated_text

                # Parse outputs after completion
                st.session_state.parsed_data = st.session_state.orchestrator.parse_conversation_history(
                    st.session_state.conversation_history
                )
                st.session_state.analysis_complete = True
                st.success("✅ Analysis complete - view structured insights in other tabs")

            # --- Tab 1: Executive Dashboard ---
            with tab1:
                if st.session_state.parsed_data:
                    render_executive_summary(st.session_state.parsed_data)
                    st.markdown("---")
                    render_threats_analysis(st.session_state.parsed_data)
                    st.markdown("---")
                    render_recommendations(st.session_state.parsed_data)
                    st.markdown("---")
                    render_quality_dashboard(st.session_state.parsed_data)
                else:
                    st.info("Run the analysis to see the executive dashboard")


        except Exception as e:
            st.error(f"❌ Error during analysis: {e}")
            import traceback
            st.code(traceback.format_exc())

# --- Display Results if Available ---
elif st.session_state.analysis_complete and st.session_state.parsed_data:
    # Show tabs even after page reload
    tab1, tab2 = st.tabs([
        "📊 Executive Dashboard",
        "💬 Full Analysis"
    ])

    with tab1:
        render_executive_summary(st.session_state.parsed_data)
        st.markdown("---")
        render_threats_analysis(st.session_state.parsed_data)
        st.markdown("---")
        render_recommendations(st.session_state.parsed_data)
        st.markdown("---")
        render_quality_dashboard(st.session_state.parsed_data)

    with tab2:
        st.header("🎙️ Multi-Agent Analysis")
        for output in st.session_state.conversation_history:
            st.markdown(output)
