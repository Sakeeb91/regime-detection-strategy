"""
Professional UI components and styling for Streamlit app.
"""

import streamlit as st


def apply_professional_styling():
    """Apply professional CSS styling with glassmorphism effects."""
    st.markdown("""
    <style>
        /* Professional glassmorphism cards */
        .stMetric {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
            border: 1px solid rgba(99, 102, 241, 0.2);
            border-radius: 12px;
            padding: 1.5rem !important;
            backdrop-filter: blur(10px);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .stMetric:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 16px rgba(99, 102, 241, 0.2);
        }

        /* Elevated metric values with gradient */
        .stMetric label {
            font-size: 0.875rem !important;
            font-weight: 600 !important;
            color: #94A3B8 !important;
        }

        .stMetric [data-testid="stMetricValue"] {
            font-size: 2rem !important;
            font-weight: 700 !important;
            background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* Modern buttons with hover effects */
        .stButton > button {
            background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
        }

        .stButton > button:active {
            transform: translateY(0);
        }

        /* Smooth tab transitions */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: rgba(30, 41, 59, 0.5);
            border-radius: 12px;
            padding: 4px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            transition: all 0.2s ease;
        }

        .stTabs [data-baseweb="tab"]:not([aria-selected="true"]) {
            color: #94A3B8;
        }

        .stTabs [data-baseweb="tab"]:not([aria-selected="true"]):hover {
            color: #CBD5E1;
            background: rgba(99, 102, 241, 0.1);
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
            color: #FFFFFF !important;
        }

        /* Refined dataframes */
        .stDataFrame {
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        /* Loading animations */
        .stSpinner > div {
            border-top-color: #6366F1 !important;
        }

        /* Sidebar refinement */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1E293B 0%, #0F172A 100%);
            border-right: 1px solid rgba(99, 102, 241, 0.2);
        }

        /* Alert boxes with refined styling */
        .stAlert {
            border-radius: 12px;
            border-left: 4px solid;
            padding: 1rem 1.5rem;
        }

        /* Expander styling */
        .streamlit-expanderHeader {
            background: rgba(30, 41, 59, 0.5);
            border-radius: 8px;
            font-weight: 600;
        }

        /* Slider styling with comprehensive selectors */

        /* Slider track - the background bar */
        .stSlider [data-baseweb="slider"] {
            background-color: rgba(148, 163, 184, 0.2) !important;
        }

        /* Slider track inner (filled portion) */
        .stSlider [data-baseweb="slider"] [data-baseweb="slider-track-inner"] {
            background: linear-gradient(90deg, #6366F1 0%, #8B5CF6 100%) !important;
        }

        /* Slider thumb - the draggable circle */
        .stSlider [data-baseweb="slider"] [role="slider"] {
            background-color: #FFFFFF !important;
            border: 3px solid #6366F1 !important;
            box-shadow: 0 2px 8px rgba(99, 102, 241, 0.4) !important;
            width: 20px !important;
            height: 20px !important;
        }

        .stSlider [data-baseweb="slider"] [role="slider"]:hover {
            background-color: #FFFFFF !important;
            border-color: #8B5CF6 !important;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.6) !important;
            transform: scale(1.1);
        }

        .stSlider [data-baseweb="slider"] [role="slider"]:active {
            background-color: #FFFFFF !important;
            border-color: #6366F1 !important;
            box-shadow: 0 2px 8px rgba(99, 102, 241, 0.8) !important;
        }

        /* Slider labels - all variations */
        .stSlider label,
        .stSlider > label,
        .stSlider div[data-testid="stMarkdownContainer"] {
            color: #F1F5F9 !important;
            font-weight: 600 !important;
        }

        /* Slider current value display */
        .stSlider [data-baseweb="slider-value"],
        .stSlider div[data-testid="stTickBar"] div {
            color: #F1F5F9 !important;
        }

        /* Slider tick marks and numbers */
        .stSlider [data-baseweb="slider"] [role="slider"] + div,
        .stSlider div[data-baseweb="base-input"] {
            color: #F1F5F9 !important;
        }

        /* Min/max labels on slider */
        .stSlider div[data-testid="stTickBar"] {
            color: #94A3B8 !important;
        }

        /* Radio buttons with better contrast */
        .stRadio label {
            color: #F1F5F9 !important;
        }

        .stRadio [data-baseweb="radio"] [aria-checked="true"] {
            background-color: #6366F1 !important;
        }

        /* Checkbox styling */
        .stCheckbox label {
            color: #F1F5F9 !important;
        }

        /* Select boxes */
        .stSelectbox label, .stMultiSelect label {
            color: #F1F5F9 !important;
            font-weight: 600;
        }

        .stSelectbox [data-baseweb="select"] span,
        .stMultiSelect [data-baseweb="select"] span {
            color: #F1F5F9 !important;
        }

        /* Text input and number input styling */
        .stTextInput label, .stNumberInput label {
            color: #F1F5F9 !important;
            font-weight: 600;
        }

        .stTextInput input, .stNumberInput input {
            color: #F1F5F9 !important;
            background-color: rgba(30, 41, 59, 0.5) !important;
            border-color: rgba(99, 102, 241, 0.3) !important;
        }

        .stTextInput input:focus, .stNumberInput input:focus {
            border-color: #6366F1 !important;
            box-shadow: 0 0 0 1px #6366F1 !important;
        }

        /* Select box styling */
        .stSelectbox [data-baseweb="select"] {
            border-radius: 8px;
        }

        /* Divider enhancement */
        hr {
            margin: 2rem 0;
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent 0%, rgba(99, 102, 241, 0.3) 50%, transparent 100%);
        }

        /* Mobile responsiveness */
        @media (max-width: 768px) {
            .stMetric {
                margin-bottom: 1rem;
            }

            .block-container {
                padding: 1rem !important;
            }

            .stTabs [data-baseweb="tab-list"] {
                overflow-x: auto;
                overflow-y: hidden;
            }

            h1 { font-size: 2rem !important; }
            h2 { font-size: 1.5rem !important; }
            h3 { font-size: 1.25rem !important; }
        }
    </style>
    """, unsafe_allow_html=True)


def render_hero_section(title, subtitle):
    """Render a professional hero section with gradient text."""
    st.markdown(f"""
    <div style="text-align: center; padding: 2rem 0 3rem 0;">
        <h1 style="font-size: 3.5rem; font-weight: 800; margin-bottom: 0.5rem;
                    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 50%, #EC4899 100%);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    line-height: 1.2;">
            {title}
        </h1>
        <p style="font-size: 1.25rem; color: #94A3B8; max-width: 600px; margin: 0 auto;">
            {subtitle}
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_feature_card(icon, title, description, badge):
    """Render a feature card with icon, title, description, and badge."""
    return f"""
    <div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.05) 0%, rgba(139, 92, 246, 0.05) 100%);
                border: 1px solid rgba(99, 102, 241, 0.2);
                border-radius: 16px; padding: 1.5rem; height: 220px;
                transition: transform 0.3s ease, box-shadow 0.3s ease;">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{icon}</div>
        <div style="font-weight: 700; font-size: 1.1rem; margin-bottom: 0.5rem; color: #F1F5F9;">{title}</div>
        <div style="color: #94A3B8; font-size: 0.9rem; margin-bottom: 0.75rem; line-height: 1.5;">{description}</div>
        <span style="background: rgba(99, 102, 241, 0.2); color: #A5B4FC;
                     padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.75rem; font-weight: 600;">
            {badge}
        </span>
    </div>
    """


def render_metric_card(icon, value, label, sublabel):
    """Render an enhanced metric card with icon."""
    return f"""
    <div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
                border: 1px solid rgba(99, 102, 241, 0.3);
                border-radius: 12px; padding: 1.5rem; text-align: center;
                transition: transform 0.3s ease;">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
        <div style="font-size: 2.5rem; font-weight: 800;
                    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            {value}
        </div>
        <div style="font-weight: 600; margin-bottom: 0.25rem; color: #F1F5F9;">{label}</div>
        <div style="color: #94A3B8; font-size: 0.85rem;">{sublabel}</div>
    </div>
    """


def render_empty_state(title, description, action_text, icon="📊"):
    """Render an empty state with call to action."""
    st.markdown(f"""
    <div style="text-align: center; padding: 4rem 2rem;
                background: rgba(99, 102, 241, 0.05); border-radius: 16px;
                border: 2px dashed rgba(99, 102, 241, 0.3); margin: 2rem 0;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">{icon}</div>
        <div style="font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem; color: #F1F5F9;">{title}</div>
        <div style="color: #94A3B8; margin-bottom: 1.5rem; max-width: 500px; margin-left: auto; margin-right: auto;">
            {description}
        </div>
        <div style="display: inline-block; background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
                    color: white; padding: 0.75rem 2rem; border-radius: 8px; font-weight: 600;
                    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);">
            {action_text}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_loading_skeleton(height="400px"):
    """Render a skeleton loading placeholder."""
    st.markdown(f"""
    <div style="animation: pulse 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite;">
        <div style="height: 80px; background: rgba(148, 163, 184, 0.1);
                    border-radius: 8px; margin-bottom: 1rem;"></div>
        <div style="height: {height}; background: rgba(148, 163, 184, 0.1);
                    border-radius: 8px;"></div>
    </div>
    <style>
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.5; }}
    }}
    </style>
    """, unsafe_allow_html=True)


def render_progress_bar(label, value, max_value=100, color="#10B981"):
    """Render a styled progress bar with label."""
    percentage = (value / max_value) * 100

    st.markdown(f"""
    <div style="margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="font-weight: 600; color: #F1F5F9;">{label}</span>
            <span style="color: {color}; font-weight: 700;">{percentage:.1f}%</span>
        </div>
        <div style="background: rgba(148, 163, 184, 0.2); height: 8px; border-radius: 4px; overflow: hidden;">
            <div style="background: linear-gradient(90deg, {color} 0%, {color}AA 100%);
                        width: {percentage}%; height: 100%; transition: width 0.5s ease;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_alert(title, message, action, priority='info'):
    """Render a styled alert with priority level."""
    priority_config = {
        'critical': {'icon': '🚨', 'color': '#EF4444', 'bg': 'rgba(239, 68, 68, 0.1)', 'label': 'CRITICAL'},
        'warning': {'icon': '⚠️', 'color': '#F59E0B', 'bg': 'rgba(245, 158, 11, 0.1)', 'label': 'WARNING'},
        'info': {'icon': '💡', 'color': '#6366F1', 'bg': 'rgba(99, 102, 241, 0.1)', 'label': 'INFO'},
        'success': {'icon': '✅', 'color': '#10B981', 'bg': 'rgba(16, 185, 129, 0.1)', 'label': 'SUCCESS'}
    }

    config = priority_config.get(priority, priority_config['info'])

    st.markdown(f"""
    <div style="background: {config['bg']}; border-left: 4px solid {config['color']};
                border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <span style="font-size: 1.5rem; margin-right: 0.5rem;">{config['icon']}</span>
            <span style="font-weight: 700; color: {config['color']}; font-size: 0.75rem;
                         background: {config['color']}22; padding: 0.25rem 0.75rem;
                         border-radius: 12px; margin-right: 0.75rem;">
                {config['label']}
            </span>
            <span style="font-weight: 700; font-size: 1.1rem; color: #F1F5F9;">{title}</span>
        </div>
        <div style="color: #94A3B8; margin-bottom: 0.5rem; padding-left: 2.5rem;">
            {message}
        </div>
        <div style="color: {config['color']}; font-weight: 600; padding-left: 2.5rem;">
            → {action}
        </div>
    </div>
    """, unsafe_allow_html=True)