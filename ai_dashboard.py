import streamlit as st
import pandas as pd
import snowflake.connector
import plotly.express as px
from groq import Groq
import json
import tempfile
# Note: Weasyprint requires system dependencies (see packages.txt)
from weasyprint import HTML
from streamlit_plotly_events import plotly_events
import os

# --- Initialization and Secrets Setup ---
# WARNING: HARDCODED PLACEHOLDER SECRETS FOR DEMO ONLY! 
# In production, use st.secrets for secure credential management.
secrets = {
    "groq": {"api_key": "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"},
    "snowflake": {
        "user": "yashwanth08",
        "password": "Yashwanth_2005",
        "account": "ap33012.ap-southeast-1",
        "warehouse": "COMPUTE_WH",
        "database": "fact_sales_db",
        "schema": "public",
        "role": "accountadmin"
    }
}

if secrets["groq"]["api_key"] == "YOUR_GROQ_API_KEY_HERE_FOR_DEMO":
    st.warning("🚨 Using placeholder credentials! Update `secrets` dictionary with real values before running.")

# --- Streamlit UI Configuration ---
st.set_page_config(
    page_title="AI-Powered Interactive Dashboard", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.markdown("<h1>🤖 AI Data Analyst</h1>", unsafe_allow_html=True)
st.subheader("Generate interactive data dashboards instantly using natural language.")


# ---------------- Snowflake Connection ----------------
@st.cache_resource
def get_snowflake_connection():
    try:
        return snowflake.connector.connect(**secrets["snowflake"])
    except Exception as e:
        st.error(f"Failed to connect to Snowflake. Check credentials. Error: {e}")
        st.stop()

# ---------------- Groq API Setup ----------------
try:
    client = Groq(api_key=secrets["groq"]["api_key"])
except Exception as e:
    st.error(f"Failed to initialize Groq client. Check API key. Error: {e}")
    st.stop()


# ---------------- Dashboard Functions ----------------
def get_dashboard_spec(prompt):
    """
    Call Groq API to get dashboard JSON specification.
    """
    system_msg = f"""
    You are a world-class data analysis assistant specializing in generating concise, executable SQL queries for Snowflake.
    Your task is to analyze the user's request and generate a single JSON object containing specifications for a data dashboard.
    The database schema is implicitly known from the Snowflake connection details (fact_sales_db.public). Assume relevant tables exist.
    
    Return ONLY the JSON object. Do not include any text, markdown, or commentary outside the JSON.
    
    The JSON structure MUST be:
    {{
      "charts": [
        {{
          "id": "chart1", 
          "type": "bar", 
          "title": "A descriptive chart title", 
          "sql": "SELECT column_x, SUM(column_y) FROM table_name GROUP BY column_x ORDER BY 2 DESC LIMIT 10" 
        }}
      ]
    }}
    
    Rules:
    1. Use only 'SELECT' queries.
    2. Always limit results using 'LIMIT 50000'.
    3. Ensure column names in the SELECT clause are appropriate for the chart type.
    """
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",   # ✅ updated to supported model
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Groq API Error: {e}")
        return None

def run_sql(cur, sql):
    try:
        st.info(f"Executing SQL: `{sql}`")
        cur.execute(sql)
        df = pd.DataFrame(cur.fetchall(), columns=[c[0] for c in cur.description])
        df.columns = [col.upper() for col in df.columns] 
        return df
    except Exception as e:
        st.error(f"SQL Execution Error: {e}")
        return pd.DataFrame()

def safe_append_filter(sql_query, condition):
    query_lower = sql_query.lower()
    if "where" in query_lower:
        if "order by" in query_lower:
            return sql_query.replace("ORDER BY", f" AND {condition} ORDER BY")
        elif "limit" in query_lower:
            return sql_query.replace("LIMIT", f" AND {condition} LIMIT")
        return sql_query + " AND " + condition
    else:
        if "order by" in query_lower:
            return sql_query.replace("ORDER BY", f" WHERE {condition} ORDER BY")
        elif "limit" in query_lower:
            return sql_query.replace("LIMIT", f" WHERE {condition} LIMIT")
        else:
            return sql_query + " WHERE " + condition

def plot_chart(df, chart_type, title):
    if df.empty:
        st.warning(f"No data available for {title}")
        return None
    
    if len(df.columns) < 2:
        st.warning(f"Chart '{title}' requires at least two columns.")
        st.dataframe(df)
        return None

    col_x, col_y = df.columns[0], df.columns[1]

    try:
        if chart_type == "bar":
            fig = px.bar(df, x=col_x, y=col_y, title=title)
        elif chart_type == "line":
            fig = px.line(df, x=col_x, y=col_y, title=title)
        elif chart_type == "pie":
            fig = px.pie(df, names=col_x, values=col_y, title=title)
        else:
            st.warning(f"Chart type {chart_type} not supported")
            return None
        
        fig.update_layout(
            clickmode='event+select', 
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="#1e1e1e",
            plot_bgcolor="#1e1e1e",
            font=dict(color="white")
        )
        return fig
    except Exception as e:
        st.error(f"Plotting Error for {title}: {e}")
        st.dataframe(df)
        return None


# --- Main App Logic ---
if 'chart_specs' not in st.session_state:
    st.session_state.chart_specs = []
if 'chart_dataframes' not in st.session_state:
    st.session_state.chart_dataframes = {}
if 'clicked_value' not in st.session_state:
    st.session_state.clicked_value = None


# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("⚙️ Dashboard Controls")
    st.markdown("---")
    st.markdown("#### Global Filters")
    region_filter = st.selectbox("Region:", ["All", "North", "South", "East", "West"], key="region_filter")
    product_filter = st.text_input("Product Search (LIKE):", "", key="product_filter")
    st.markdown("---")

    st.markdown("#### Export Dashboard")
    if st.button("Download as PDF", disabled=not st.session_state.chart_specs):
        with st.spinner("Generating PDF..."):
            html_tmp_path = None
            pdf_file_path = None
            try:
                html_content = "<html><head><title>AI Dashboard Export</title>"
                html_content += "<style>body { font-family: sans-serif; margin: 50px; } .chart-container { page-break-inside: avoid; margin-bottom: 30px; }</style>"
                html_content += "</head><body><h1>AI Dashboard Export</h1>"
                
                for chart_id, df in st.session_state.chart_dataframes.items():
                    if not df.empty and df.shape[1] >= 2:
                        spec = next((c for c in st.session_state.chart_specs if c['id'] == chart_id), None)
                        if spec:
                            fig = plot_chart(df, spec["type"], spec["title"])
                            if fig:
                                html_content += f'<div class="chart-container"><h2>{spec["title"]}</h2>'
                                fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", font=dict(color="black"))
                                html_content += fig.to_html(full_html=False, include_plotlyjs="cdn")
                                html_content += '</div>'
                html_content += "</body></html>"
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as html_tmp:
                    html_tmp.write(html_content)
                    html_tmp_path = html_tmp.name
                
                pdf_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
                HTML(html_tmp_path).write_pdf(pdf_file_path)
                
                with open(pdf_file_path, "rb") as pdf_file:
                    st.download_button(
                        label="Click to Download PDF",
                        data=pdf_file,
                        file_name="ai_dashboard_export.pdf",
                        mime="application/pdf"
                    )
                st.success("PDF created successfully!")
            except Exception as e:
                st.error(f"PDF Export Error: {e}")
            finally:
                if html_tmp_path and os.path.exists(html_tmp_path): os.remove(html_tmp_path)
                if pdf_file_path and os.path.exists(pdf_file_path): os.remove(pdf_file_path)

# ---------------- Main Interaction ----------------
user_prompt = st.chat_input("Ask for a dashboard, e.g., 'Show me quarterly revenue and top 5 products by sales last year.'")

if st.button("Generate Dashboard"):
    if user_prompt:
        st.session_state.last_prompt = user_prompt
    elif "last_prompt" in st.session_state:
        user_prompt = st.session_state.last_prompt
        
if user_prompt:
    st.session_state.last_prompt = user_prompt
    with st.container(border=True):
        st.markdown(f"**Your Query:** *{user_prompt}*")
        with st.spinner("🤖 AI Analyst generating dashboard..."):
            spec = get_dashboard_spec(user_prompt)

            if spec and "charts" in spec:
                st.session_state.chart_specs = spec["charts"]
                st.session_state.chart_dataframes = {}

                conn = get_snowflake_connection()
                cur = conn.cursor()
                st.markdown("---")
                st.markdown("#### Generated Charts")
                chart_cols = st.columns(2)
                
                for idx, chart in enumerate(st.session_state.chart_specs):
                    sql_query = chart["sql"]
                    if region_filter != "All":
                        sql_query = safe_append_filter(sql_query, f"REGION = '{region_filter}'")
                    if product_filter:
                        sql_query = safe_append_filter(sql_query, f"PRODUCT_NAME ILIKE '%{product_filter}%'")
                    df = run_sql(cur, sql_query)
                    st.session_state.chart_dataframes[chart["id"]] = df

                    col = chart_cols[idx % 2]
                    with col:
                        fig = plot_chart(df, chart["type"], chart["title"])
                        if fig:
                            clicked_points = plotly_events(
                                fig, 
                                click_event=True,
                                key=f"plotly_event_{chart['id']}",
                                override_height=400
                            )
                            if clicked_points:
                                clicked_x = clicked_points[0].get('x') 
                                if clicked_x is not None:
                                    st.session_state.clicked_value = str(clicked_x)
                                    st.rerun()
                cur.close()
                conn.close()
            else:
                st.error("AI failed to generate a valid chart specification (JSON structure). Try a more specific query.")

# ---------------- Cross-Filter ----------------
if st.session_state.chart_dataframes:
    st.markdown("---")
    filter_value = None
    if st.session_state.clicked_value:
        filter_value = st.session_state.clicked_value
        st.subheader(f"Cross-Filter Active: Filtering by **`{filter_value}`**")
        st.markdown("*(Click reset to clear the filter.)*")
        if st.button("🔄 Reset Filter", key="reset_button_main"):
            st.session_state.clicked_value = None
            st.rerun()

    if filter_value:
        st.markdown("---")
        chart_cols = st.columns(2)
        for idx, chart_id in enumerate(st.session_state.chart_dataframes.keys()):
            df = st.session_state.chart_dataframes[chart_id]
            spec = next((c for c in st.session_state.chart_specs if c['id'] == chart_id), None)
            if spec and not df.empty and df.shape[1] >= 2:
                filtered_df = df[df.iloc[:, 0].astype(str) == filter_value]
                col = chart_cols[idx % 2]
                with col:
                    fig = plot_chart(filtered_df, spec["type"], f"{spec['title']} (Filtered by {filter_value})")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
    else:
        chart_cols = st.columns(2)
        for idx, chart_id in enumerate(st.session_state.chart_dataframes.keys()):
            df = st.session_state.chart_dataframes[chart_id]
            spec = next((c for c in st.session_state.chart_specs if c['id'] == chart_id), None)
            if spec:
                col = chart_cols[idx % 2]
                with col:
                    fig = plot_chart(df, spec["type"], spec["title"])
                    if fig:
                        plotly_events(
                            fig, 
                            click_event=True,
                            key=f"plotly_event_{chart_id}_rerender",
                            override_height=400
                        )
