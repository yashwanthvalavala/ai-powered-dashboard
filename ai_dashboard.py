import streamlit as st
import pandas as pd
import snowflake.connector
import plotly.express as px
from groq import Groq
import json
from streamlit_plotly_events import plotly_events

# --- Initialization and Secrets Setup ---
secrets = {
    "groq": {"api_key": "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"},
    "snowflake": {
        "user": "yashwanth08",
        "password": "Yashwanth_2005",
        "account": "ap33012.ap-southeast-1",
        "warehouse": "COMPUTE_WH",
        "database": "FACT_SALES_DB",
        "schema": "PUBLIC",
        "role": "ACCOUNTADMIN"
    }
}

# --- Streamlit UI Configuration ---
st.set_page_config(
    page_title="AI-Powered Interactive Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("<h1>🤖 AI Data Analyst</h1>", unsafe_allow_html=True)
st.subheader("Generate interactive data dashboards instantly using natural language.")

# ---------------- Snowflake Connection ----------------
def get_snowflake_connection():
    if "snowflake_conn" not in st.session_state:
        try:
            conn = snowflake.connector.connect(**secrets["snowflake"])
            # Set context
            with conn.cursor() as cur:
                cur.execute(f"USE WAREHOUSE {secrets['snowflake']['warehouse']}")
                cur.execute(f"USE DATABASE {secrets['snowflake']['database']}")
                cur.execute(f"USE SCHEMA {secrets['snowflake']['schema']}")
                cur.execute(f"USE ROLE {secrets['snowflake']['role']}")
            st.session_state.snowflake_conn = conn
            st.success(f"✅ Connected to Snowflake: {secrets['snowflake']['database']}.{secrets['snowflake']['schema']}")
        except Exception as e:
            st.error(f"❌ Failed to connect to Snowflake. Error: {e}")
            st.stop()
    return st.session_state.snowflake_conn

# ---------------- Groq API Setup ----------------
try:
    client = Groq(api_key=secrets["groq"]["api_key"])
except Exception as e:
    st.error(f"Failed to initialize Groq client. Check API key. Error: {e}")
    st.stop()

# ---------------- Dashboard Functions ----------------
def get_dashboard_spec(prompt):
    system_msg = f"""
    You are a world-class data analysis assistant specializing in generating concise, executable SQL queries for Snowflake.
    The main sales table is FACT_SALES_DB.PUBLIC.FACT_SALES.
    Return ONLY the JSON object in this format:
    {{
      "charts": [
        {{
          "id": "chart1", 
          "type": "bar", 
          "title": "Descriptive title", 
          "sql": "SELECT column_x, SUM(column_y) FROM FACT_SALES_DB.PUBLIC.FACT_SALES GROUP BY column_x ORDER BY 2 DESC LIMIT 50000" 
        }}
      ]
    }}
    Rules:
    1. Use only SELECT queries.
    2. Fully qualified table name: FACT_SALES_DB.PUBLIC.FACT_SALES.
    3. Limit results using LIMIT 50000.
    """
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
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

def run_sql(conn, sql):
    try:
        with conn.cursor() as cur:
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

# ---------------- Main Interaction ----------------
user_prompt = st.chat_input("Ask for a dashboard, e.g., 'Show me quarterly revenue and top 5 products by sales last year.'")

if st.button("Generate Dashboard") and user_prompt:
    st.session_state.last_prompt = user_prompt
elif "last_prompt" in st.session_state:
    user_prompt = st.session_state.last_prompt

if user_prompt:
    st.session_state.last_prompt = user_prompt
    st.markdown(f"**Your Query:** *{user_prompt}*")
    with st.spinner("🤖 AI Analyst generating dashboard..."):
        spec = get_dashboard_spec(user_prompt)

        if spec and "charts" in spec:
            st.session_state.chart_specs = spec["charts"]
            st.session_state.chart_dataframes = {}
            conn = get_snowflake_connection()

            st.markdown("---")
            st.markdown("#### Generated Charts")
            chart_cols = st.columns(2)

            for idx, chart in enumerate(st.session_state.chart_specs):
                sql_query = chart["sql"]
                if region_filter != "All":
                    sql_query = safe_append_filter(sql_query, f"REGION = '{region_filter}'")
                if product_filter:
                    sql_query = safe_append_filter(sql_query, f"PRODUCT_NAME ILIKE '%{product_filter}%'")
                
                df = run_sql(conn, sql_query)
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
        else:
            st.error("AI failed to generate a valid chart specification (JSON structure). Try a more specific query.")
