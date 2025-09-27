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
        # CRITICAL FIX: All keys must be strings (quoted) and separated by a colon (:)
        "user": "yashwanth08",
        "password": "Yashwanth_2005",
        "account": "ap33012.ap-southeast-1",
        "warehouse": "COMPUTE_WH",
        "database": "fact_sales_db",
        "schema": "public",
        "role": "accountadmin"
    }
}

# The warning will not trigger now since you provided a real Groq key
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
    """Connects to Snowflake using credentials from the hardcoded 'secrets' dictionary."""
    try:
        # Use the hardcoded 'secrets' dictionary
        return snowflake.connector.connect(**secrets["snowflake"])
    except Exception as e:
        st.error(f"Failed to connect to Snowflake. Check credentials. Error: {e}")
        st.stop()

# ---------------- Groq API Setup ----------------
try:
    # Use the hardcoded 'secrets' dictionary
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
          "type": "bar", // Must be one of: line, bar, pie
          "title": "A descriptive chart title", 
          "sql": "SELECT column_x, SUM(column_y) FROM table_name GROUP BY column_x ORDER BY 2 DESC LIMIT 10" 
        }},
        // Add more charts as needed, up to 3 for a complex query.
      ]
    }}
    
    Rules:
    1. Use only 'SELECT' queries.
    2. Always limit results using 'LIMIT 50000'.
    3. Ensure column names in the SELECT clause are appropriate for the chart type (e.g., two columns for bar/line/pie).
    """
    
    try:
        response = client.chat.completions.create(
            # Switching to a currently supported, larger Llama 3 model
            model="llama3-70b-8192", 
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
    """Run SQL on Snowflake and return DataFrame"""
    try:
        st.info(f"Executing SQL: `{sql}`")
        cur.execute(sql)
        df = pd.DataFrame(cur.fetchall(), columns=[c[0] for c in cur.description])
        # Convert all column names to uppercase for consistency with Snowflake results
        df.columns = [col.upper() for col in df.columns] 
        return df
    except Exception as e:
        st.error(f"SQL Execution Error: {e}")
        return pd.DataFrame()

def safe_append_filter(sql_query, condition):
    """Safely append WHERE or AND to SQL query."""
    query_lower = sql_query.lower()
    
    # Check if WHERE already exists
    if "where" in query_lower:
        if "order by" in query_lower:
            return sql_query.replace("ORDER BY", f" AND {condition} ORDER BY")
        elif "limit" in query_lower:
            return sql_query.replace("LIMIT", f" AND {condition} LIMIT")
        return sql_query + " AND " + condition
    else:
        # If no WHERE clause, find where to insert it, typically before ORDER BY or LIMIT
        if "order by" in query_lower:
            return sql_query.replace("ORDER BY", f" WHERE {condition} ORDER BY")
        elif "limit" in query_lower:
            return sql_query.replace("LIMIT", f" WHERE {condition} LIMIT")
        else:
            return sql_query + " WHERE " + condition

def plot_chart(df, chart_type, title):
    """Return Plotly figure for a chart type, ensuring column names are handled."""
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
            paper_bgcolor="#1e1e1e", # Dark background for aesthetics
            plot_bgcolor="#1e1e1e",
            font=dict(color="white")
        )
        return fig
    except Exception as e:
        st.error(f"Plotting Error for {title}: {e}")
        st.dataframe(df) # Show raw data on plot error
        return None

# --- Main App Logic ---

# Initialize session state for filtering and chart data
if 'chart_specs' not in st.session_state:
    st.session_state.chart_specs = []
if 'chart_dataframes' not in st.session_state:
    st.session_state.chart_dataframes = {}
if 'clicked_value' not in st.session_state:
    st.session_state.clicked_value = None


# ---------------- Sidebar for Filters and Options ----------------
with st.sidebar:
    st.title("⚙️ Dashboard Controls")
    
    # Global Filters
    st.markdown("---")
    st.markdown("#### Global Filters")
    region_filter = st.selectbox("Region:", ["All", "North", "South", "East", "West"], key="region_filter")
    product_filter = st.text_input("Product Search (LIKE):", "", key="product_filter")
    st.markdown("---")

    # Export Section (Restored for Weasyprint/packages.txt usage)
    st.markdown("#### Export Dashboard")
    
    # PDF Export Logic
    if st.button("Download as PDF", disabled=not st.session_state.chart_specs):
        with st.spinner("Generating PDF... (Requires dependencies in packages.txt)"):
            html_tmp_path = None
            pdf_file_path = None
            try:
                # 1. Create HTML content string
                html_content = "<html><head><title>AI Dashboard Export</title>"
                html_content += "<style>body { font-family: sans-serif; margin: 50px; } .chart-container { page-break-inside: avoid; margin-bottom: 30px; }</style>"
                html_content += "</head><body><h1>AI Dashboard Export</h1>"
                
                for chart_id, df in st.session_state.chart_dataframes.items():
                    if not df.empty and df.shape[1] >= 2:
                        # Find the original chart spec to get type and title
                        spec = next((c for c in st.session_state.chart_specs if c['id'] == chart_id), None)
                        if spec:
                            # Use the plotting function to get the figure (restoring dark background for consistency)
                            fig = plot_chart(df, spec["type"], spec["title"])
                            if fig:
                                html_content += f'<div class="chart-container"><h2>{spec["title"]}</h2>'
                                # Plotly figures must be exported with light background for standard PDF printing
                                fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", font=dict(color="black"))
                                # Use Plotly's to_html to embed the chart
                                html_content += fig.to_html(full_html=False, include_plotlyjs="cdn")
                                html_content += '</div>'
                                
                html_content += "</body></html>"
                
                # 2. Use tempfile to write HTML and PDF
                with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as html_tmp:
                    html_tmp.write(html_content)
                    html_tmp_path = html_tmp.name
                
                pdf_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
                
                # 3. Generate PDF using WeasyPrint
                HTML(html_tmp_path).write_pdf(pdf_file_path)
                
                # 4. Provide download link
                with open(pdf_file_path, "rb") as pdf_file:
                    st.download_button(
                        label="Click to Download PDF",
                        data=pdf_file,
                        file_name="ai_dashboard_export.pdf",
                        mime="application/pdf"
                    )
                st.success("PDF created successfully!")

            except Exception as e:
                # Catching WeasyPrint or related system errors
                st.error(f"PDF Export Error: Could not generate PDF. Ensure all dependencies in `packages.txt` are installed. Error: {e}")
            finally:
                # Clean up temp files
                if html_tmp_path and os.path.exists(html_tmp_path): os.remove(html_tmp_path)
                if pdf_file_path and os.path.exists(pdf_file_path): os.remove(pdf_file_path)

# ---------------- Main Interaction Area (Chat/Input) ----------------

# User Input for Query
user_prompt = st.chat_input("Ask for a dashboard, e.g., 'Show me quarterly revenue and top 5 products by sales last year.'")

# The Generate Button (only appears if user uses the text input above)
if st.button("Generate Dashboard"):
    # Use the content of the chat input if button is pressed
    if user_prompt:
        st.session_state.last_prompt = user_prompt
    elif "last_prompt" in st.session_state:
        user_prompt = st.session_state.last_prompt
        
if user_prompt:
    st.session_state.last_prompt = user_prompt
    
    with st.container(border=True):
        st.markdown(f"**Your Query:** *{user_prompt}*")
        
        with st.spinner("🤖 AI Analyst generating dashboard..."):
            
            # 1. Get Dashboard Spec from Groq
            spec = get_dashboard_spec(user_prompt)

            if spec and "charts" in spec:
                st.session_state.chart_specs = spec["charts"]
                st.session_state.chart_dataframes = {}

                conn = get_snowflake_connection()
                cur = conn.cursor()
                
                st.markdown("---")
                st.markdown("#### Generated Charts")
                
                # Layout for Charts (2 columns)
                chart_cols = st.columns(2)
                
                for idx, chart in enumerate(st.session_state.chart_specs):
                    sql_query = chart["sql"]

                    # 2. Apply Filters
                    if region_filter != "All":
                        sql_query = safe_append_filter(sql_query, f"REGION = '{region_filter}'")
                    if product_filter:
                        sql_query = safe_append_filter(sql_query, f"PRODUCT_NAME ILIKE '%{product_filter}%'")
                        
                    # 3. Run SQL
                    df = run_sql(cur, sql_query)
                    st.session_state.chart_dataframes[chart["id"]] = df

                    # 4. Plot Chart
                    col = chart_cols[idx % 2]
                    with col:
                        fig = plot_chart(df, chart["type"], chart["title"])
                        if fig:
                            # Use plotly_events for interactivity
                            clicked_points = plotly_events(
                                fig, 
                                click_event=True,
                                key=f"plotly_event_{chart['id']}",
                                override_height=400
                            )
                            # If a click event occurred, update the cross-filter state
                            if clicked_points:
                                # Assume the first column (X-axis) is the filter dimension
                                clicked_x = clicked_points[0].get('x') 
                                if clicked_x is not None:
                                    st.session_state.clicked_value = str(clicked_x)
                                    st.rerun() # Rerun to apply cross-filter immediately

                cur.close()
                conn.close()
            else:
                st.error("AI failed to generate a valid chart specification (JSON structure). Try a more specific query.")
                
# ---------------- Cross-Filter and Display Results ----------------

if st.session_state.chart_dataframes:
    st.markdown("---")
    
    # Determine the cross-filter value
    filter_value = None
    if st.session_state.clicked_value:
        filter_value = st.session_state.clicked_value
        st.subheader(f"Cross-Filter Active: Filtering by **`{filter_value}`**")
        st.markdown("*(Click on a new chart element or click the reset button to clear the filter.)*")
        if st.button("🔄 Reset Filter", key="reset_button_main"):
            st.session_state.clicked_value = None
            st.rerun()

    if filter_value:
        st.markdown("---")
        chart_cols = st.columns(2)
        
        # Display the filtered charts below the main charts
        for idx, chart_id in enumerate(st.session_state.chart_dataframes.keys()):
            df = st.session_state.chart_dataframes[chart_id]
            # CRITICAL FIX: The next line contained a typo (st.session_session.chart_specs). This has been corrected.
            spec = next((c for c in st.session_state.chart_specs if c['id'] == chart_id), None)
            
            if spec and not df.empty and df.shape[1] >= 2:
                # Apply filter logic
                # Assume filter is always applied to the first column (the X-axis dimension)
                filtered_df = df[df.iloc[:, 0].astype(str) == filter_value]

                col = chart_cols[idx % 2]
                with col:
                    fig = plot_chart(filtered_df, spec["type"], f"{spec['title']} (Filtered by {filter_value})")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

    else:
        # Display the initial charts if no filter is active
        # This prevents chart duplication when the user clicks 'Generate Dashboard'
        if not user_prompt:
             chart_cols = st.columns(2)
             for idx, chart_id in enumerate(st.session_state.chart_dataframes.keys()):
                df = st.session_state.chart_dataframes[chart_id]
                spec = next((c for c in st.session_state.chart_specs if c['id'] == chart_id), None)
                if spec:
                    col = chart_cols[idx % 2]
                    with col:
                        fig = plot_chart(df, spec["type"], spec["title"])
                        if fig:
                            # Re-run the plotly_events component to maintain interactivity after a rerun
                            plotly_events(
                                fig, 
                                click_event=True,
                                key=f"plotly_event_{chart_id}_rerender",
                                override_height=400
                            )
