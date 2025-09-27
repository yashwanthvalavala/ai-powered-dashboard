import streamlit as st
import pandas as pd
import snowflake.connector
import plotly.express as px
from groq import Groq
import json
import tempfile
from weasyprint import HTML
from streamlit_plotly_events import plotly_events  # For capturing click events

# ---------------- Snowflake Connection ----------------
def get_snowflake_connection():
    return snowflake.connector.connect(
        user="yashwanth08",
        password="Yashwanth_2005",
        account="ap33012.ap-southeast-1",
        warehouse="COMPUTE_WH",
        database="fact_sales_db",
        schema="public",
        role="accountadmin"
    )

# ---------------- Groq API Setup ----------------
client = Groq(api_key="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")

# ---------------- Dashboard Functions ----------------
def get_dashboard_spec(prompt):
    """Call LLaMA 3 via Groq API to get dashboard JSON spec"""
    system_msg = """
    You are a data assistant. 
    Return JSON with:
    {
      "charts": [
        {"id":"chart1","type":"line","title":"Monthly Revenue","sql":"..."},
        {"id":"chart2","type":"bar","title":"Top Products","sql":"..."}
      ]
    }
    Use only SELECT queries, limit rows to 50k.
    """
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )
    # FIX: Access message content correctly
    return json.loads(response.choices[0].message.content)

def run_sql(cur, sql):
    """Run SQL on Snowflake and return DataFrame"""
    cur.execute(sql)
    return pd.DataFrame(cur.fetchall(), columns=[c[0] for c in cur.description])

def safe_append_filter(sql_query, condition):
    """Safely append WHERE or AND to SQL"""
    if "where" in sql_query.lower():
        return sql_query + " AND " + condition
    else:
        return sql_query + " WHERE " + condition

def plot_chart(df, chart_type, title):
    """Return Plotly figure for a chart type"""
    if df.empty:
        st.warning(f"No data available for {title}")
        return None

    if chart_type == "bar":
        fig = px.bar(df, x=df.columns[0], y=df.columns[1], title=title)
    elif chart_type == "line":
        fig = px.line(df, x=df.columns[0], y=df.columns[1], title=title)
    elif chart_type == "pie":
        fig = px.pie(df, names=df.columns[0], values=df.columns[1], title=title)
    else:
        st.warning(f"Chart type {chart_type} not supported")
        return None
    fig.update_layout(clickmode='event+select')  # Enable click events
    return fig

# ---------------- Streamlit UI ----------------
st.set_page_config(layout="wide")
st.markdown("<h1>📊 AI-Powered Interactive Dashboard</h1>", unsafe_allow_html=True)

# ✅ Default message (so page is never blank)
st.info("Enter a query and click **Generate Dashboard** to begin.")

user_prompt = st.text_input("Enter your dashboard query:")

# Filters
st.markdown("<h2>🔎 Filters</h2>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    region_filter = st.selectbox("Select Region:", ["All", "North", "South", "East", "West"])
with col2:
    product_filter = st.text_input("Filter Product (optional):", "")

# ---------------- Generate Dashboard ----------------
if st.button("Generate Dashboard") and user_prompt.strip():
    try:
        spec = get_dashboard_spec(user_prompt)

        # Connect Snowflake
        conn = get_snowflake_connection()
        cur = conn.cursor()

        chart_dataframes = {}
        chart_placeholders = {}

        # Layout 2 charts per row
        columns = st.columns(2)

        for idx, chart in enumerate(spec["charts"]):
            sql_query = chart["sql"]

            # Apply filters safely
            if region_filter != "All":
                sql_query = safe_append_filter(sql_query, f"region = '{region_filter}'")
            if product_filter:
                sql_query = safe_append_filter(sql_query, f"product_name LIKE '%{product_filter}%'")

            df = run_sql(cur, sql_query)
            chart_dataframes[chart["id"]] = df

            col = columns[idx % 2]
            with col:
                fig = plot_chart(df, chart["type"], chart["title"])
                placeholder = st.empty()
                chart_placeholders[chart["id"]] = placeholder

                if fig:
                    # Capture click events
                    clicked_points = plotly_events(fig, click_event=True)
                    if clicked_points:
                        st.session_state.clicked_value = str(clicked_points[0]['x'])

                    placeholder.plotly_chart(fig, use_container_width=True)

        # ---------------- Drill-down / Cross-filter ----------------
        st.markdown("<h2>🖱 Drill-down / Cross-filter</h2>", unsafe_allow_html=True)
        st.write("Click on any chart element to filter all charts dynamically.")

        # Input box as fallback
        filter_value = st.text_input("Or enter value to filter charts manually:")
        if filter_value:
            st.session_state.clicked_value = filter_value

        # Update charts if clicked_value is set
        if "clicked_value" in st.session_state and st.session_state.clicked_value:
            clicked_value = st.session_state.clicked_value
            st.write(f"Filtering all charts by: {clicked_value}")
            for chart_id, df in chart_dataframes.items():
                if not df.empty:
                    filtered_df = df[df.iloc[:, 0].astype(str) == clicked_value]
                    fig = plot_chart(filtered_df, spec["charts"][0]["type"], f"{spec['charts'][0]['title']} (Filtered)")
                    if fig:
                        chart_placeholders[chart_id].plotly_chart(fig, use_container_width=True)

        # ---------------- Export ----------------
        st.markdown("<h2>💾 Export Dashboard</h2>", unsafe_allow_html=True)

        if st.button("Export as HTML"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
                # Export all charts into one HTML file
                for df in chart_dataframes.values():
                    if not df.empty:
                        fig = plot_chart(df, "bar", "Exported Chart")
                        if fig:
                            fig.write_html(tmp.name, include_plotlyjs="cdn", full_html=False, append=True)
                st.success(f"Dashboard exported as {tmp.name}")

        if st.button("Export as PDF"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as html_tmp:
                html_content = "<h1>Dashboard Export</h1>"
                for df in chart_dataframes.values():
                    if not df.empty:
                        fig = plot_chart(df, "bar", "Exported Chart")
                        if fig:
                            html_content += fig.to_html(full_html=False, include_plotlyjs="cdn")
                html_tmp.write(html_content.encode())
                html_tmp.flush()
                pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                HTML(html_tmp.name).write_pdf(pdf_file.name)
                st.success(f"Dashboard exported as {pdf_file.name}")

        # Close Snowflake
        cur.close()
        conn.close()

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
