import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
from google import genai

# Fetch Gemini API key from secrets.toml
api_key = st.secrets["gemini"]["api_key"]

# Initialize your client
client = genai.Client(api_key=api_key)


# --- Gemini API Setup ---
api_key = "temp key"  # Replace with your Gemini API key
client = genai.Client(api_key=api_key)

# --- Helper Functions ---
def summarize_text_with_gemini(prompt):
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    return response.text

def read_uploaded_file(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith(".tsv"):
        return pd.read_csv(file, sep="\t")
    elif file.name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    elif file.name.endswith(".json"):
        try:
            data = json.load(file)
            return pd.json_normalize(data)
        except Exception as e:
            st.error(f"Error reading JSON: {e}")
            return None
    else:
        st.warning("Unsupported file type.")
        return None

# --- Streamlit UI Setup ---
st.set_page_config(page_title="AI Data Summary Tool", layout="wide")

# Set slightly darker sage green background
st.markdown("""
    <style>
        body {
            background-color: #9CAF88;
        }
        .stApp {
            background-color: #9CAF88;
        }
    </style>
""", unsafe_allow_html=True)

st.title("AI-Powered Analysis Tool")

# Optional: Custom download button style
st.markdown("""
    <style>
    .stDownloadButton button {
        background-color: #88b04b;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .stDownloadButton button:hover {
        background-color: #729d39;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar File Uploads ---
st.sidebar.header("📊 Upload Your Data")
num_datasets = st.sidebar.number_input("Number of Data Files", min_value=1, max_value=6, value=3, step=1)

uploaded_files = {}
for i in range(num_datasets):
    label = st.sidebar.text_input(f"Label for Dataset {i+1}", key=f"label_{i}")
    file = st.sidebar.file_uploader(
        f"Upload File for {label or f'Dataset {i+1}'}",
        type=["csv", "xlsx", "xls", "json", "tsv"],
        key=f"file_{i}"
    )
    if label and file:
        df = read_uploaded_file(file)
        if df is not None:
            uploaded_files[label] = df

# --- Data Processing ---
data_loaded = len(uploaded_files) == num_datasets

if data_loaded:
    if st.checkbox("Preview Uploaded Data"):
        for label, df in uploaded_files.items():
            with st.expander(f"📄 {label} Data", expanded=False):
                st.dataframe(df)

    if st.button("Generate Insightful Summary"):
        with st.spinner("Generating insights with Gemini..."):
            prompt = "Analyze the uploaded demographic datasets.\n\n"
            for label, df in uploaded_files.items():
                summary_text = df.describe(include="all").to_string()
                prompt += f"--- {label.upper()} DATA SUMMARY ---\n{summary_text}\n\n"

            prompt += """
Please analyze the data to extract meaningful insights and highlight the most significant trends. Focus on:

1. **Most Affected Groups** – Who is most impacted? (1–2 bullet points)
2. **Notable Disparities** – Any outliers or differences worth noting? (1–2 bullet points)
3. **Patterns & Trends** – Summarize any consistent or emerging patterns. (2–3 bullets)
4. **Practical Implications** – What should someone do or consider based on this data? (2 concise bullets)

Use bullet points under each heading. No longer than 8–10 bullet points total. Avoid repeating data descriptions.
"""

            ai_summary = summarize_text_with_gemini(prompt)

        st.subheader("📊 Gemini AI Summary")
        st.markdown(
            f"""
            <style>
                .summary-box {{
                    background-color: #edf5ef;
                    color: #2e4d3f;
                    padding: 24px;
                    border-radius: 12px;
                    font-size: 17px;
                    line-height: 1.7;
                    border-left: 6px solid #88b04b;
                }}
                .summary-title {{
                    color: #88b04b;
                    font-size: 20px;
                    font-weight: bold;
                    margin-bottom: 10px;
                }}
            </style>
            <div class='summary-box'>
                <div class='summary-title'>📈 Key Insights</div>
                {ai_summary.replace('\n', '<br>')}
            </div>
            """,
            unsafe_allow_html=True
        )
        st.download_button("📥 Download Summary", ai_summary, file_name="data_summary.txt")

    st.subheader("💬 Ask a Question About Your Data")
    user_question = st.text_input("Ask anything (e.g., 'Which group has the highest counts?')")

    if user_question:
        with st.spinner("Thinking..."):
            question_prompt = "Answer the following question using the data summaries:\n\n"
            for label, df in uploaded_files.items():
                question_prompt += f"--- {label.upper()} DATA ---\n{df.describe(include='all').to_string()}\n\n"
            question_prompt += f"\nQuestion: {user_question}"
            ai_answer = summarize_text_with_gemini(question_prompt)
        st.markdown(
            f"""
            <style>
                .question-response {{
                    background-color: #edf5ef;
                    color: #2e4d3f;
                    padding: 20px;
                    border-radius: 10px;
                    font-size: 16px;
                    line-height: 1.6;
                    border-left: 5px solid #edf5ef;
                    margin-top: 10px;
                }}
            </style>
            <div class='question-response'>
                {ai_answer.replace('\n', '<br>')}
            </div>
            """,
            unsafe_allow_html=True
        )

    # --- Visualizations ---
    st.subheader("📊 Visualizations")
    for label, df in uploaded_files.items():
        try:
            st.markdown(f"**{label} Distribution**")
            col_names = df.columns.tolist()
            group_col = col_names[0]
            value_col = next((c for c in col_names[1:] if pd.api.types.is_numeric_dtype(df[c])), None)

            if value_col:
                plt.style.use('seaborn-v0_8-whitegrid')
                fig, ax = plt.subplots()
                df.groupby(group_col)[value_col].sum().plot(kind='bar', ax=ax, color="#88b04b", edgecolor="#2e4d3f")
                ax.set_title(f"{value_col} by {group_col}", color="#2e4d3f")
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig)
            else:
                st.info(f"No numeric column found to plot for {label}.")
        except Exception as e:
            st.warning(f"Error generating plot for {label}: {e}")

else:
    st.info("Please label and upload all the required data files.")
