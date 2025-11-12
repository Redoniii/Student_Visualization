# student_pdf_viz.py
import streamlit as st
import pdfplumber
import pandas as pd
import numpy as np
import io
import re
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Student Results PDF → Visualizer", layout="wide")
st.title("Student Results — PDF extractor & visualizer")
st.markdown("""
Upload a PDF with your student results.  
The app will extract tables, detect accepted/not accepted/communities, clean the data, and provide visualizations.
""")

uploaded = st.file_uploader("Upload results PDF", type=["pdf"])
if not uploaded:
    st.info("Please upload a PDF file (for example: the 'ShtypRezultatet...' PDF).")
    st.stop()

# --- Hard-coded header with Emri Prindi Mbiermi as single column ---
header_detected = [
    "Nr.", "Emri Prindi Mbiermi", "Nr.Ap.",
    "k.10", "k.11", "k.12", "k.13", "S.Ll.",
    "L1", "L2", "L3", "L.Ll.", "M",
    "M.Ll.", "P.P.", "P.P.Ll.", "Gjithsej"
]

@st.cache_data
def extract_tables_from_pdf(file_bytes: bytes):
    dfs = []
    page_infos = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""

            # Detect section
            sec = None
            if re.search(r"pranuar për regjistrim", text, flags=re.I):
                sec = "Accepted (regular)"
            elif re.search(r"nuk janë pranuar|nuk kanë kaluar pragun", text, flags=re.I):
                sec = "Not accepted / below threshold"
            elif re.search(r"komunitete", text, flags=re.I):
                sec = "Accepted (communities)"

            try:
                page_tables = page.extract_tables()
            except Exception:
                page_tables = []

            if not page_tables:
                page_infos.append({"page": i, "section": sec, "n_tables": 0})
                continue

            for table in page_tables:
                df = pd.DataFrame(table)
                page_infos.append({"page": i, "section": sec, "n_tables": len(page_tables)})
                dfs.append({"df": df, "page": i, "section": sec})
    return dfs, page_infos

# --- Extract tables ---
bytes_data = uploaded.read()
with st.spinner("Extracting tables from PDF..."):
    extracted, page_infos = extract_tables_from_pdf(bytes_data)

if len(extracted) == 0:
    st.error("No tables found. If the PDF is scanned, OCR is required.")
    st.stop()

st.success(f"Found {len(extracted)} tables across {len({p['page'] for p in page_infos})} pages.")

# --- Process tables using hard-coded header ---
clean_tables = []

for item in extracted:
    raw = item["df"].copy()
    if raw.empty:
        continue

    # Skip the header row if it exists
    first_row = list(raw.iloc[0].fillna("").astype(str).str.strip())
    if first_row[:len(header_detected)] == header_detected:
        data_start_idx = 1
    else:
        data_start_idx = 0

    new_df = raw.iloc[data_start_idx:].copy().reset_index(drop=True)

    # Force alignment to 17 columns
    if new_df.shape[1] < len(header_detected):
        for i in range(len(header_detected) - new_df.shape[1]):
            new_df[f"extra_{i}"] = ""
    elif new_df.shape[1] > len(header_detected):
        new_df = new_df.iloc[:, :len(header_detected)]

    new_df.columns = header_detected
    new_df.attrs["page"] = item["page"]
    new_df.attrs["section_guess"] = item["section"]
    clean_tables.append(new_df)

# --- Concatenate all tables ---
big = pd.concat(clean_tables, ignore_index=True, sort=False)

# --- Keep only original PDF columns ---
big = big[header_detected]

# --- Column mapping ---
cols = list(big.columns)
app_no_col = "Nr.Ap."
total_col = "Gjithsej"
chosen_section = "_section_guess"

st.sidebar.header("Column mapping (auto-detected — change if wrong)")
chosen_appno = st.sidebar.selectbox("Application number column", options=[None]+cols, index=cols.index(app_no_col)+1)
chosen_total = st.sidebar.selectbox("Total points column", options=[None]+cols, index=cols.index(total_col)+1)

# --- Apply mapping ---
df_clean = big.copy()

# Name is already in one column
df_clean["Name_full"] = df_clean["Emri Prindi Mbiermi"].astype(str).str.strip()

# Extract first name for gender detection
df_clean["FirstName"] = df_clean["Name_full"].apply(lambda x: str(x).split(" ")[0] if x else "")

# Gender detection: if first name ends with 'a' or 'ë' → girl, else boy
df_clean["Gender"] = df_clean["FirstName"].apply(lambda x: "Femër" if x.lower().endswith(("a", "ë")) else "Mashkull")

# Application number
if chosen_appno:
    df_clean["ApplicationNo"] = df_clean[chosen_appno].astype(str).str.strip()
else:
    df_clean["ApplicationNo"] = df_clean.index.astype(str)

# Total points
if chosen_total:
    df_clean["TotalPoints"] = pd.to_numeric(
        df_clean[chosen_total].astype(str).str.replace(",", ".").str.extract(r"([0-9]+(?:\.[0-9]+)?)", expand=False),
        errors="coerce"
    )
else:
    df_clean["TotalPoints"] = np.nan

# Section assignment (auto by Nr. resets)
df_clean["Nr_int"] = pd.to_numeric(df_clean["Nr."].astype(str).str.strip(), errors="coerce")
df_clean["Section_auto"] = ""

section_order = ["Accepted (regular)", "Not accepted", "Accepted (komunitet)"]
current_section_idx = 0
prev_nr = None

for idx, row in df_clean.iterrows():
    nr = row["Nr_int"]
    if pd.isna(nr):
        continue
    if nr == 1 and prev_nr is not None:
        # Nr. restarted → next section
        current_section_idx = min(current_section_idx + 1, len(section_order)-1)
    df_clean.at[idx, "Section_auto"] = section_order[current_section_idx]
    prev_nr = nr

# --- Filters ---
st.sidebar.markdown("## Filters")
min_pts = st.sidebar.number_input("Min total points (filter)", value=float(df_clean["TotalPoints"].min() or 0.0))
max_pts = st.sidebar.number_input("Max total points (filter)", value=float(df_clean["TotalPoints"].max() or 100.0))
name_search = st.sidebar.text_input("Search name (contains)")
section_filter = st.sidebar.selectbox("Section filter", options=["All"] + section_order)

df_view = df_clean.copy()
df_view = df_view[(df_view["TotalPoints"].fillna(-9999) >= min_pts) & (df_view["TotalPoints"].fillna(9999) <= max_pts)]
if name_search:
    df_view = df_view[df_view["Name_full"].str.contains(name_search, case=False, na=False)]
if section_filter != "All":
    df_view = df_view[df_view["Section_auto"]==section_filter]

# --- Show 3 previews ---
st.subheader("Accepted (regular) students")
accepted_df = df_clean[df_clean["Section_auto"]=="Accepted (regular)"]
st.dataframe(accepted_df.head(200))

st.subheader("Not accepted students")
not_accepted_df = df_clean[df_clean["Section_auto"]=="Not accepted"]
st.dataframe(not_accepted_df.head(200))

st.subheader("Accepted (komunitet) students")
komunitet_df = df_clean[df_clean["Section_auto"]=="Accepted (komunitet)"]
st.dataframe(komunitet_df.head(200))

# --- Summary metrics ---
st.subheader("Summary statistics")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Accepted (regular)", len(accepted_df))
with col2:
    st.metric("Not accepted", len(not_accepted_df))
with col3:
    st.metric("Accepted (komunitet)", len(komunitet_df))

tp = df_clean["TotalPoints"].dropna()
if not tp.empty:
    st.write(pd.DataFrame({
        "count": [int(tp.count())],
        "mean": [float(tp.mean())],
        "median": [float(tp.median())],
        "min": [float(tp.min())],
        "max": [float(tp.max())],
    }).T.rename(columns={0:"value"}))
else:
    st.info("No numeric TotalPoints detected — map the correct column in the sidebar.")

# --- Visualizations (sections + gender) ---
st.subheader("Counts by Section")
sect_counts = df_clean[df_clean["Section_auto"].isin(section_order)]["Section_auto"].value_counts().reindex(section_order).reset_index()
sect_counts.columns = ["Section", "Count"]
fig1 = px.bar(sect_counts, x="Section", y="Count", title="Counts by Section")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Gender distribution per Section")
gender_counts = df_clean[df_clean["Section_auto"].isin(section_order)].groupby(["Section_auto", "Gender"]).size().reset_index(name="Count")
fig_gender = px.bar(
    gender_counts,
    x="Section_auto",
    y="Count",
    color="Gender",
    barmode="stack",
    title="Boys vs Girls per Section",
    color_discrete_map={"Mashkull": "blue", "Girl": "red"}
)
st.plotly_chart(fig_gender, use_container_width=True)


if not tp.empty:
    fig2 = px.histogram(df_view, x="TotalPoints", nbins=25, title="Distribution of Total Points")
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("#### Top students by total points")
top_n = st.number_input("Top N", min_value=1, max_value=200, value=10)
top_df = df_clean.dropna(subset=["TotalPoints"]).sort_values("TotalPoints", ascending=False).head(top_n)
if not top_df.empty:
    fig3 = px.bar(top_df, x="Name_full", y="TotalPoints", title=f"Top {top_n} students", text="TotalPoints")
    fig3.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig3, use_container_width=True)
    st.dataframe(top_df[["Name_full", "ApplicationNo", "TotalPoints", "Section_auto", "Gender"]].reset_index(drop=True))

# --- Export ---
st.subheader("Export cleaned data")
buffer = io.BytesIO()
df_clean.to_csv(buffer, index=False)
st.download_button("Download CSV", data=buffer.getvalue(), file_name=f"parsed_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")

buffer_xlsx = io.BytesIO()
df_clean.to_excel(buffer_xlsx, index=False, engine="openpyxl")
st.download_button("Download Excel (.xlsx)", data=buffer_xlsx.getvalue(), file_name=f"parsed_results_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")
st.markdown("""**Notes & tips**:
- Sections are auto-detected based on Nr. column resets.
- Only the 3 main sections are visualized.
- Gender is inferred from the first name (last character 'a' or 'ë' → girl, else boy).
- Name is in one column 'Emri Prindi Mbiermi'.
- Extra/missing columns automatically aligned.
- If PDF is scanned, OCR is required.
- Adjust column mapping in the sidebar if needed.
""")

#python -m streamlit run student_pdf_viz4.py