import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Data Science & MLOps Landscape", layout="wide")

st.title("📊 Data Science & MLOps Job Landscape")
st.markdown("Advanced data visualization and geospatial analysis of data science jobs worldwide.")

# --------------------------------------------------
# Load data
# --------------------------------------------------
df = pd.read_csv("ds_salaries.csv")
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# --------------------------------------------------
# Top 5 job roles
# --------------------------------------------------
st.subheader("Top 5 Data Science Job Roles")
job_counts = df["job_title"].value_counts().head(5)
fig = px.bar(
    x=job_counts.index,
    y=job_counts.values,
    labels={"x": "Job Role", "y": "Number of Jobs"},
    title="Top 5 Data Science Job Roles"
)
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Salary distribution by experience level
# --------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    fig = px.box(
        df,
        x="experience_level",
        y="salary_in_usd",
        title="Salary Distribution by Experience Level"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.pie(
        df,
        names="employment_type",
        title="Employment Type Distribution",
        hole=0.5
    )
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# World map
# --------------------------------------------------
st.subheader("Global Distribution of Data Science Jobs")
country_dict = {
    "US": "United States", "GB": "United Kingdom", "CA": "Canada",
    "DE": "Germany", "IN": "India", "FR": "France", "ES": "Spain",
    "GR": "Greece", "JP": "Japan", "NL": "Netherlands"
}
df["company_country"] = df["company_location"].map(country_dict)

fig = px.scatter_geo(
    df,
    locations="company_country",
    locationmode="country names",
    color="experience_level",
    size="salary_in_usd",
    hover_name="job_title",
    title="Top Data Science Jobs Around the World"
)
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Heatmap
# --------------------------------------------------
st.subheader("Average Salary by Job Title and Experience Level")
heat_data = df.pivot_table(
    index="experience_level",
    columns="job_title",
    values="salary_in_usd",
    aggfunc="mean"
).fillna(0)

fig = px.imshow(
    heat_data,
    text_auto=True,
    aspect="auto",
    title="Average Salary by Job Title and Experience Level"
)
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Parallel categories
# --------------------------------------------------
st.subheader("Salary Flow Across Experience, Employment Type, and Company Size")
fig = px.parallel_categories(
    df,
    dimensions=["experience_level", "employment_type", "company_size"],
    color="salary_in_usd",
    color_continuous_scale=px.colors.sequential.Inferno,
    labels={"salary_in_usd": "Salary (USD)"}
)
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Bubble chart
# --------------------------------------------------
col3, col4 = st.columns(2)

with col3:
    fig = px.scatter(
        df,
        x="experience_level",
        y="salary_in_usd",
        size="salary_in_usd",
        color="employment_type",
        hover_name="job_title",
        title="Salary vs Experience Level Bubble Chart"
    )
    st.plotly_chart(fig, use_container_width=True)

with col4:
    funnel_data = df.groupby("job_title")["salary_in_usd"].mean().sort_values(ascending=False).head(5)
    fig = px.funnel(
        y=funnel_data.index,
        x=funnel_data.values,
        title="Top 5 Job Roles by Average Salary"
    )
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Radar chart
# --------------------------------------------------
st.subheader("Average Salary by Experience Level for Top Job Roles")
top_jobs = df["job_title"].value_counts().head(3).index
categories = list(df["experience_level"].unique())

fig = go.Figure()
for job in top_jobs:
    values = []
    for exp in categories:
        avg_salary = df[(df["job_title"] == job) & (df["experience_level"] == exp)]["salary_in_usd"].mean()
        values.append(avg_salary)
    fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill="toself", name=job))

fig.update_layout(
    polar=dict(radialaxis=dict(visible=True)),
    title="Average Salary by Experience Level for Top Job Roles"
)
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Icicle chart
# --------------------------------------------------
st.subheader("Job Distribution by Company Size, Employment Type, and Experience Level")
fig = px.icicle(
    df,
    path=["company_size", "employment_type", "experience_level"],
    values="salary_in_usd",
    title="Job Distribution by Company Size, Employment Type, and Experience Level"
)
st.plotly_chart(fig, use_container_width=True)
