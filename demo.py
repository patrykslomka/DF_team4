# app.py
import streamlit as st
from bertopic import BERTopic
import pandas as pd
import plotly.express as px
from typing import Dict, List
import json



# Load models (cache to avoid reloading)
@st.cache_resource
def load_models():
    darkweb_model = BERTopic.load("model_comparison/best_models/BERTopic_darkweb")
    reddit_model = BERTopic.load("model_comparison/best_models/BERTopic_reddit")
    return darkweb_model, reddit_model

darkweb_model, reddit_model = load_models()

def load_data() -> Dict[str, List[str]]:
  with open('processed_data.json', 'r', encoding='utf-8') as f:
      data = json.load(f)
  df = pd.DataFrame(data)
  return {
      'darkweb': df[df['source'] == 'darkweb']['text'].tolist(),
      'reddit': df[df['source'] == 'reddit']['text'].tolist()
  }
    
def load_documents():
  data = load_data()
  # Replace with your actual document lists
  darkweb_docs = darkweb_model.get_document_info(data['darkweb'])
  reddit_docs = reddit_model.get_document_info(data['reddit'])
  return darkweb_docs, reddit_docs

darkweb_docs, reddit_docs = load_documents()

st.title("Topic Similarity Dashboard")
user_input = st.text_input("Enter a topic:")

if user_input:
    top_n = st.slider("Number of similar topics", 3, 10, 5)

    # Find similar topics and similarities
    dark_topics, dark_sim = darkweb_model.find_topics(user_input, top_n=top_n)
    reddit_topics, reddit_sim = reddit_model.find_topics(user_input, top_n=top_n)

    def build_topic_df(model, topic_ids, sims, source):
        info = model.get_topic_info()
        rows = []
        for topic_id, sim in zip(topic_ids, sims):
            if topic_id == -1:
                continue
            count = info[info["Topic"] == topic_id]["Count"].values[0]
            keywords_list = model.get_topic(topic_id)
            keywords = ", ".join([kw[0] for kw in keywords_list[:5]])
            topic_name = keywords
            rows.append({
                "Topic ID": topic_id,
                "Name": topic_name,
                "Keywords": keywords,
                "Count": count,
                "Cosine Sim": round(sim, 3),
                "Source": source
            })
        return pd.DataFrame(rows)



    dark_df = build_topic_df(darkweb_model, dark_topics, dark_sim, "Dark Web")
    reddit_df = build_topic_df(reddit_model, reddit_topics, reddit_sim, "Reddit")

    # Combine
    combined_df = pd.concat([dark_df, reddit_df])

    # Side-by-side tables with click options
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dark Web Topics")
        selected_dark = st.dataframe(dark_df[["Name", "Count", "Cosine Sim", "Keywords"]], use_container_width=True)
    with col2:
        st.subheader("Reddit Topics")
        selected_reddit = st.dataframe(reddit_df[["Name", "Count", "Cosine Sim", "Keywords"]], use_container_width=True)

    # Bar chart comparison
    st.subheader("Topic Frequency Comparison")
    fig = px.bar(
        combined_df,
        x="Name",
        y="Count",
        color="Source",
        hover_data=["Keywords", "Cosine Sim"],
        barmode="group"
    )
    st.plotly_chart(fig)

    # Topic Detail Viewer
    st.subheader("Click a topic below to view details and an example document")
    topic_options = combined_df["Name"] + " (" + combined_df["Source"] + ")"
    selected = st.selectbox("Select a topic:", topic_options)

    if selected:
      selected_row = combined_df[topic_options == selected].iloc[0]
      
      st.markdown(f"### Topic Details")
      st.markdown(f"- **Name**: {selected_row['Name']}")
      st.markdown(f"- **Source**: {selected_row['Source']}")
      st.markdown(f"- **Cosine Similarity**: {selected_row['Cosine Sim']}")
      st.markdown(f"- **Keywords**: {selected_row['Keywords']}")
      # Removed: st.code(selected_row["Example"])
