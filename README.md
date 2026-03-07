# 🤖 LLM-Based Complaint Segmentation and Labeling Dashboard

## 📌 Overview

This project demonstrates an **LLM-assisted complaint analysis system** deployed as an interactive application on **Hugging Face Spaces**.

The system automatically analyzes customer complaint data, groups similar complaints into clusters, and generates **meaningful labels for each complaint segment using a Large Language Model (LLM)**.

In addition, the system visualizes complaint distribution using **interactive bar charts**, allowing users to easily understand which complaint categories occur most frequently.

Live Application
https://huggingface.co/spaces/akilanjai71/churn_prediction_labeling

---

# 🧠 Problem Statement

Organizations receive large volumes of customer complaints from different channels such as:

• Customer support systems
• Social media
• Product reviews
• Service feedback forms

Manually categorizing these complaints is **time-consuming and inconsistent**.

This system automates the process by:

• Grouping similar complaints into clusters
• Generating meaningful segment labels using an LLM
• Visualizing complaint distribution for easy interpretation

---

# ⚙️ System Workflow

The pipeline follows these steps:

1️⃣ Complaint text preprocessing
Cleaning and preparing complaint text data for analysis.

2️⃣ Text embeddings
Generating semantic embeddings for complaints to capture contextual meaning.

3️⃣ Dimensionality reduction
Using **UMAP** to reduce embedding dimensionality for clustering.

4️⃣ Complaint clustering
Applying **KMeans clustering** to group similar complaints into segments.

5️⃣ Topic extraction
Using **BERTopic** to extract representative keywords for each complaint cluster.

6️⃣ LLM label generation
Using a **Large Language Model** to generate meaningful labels describing each complaint segment.

7️⃣ Visualization dashboard
Displaying complaint segments and counts through **interactive bar charts**.

---

# 📊 Dashboard Features

✔ Automatic complaint clustering
✔ LLM-generated complaint category names
✔ Interactive bar chart of complaints by segment
✔ Easy exploration of complaint distribution
✔ Real-time topic labeling interface

---

# 📈 Example Output

Cluster Keywords
billing, charge, refund, payment

Generated Label
Billing and Payment Issues

The dashboard then displays the **number of complaints in each segment** using a bar graph.

---

# 🛠️ Technology Stack

Programming
Python

NLP and Topic Modeling
BERTopic

Dimensionality Reduction
UMAP

Clustering
KMeans

LLM
Google FLAN-T5

Visualization
Plotly / Matplotlib

Deployment
Hugging Face Spaces

---

# 🚀 Key Benefits

• Automated complaint categorization
• Improved interpretability using LLM-generated labels
• Faster analysis of customer feedback
• Interactive visualization for business insights

---

# 🌐 Live Demo

Try the live application:

https://huggingface.co/spaces/akilanjai71/churn_prediction_labeling
