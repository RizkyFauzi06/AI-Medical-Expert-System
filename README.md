
<p align="center">
  <strong>AI Medical Expert System</strong>
</p>

<p align="center">
  An interactive AI-based medical expert system that combines Machine Learning, Bayesian reasoning, and rule-based inference to analyze reported symptoms and provide an initial assessment.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn">
</p>

---

### Overview

This project demonstrates the integration of multiple Artificial Intelligence techniques into an interactive medical chatbot. Users can describe their symptoms in natural language, while the system extracts relevant information, estimates disease probabilities, and asks follow-up questions when additional information is needed.

### Core Features

- **Symptom Extraction** — Keyword matching, synonyms, fuzzy matching, and negation detection.
- **Machine Learning** — TF-IDF feature extraction and Multinomial Naive Bayes classification.
- **Bayesian Reasoning** — Estimates and updates disease probabilities based on reported symptoms.
- **Rule-Based Inference** — Uses Forward Chaining to evaluate critical symptoms and diagnosis thresholds.
- **Information-Guided Search** — Uses Entropy and Expected Information Gain to select relevant follow-up questions.
- **Reasoning Visualization** — Provides an inspectable debug panel showing probabilities, rules, and decision processes.

### Technology Stack

`Python` · `Streamlit` · `scikit-learn` · `TF-IDF` · `Naive Bayes` · `Bayesian Probability`

---

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <sub>Developed for educational purposes. Not intended to replace professional medical diagnosis or consultation.</sub>
  
</p>
