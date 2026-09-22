# Smart Recipe Recommendation & Nutritional Analytics Engine

An intelligent culinary recommendation and nutritional analytics platform developed in Python and Streamlit. The application matches household pantry ingredients with recipe databases using natural language processing (TF-IDF and Cosine Similarity), re-ranks candidates using multi-criteria dietary goals, provides interactive nutritional visualizations, and computes dynamic insights to combat domestic food waste.

---

## 1. Project Overview

Household food waste is a widespread economic and environmental issue, often resulting from poorly planned grocery usage and a lack of cooking inspiration for leftover ingredients. This project addresses the problem through an automated decision-support system:

- **Food Waste Minimization**: Helps users utilize on-hand pantry ingredients before they spoil.
- **Ingredient-Based Recommendation**: Matches arbitrary ingredient queries against thousands of real-world recipes.
- **Dietary Preference-Based Ranking**: Adapts recommendations to health objectives (Balanced, High Protein, Low Calorie, or Quick Prep).
- **Nutritional Analytics**: Computes caloric and macronutrient breakdowns with comparative charts and averages.
- **Actionable Health & Waste Insights**: Calculates exact pantry utilization percentages, identifies unused items, and provides non-medical nutritional guidance.

---

## 2. Key Features

- **Food.com Dataset Support**: Built-in detection for local `RAW_recipes.csv` with a Streamlit file-uploader fallback and reproducible sampling.
- **NLP Ingredient Matching**: Text preprocessing, tokenization, and vectorization using Scikit-learn's TF-IDF and Cosine Similarity.
- **Dietary Goal Selection**:
  - **Balanced**: Balances protein density with moderate caloric, carbohydrate, and fat intake.
  - **High Protein**: Prioritizes protein-dense recipes for muscle repair and prolonged satiety.
  - **Low Calorie**: Prioritizes meals with lower caloric density for weight management.
  - **Quick Prep**: Prioritizes recipes with the shortest preparation and cooking times.
- **Interactive Preparation-Time Filter**: Dynamic slider (5 to 300 minutes) to eliminate recipes exceeding user time limits.
- **Nutritional Multi-Criteria Ranking**: Normalizes nutritional parameters and combines similarity with goal suitability.
- **Top 3 Recipe Cards**: Displays ranked recipe cards with final composite score, ingredient similarity %, prep time, and macronutrient metrics.
- **Step-by-Step Cooking Instructions**: Parses and renders actual preparation steps within an expandable UI container.
- **Nutritional Visualization**: Interactive bar charts comparing calories and macronutrients across recommendations.
- **Food Waste Utilization Analytics**: Pantry utilization metrics, leftover ingredient identification, and pantry champion highlights.

---

## 3. Tech Stack

- **Programming Language**: Python (3.9+)
- **Web Application Framework**: Streamlit
- **Data Manipulation & Analysis**: Pandas
- **Numerical Computing & Scaling**: NumPy
- **Machine Learning & NLP**: Scikit-learn (`TfidfVectorizer`, `cosine_similarity`)

---

## 4. Machine Learning & Recommendation Approach

The recommendation engine employs a hybrid content-based filtering and multi-criteria decision-making pipeline:

1. **Ingredient Preprocessing**:
   - Cleans recipe and user ingredients by converting text to lowercase.
   - Strips brackets, quotes, punctuation, and extra whitespace.
   - Formats ingredient tokens into clean, searchable comma-separated strings.

2. **TF-IDF Vectorization**:
   - Converts the recipe ingredient corpus into a sparse Term Frequency-Inverse Document Frequency (TF-IDF) feature matrix.
   - Employs English stop-word filtering to focus on informative culinary terms.

3. **Cosine Similarity Computation**:
   - Maps the user's pantry query into the same vector space.
   - Calculates the cosine angle between the query vector and all recipe vectors.
   - Applies a minimum similarity threshold (`0.05`) to discard irrelevant recipes.

4. **Nutritional Feature Normalization**:
   - Uses min-max scaling to project calories, protein, carbohydrates, fats, and preparation minutes onto a uniform `[0.0, 1.0]` scale.
   - Fills missing values with medians to prevent scaling distortions.

5. **Multi-Criteria Scoring & Re-Ranking**:
   - **Standard Goals (Balanced, High Protein, Low Calorie)**:
     $$\text{Final Score} = 0.60 \times \text{Similarity} + 0.40 \times \text{Nutrition Suitability}$$
   - **Quick Prep**:
     $$\text{Final Score} = 0.50 \times \text{Similarity} + 0.35 \times \text{Time Suitability} + 0.15 \times \text{Nutrition Suitability}$$

6. **Final Selection**:
   - Sorts candidates descending by final composite score and returns the top 3 best-matching recipes.

---

## 5. Dataset

The system is configured to work with the **Food.com (formerly GeniusKitchen) `RAW_recipes.csv`** dataset containing recipe names, cooking times, ingredient lists, preparation steps, and parsed nutritional values:

- `name`: Recipe title
- `ingredients`: List of recipe ingredients
- `nutrition`: 7-element vector `[Calories, Fat, Sugar, Sodium, Protein, Saturated Fat, Carbohydrates]`
- `minutes`: Preparation time in minutes
- `steps`: Step-by-step cooking instructions

The application automatically checks for `RAW_recipes.csv` in the local project directory and also provides a file uploader in the user interface.

---

## 6. Project Structure

```text
Smart Recipe recommendation/
├── app.py                  # Complete end-to-end application code (UI, NLP, Re-ranking, Analytics)
├── requirements.txt        # Python dependency manifest
├── README.md               # Project documentation and setup guide
└── project_report.docx     # Academic project report and system documentation
```

> **Architecture Note**: The entire application architecture (data preprocessing, TF-IDF recommendation model, multi-criteria ranking, and Streamlit UI) is deliberately contained within a single `app.py` file to eliminate external module coupling and ensure seamless portability for academic evaluation.

---

## 7. Installation

1. Clone or download this project repository.
2. Open a terminal in the project directory.
3. Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## 8. How to Run

Launch the Streamlit web application:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

---

## 9. How to Use

1. **Load/Upload Dataset**:
   Ensure `RAW_recipes.csv` is present in the project folder (detected automatically) or upload it using the file uploader in Section 1.
2. **Enter Pantry Ingredients**:
   In Section 2, enter ingredients currently available in your kitchen, separated by commas (e.g., `chicken, rice, onion, tomato, garlic`).
3. **Select Dietary Goal**:
   In Section 3, choose your preferred dietary goal: `Balanced`, `High Protein`, `Low Calorie`, or `Quick Prep`.
4. **Set Preparation-Time Limit**:
   Use the slider to specify the maximum preparation time you are willing to spend (5 to 300 minutes).
5. **Generate Recommendations**:
   Click **Find Matching Recipes** to execute TF-IDF similarity calculation and nutritional re-ranking.
6. **Review Recommendations & Insights**:
   - View the Top 3 recipe cards and expand preparation instructions.
   - Inspect the calorie and macronutrient charts in Section 5.
   - Check pantry utilization %, leftover items, and nutritional guidance in Section 6.

---

## 10. Expected Output

- **Recipe Recommendation Cards**:
  - Distinct rank badges (`#1`, `#2`, `#3`) and recipe titles.
  - Final composite score and ingredient similarity percentage.
  - Preparation time in minutes.
  - Calories, protein, carbohydrates, and fat values.
  - Expandable preparation instructions with numbered cooking steps.
- **Nutrition Analysis Dashboard**:
  - Summary cards displaying average calories, protein, carbs, and fat across top picks.
  - Interactive calorie and macronutrient distribution bar charts.
  - Standout highlights: Highest protein, lowest calories, and shortest prep time.
  - Dynamic goal interpretation text.
- **Health & Food Waste Insights**:
  - Pantry utilization rate (`%`) and utilized vs. unused ingredient counts.
  - Clear breakdown of ingredients used and leftover pantry items.
  - Top Pantry Champion recipe highlight.
  - Practical food preservation advice and non-medical dietary guidance.

---

## 11. Important Notes

- **Educational & Informational Purpose**: This project is developed as an academic software engineering and machine learning prototype.
- **Non-Clinical Disclaimer**: The nutritional scores and dietary suggestions are calculated using mathematical heuristics and multi-criteria optimization based on recipe metadata. They are not intended as medical, therapeutic, or clinical dietary prescriptions.
"# Smart-recipe-recommendation" 
