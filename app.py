import ast
import os
import re
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Smart Recipe Recommendation & Nutritional Analytics Engine",
    page_icon="🍳",
    layout="wide",
)

# ==============================================================================
# Header & Project Overview
# ==============================================================================
st.title("🍳 Smart Recipe Recommendation & Nutritional Analytics Engine")
st.write(
    "A smart recommendation tool that suggests personalized recipes based on "
    "available ingredients and dietary preferences, complete with detailed "
    "nutritional analytics and insights to minimize food waste."
)
st.divider()

# ==============================================================================
# Constants & Configurations
# ==============================================================================
REQUIRED_COLUMNS = ["name", "ingredients", "nutrition", "minutes"]

NUTRITION_COLUMNS = [
    "calories",
    "fat",
    "sugar",
    "sodium",
    "protein",
    "saturated_fat",
    "carbohydrates",
]

DEFAULT_SAMPLE_SIZE = 15000
RANDOM_STATE = 42
LOCAL_CSV_FILENAME = "RAW_recipes.csv"
MIN_SIMILARITY_THRESHOLD = 0.05

# Multi-Criteria Ranking Weights
# Standard criteria (Balanced, High Protein, Low Calorie)
WEIGHT_SIMILARITY_STANDARD = 0.60
WEIGHT_NUTRITION_STANDARD = 0.40

# Quick Prep criteria (incorporates preparation-time suitability)
WEIGHT_SIMILARITY_QUICK_PREP = 0.50
WEIGHT_TIME_QUICK_PREP = 0.35
WEIGHT_NUTRITION_QUICK_PREP = 0.15


# ==============================================================================
# Helper Functions for Cleaning & Parsing
# ==============================================================================
def clean_ingredients(raw_ingredients):
    """
    Cleans the ingredients data:
    - Converts text to lowercase
    - Strips brackets, quotes, and unnecessary whitespace
    - Converts list/string into a clean, searchable comma-separated string
    - Handles missing or malformed inputs safely
    """
    if pd.isna(raw_ingredients):
        return ""

    # If already a list or tuple
    if isinstance(raw_ingredients, (list, tuple)):
        items = [str(x).strip().lower() for x in raw_ingredients if str(x).strip()]
        return ", ".join(items)

    raw_str = str(raw_ingredients).strip()
    if not raw_str:
        return ""

    # Attempt safe evaluation if it looks like a Python list literal
    if raw_str.startswith("[") and raw_str.endswith("]"):
        try:
            parsed = ast.literal_eval(raw_str)
            if isinstance(parsed, (list, tuple)):
                items = [str(x).strip().lower() for x in parsed if str(x).strip()]
                return ", ".join(items)
        except (ValueError, SyntaxError):
            pass

    # Fallback: remove enclosing brackets and quotes, then split by commas
    cleaned_str = raw_str.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    items = [part.strip().lower() for part in cleaned_str.split(",") if part.strip()]
    return ", ".join(items)


def parse_nutrition(raw_nutrition):
    """
    Parses the Food.com nutrition column which follows the order:
    [Calories, Fat, Sugar, Sodium, Protein, Saturated Fat, Carbohydrates]
    Returns a list of 7 floats, using np.nan for missing/invalid values.
    """
    nan_list = [np.nan] * len(NUTRITION_COLUMNS)
    if pd.isna(raw_nutrition):
        return nan_list

    parsed_items = None
    if isinstance(raw_nutrition, (list, tuple)):
        parsed_items = raw_nutrition
    elif isinstance(raw_nutrition, str):
        raw_str = raw_nutrition.strip()
        if raw_str.startswith("[") and raw_str.endswith("]"):
            try:
                val = ast.literal_eval(raw_str)
                if isinstance(val, (list, tuple)):
                    parsed_items = val
            except (ValueError, SyntaxError):
                pass

        if parsed_items is None:
            cleaned_str = raw_str.replace("[", "").replace("]", "")
            parsed_items = [x.strip() for x in cleaned_str.split(",") if x.strip()]

    if not parsed_items:
        return nan_list

    result = []
    for i in range(len(NUTRITION_COLUMNS)):
        if i < len(parsed_items):
            try:
                result.append(float(parsed_items[i]))
            except (ValueError, TypeError):
                result.append(np.nan)
        else:
            result.append(np.nan)
    return result


def parse_steps(raw_steps):
    """
    Safely parses recipe preparation instructions into a list of step strings.
    Handles Python-style list strings, native lists, plain strings, and missing values.
    """
    if pd.isna(raw_steps):
        return []

    # If already a list or tuple
    if isinstance(raw_steps, (list, tuple)):
        return [str(s).strip() for s in raw_steps if str(s).strip()]

    raw_str = str(raw_steps).strip()
    if not raw_str:
        return []

    # Attempt safe Python list literal evaluation
    if raw_str.startswith("[") and raw_str.endswith("]"):
        try:
            parsed = ast.literal_eval(raw_str)
            if isinstance(parsed, (list, tuple)):
                return [str(s).strip() for s in parsed if str(s).strip()]
        except (ValueError, SyntaxError):
            pass

    # Fallback: remove enclosing brackets and split by quoted items or commas
    cleaned = raw_str.strip("[]")
    if "', '" in cleaned or '", "' in cleaned:
        delimiter = "', '" if "', '" in cleaned else '", "'
        parts = cleaned.split(delimiter)
        return [p.strip(" '\"\n\r\t") for p in parts if p.strip(" '\"\n\r\t")]

    parts = cleaned.split("\n") if "\n" in cleaned else cleaned.split(",")
    steps = [p.strip(" '\"\n\r\t") for p in parts if p.strip(" '\"\n\r\t")]
    return steps if steps else [raw_str]


# ==============================================================================
# Cached Data Loading & Preprocessing
# ==============================================================================
@st.cache_data(show_spinner=False)
def load_and_preprocess_dataset(file_source, sample_size=DEFAULT_SAMPLE_SIZE, random_state=RANDOM_STATE):
    """
    Loads and preprocesses the recipes dataset:
    - Validates presence of required columns
    - Removes invalid/empty recipe names and ingredients
    - Performs reproducible sampling if larger than sample_size
    - Cleans ingredient text
    - Parses nutrition into 7 separate numeric columns
    - Cleans and converts minutes to integer
    """
    if hasattr(file_source, "seek"):
        file_source.seek(0)

    # Read CSV file
    df = pd.read_csv(file_source)

    # Validate required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"The uploaded dataset is missing required column(s): {', '.join(missing_cols)}. "
            f"Expected columns: {', '.join(REQUIRED_COLUMNS)}."
        )

    # Step 1: Drop rows with null or blank names and ingredients
    df = df.dropna(subset=["name", "ingredients"]).copy()
    df = df[
        df["name"].astype(str).str.strip().ne("") &
        df["ingredients"].astype(str).str.strip().ne("")
    ].copy()

    if df.empty:
        raise ValueError("The dataset contains no valid recipe entries after removing empty names and ingredients.")

    # Step 2: Reproducible sampling for performance if dataset is larger
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state).copy()

    df = df.reset_index(drop=True)

    # Step 3: Clean ingredients into lowercase, quote/bracket-free searchable text
    df["cleaned_ingredients"] = df["ingredients"].apply(clean_ingredients)
    df["ingredients"] = df["cleaned_ingredients"]
    # Filter out rows whose cleaned ingredients ended up empty
    df = df[df["cleaned_ingredients"].str.strip().ne("")].copy()

    if df.empty:
        raise ValueError("No recipes with valid ingredients could be extracted from this dataset.")

    # Step 4: Parse nutrition values into 7 separate numeric columns
    nutrition_parsed = df["nutrition"].apply(parse_nutrition).tolist()
    nutrition_df = pd.DataFrame(nutrition_parsed, columns=NUTRITION_COLUMNS, index=df.index)
    for col in NUTRITION_COLUMNS:
        df[col] = nutrition_df[col]

    # Step 5: Convert minutes to numeric, clipping at 0 and filling invalid values with 0
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0).clip(lower=0).astype(int)

    return df.reset_index(drop=True)


# ==============================================================================
# TF-IDF Vectorization & Recommendation Engine
# ==============================================================================
def clean_user_input(raw_input):
    """
    Cleans the user's input ingredients:
    - Converts text to lowercase
    - Strips whitespace around individual ingredients
    - Formats as clean searchable text
    """
    if not raw_input or not isinstance(raw_input, str):
        return ""
    tokens = [item.strip().lower() for item in raw_input.split(",") if item.strip()]
    return ", ".join(tokens)


@st.cache_resource(show_spinner=False)
def build_tfidf_model(ingredients_series):
    """
    Fits a TF-IDF vectorizer on the dataset's cleaned ingredients:
    - Creates TfidfVectorizer(stop_words="english")
    - Fits the vectorizer
    - Creates the TF-IDF matrix
    - Returns the vectorizer and matrix
    """
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(ingredients_series)
    except ValueError:
        vectorizer = TfidfVectorizer(stop_words=None)
        tfidf_matrix = vectorizer.fit_transform(ingredients_series)
    return vectorizer, tfidf_matrix


def min_max_scale(series):
    """
    Safely normalizes a pandas Series to the range [0.0, 1.0].
    Fills NaN values with the median (or 0.0 if all NaN) to prevent app crashes.
    Avoids division by zero when min and max are equal.
    """
    fill_value = series.median() if not pd.isna(series.median()) else 0.0
    s_clean = series.fillna(fill_value)
    min_val = s_clean.min()
    max_val = s_clean.max()
    if max_val - min_val > 1e-6:
        return (s_clean - min_val) / (max_val - min_val)
    return pd.Series(0.5, index=series.index)


def rank_recipes_by_goal(candidates_df, goal):
    """
    Calculates nutritional suitability and a weighted composite final score:
    - Balanced: protein balance, reasonable calories, moderate carbs and fat
    - High Protein: prioritized higher protein content
    - Low Calorie: prioritized lower calories
    - Quick Prep: prioritized shorter preparation time

    Returns candidates_df with 'nutrition_score' and 'final_score' columns,
    sorted by 'final_score' in descending order.
    """
    if candidates_df.empty:
        return candidates_df

    df = candidates_df.copy()

    # Step 1: Normalize relevant nutrient values and prep time to [0, 1]
    norm_calories = min_max_scale(df["calories"])
    norm_protein = min_max_scale(df["protein"])
    norm_carbs = min_max_scale(df["carbohydrates"])
    norm_fat = min_max_scale(df["fat"])
    norm_time = min_max_scale(df["minutes"])

    # Step 2: Compute goal-specific nutritional/time suitability score
    if goal == "High Protein":
        # Prioritize higher protein content
        nutrition_score = norm_protein
        final_score = (
            WEIGHT_SIMILARITY_STANDARD * df["similarity_score"]
            + WEIGHT_NUTRITION_STANDARD * nutrition_score
        )

    elif goal == "Low Calorie":
        # Prioritize lower calories (inverse of normalized calories)
        nutrition_score = 1.0 - norm_calories
        final_score = (
            WEIGHT_SIMILARITY_STANDARD * df["similarity_score"]
            + WEIGHT_NUTRITION_STANDARD * nutrition_score
        )

    elif goal == "Quick Prep":
        # Prioritize shorter preparation time and moderate nutrition
        time_suitability = 1.0 - norm_time
        nutrition_score = 0.5 * norm_protein + 0.5 * (1.0 - norm_calories)
        final_score = (
            WEIGHT_SIMILARITY_QUICK_PREP * df["similarity_score"]
            + WEIGHT_TIME_QUICK_PREP * time_suitability
            + WEIGHT_NUTRITION_QUICK_PREP * nutrition_score
        )

    else:  # "Balanced"
        # Prioritize good protein, reasonable calories, moderate carbs & fat
        cal_balance = np.clip(1.0 - np.abs(norm_calories - 0.5) * 2, 0.0, 1.0)
        carb_balance = np.clip(1.0 - np.abs(norm_carbs - 0.5) * 2, 0.0, 1.0)
        fat_balance = np.clip(1.0 - np.abs(norm_fat - 0.5) * 2, 0.0, 1.0)

        nutrition_score = (
            0.40 * norm_protein
            + 0.30 * cal_balance
            + 0.15 * carb_balance
            + 0.15 * fat_balance
        )
        final_score = (
            WEIGHT_SIMILARITY_STANDARD * df["similarity_score"]
            + WEIGHT_NUTRITION_STANDARD * nutrition_score
        )

    df["nutrition_score"] = nutrition_score
    df["final_score"] = final_score

    # Sort descending by final composite score
    return df.sort_values(by="final_score", ascending=False)


def recommend_recipes(
    user_query,
    df,
    vectorizer,
    tfidf_matrix,
    goal="Balanced",
    max_prep_time=60,
    top_n=3,
    min_threshold=MIN_SIMILARITY_THRESHOLD,
):
    """
    Computes recipe recommendations combining TF-IDF similarity and multi-criteria dietary ranking:
    1. Transforms user query using fitted TF-IDF vectorizer
    2. Calculates cosine similarity with all recipes
    3. Filters by minimum similarity threshold and maximum preparation time
    4. Re-ranks matching candidates using nutritional criteria for the chosen goal
    5. Returns top matching recipes and execution status
    """
    cleaned_query = clean_user_input(user_query)
    if not cleaned_query:
        return pd.DataFrame(), "EMPTY_INPUT"

    # Transform user query using the fitted TF-IDF vectorizer
    user_vector = vectorizer.transform([cleaned_query])

    # Check if the query has any recognized words in vocabulary
    if user_vector.nnz == 0:
        return pd.DataFrame(), "VOCAB_MISMATCH"

    # Calculate cosine similarity with all recipes
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

    # Add similarity_score column
    scored_df = df.copy()
    scored_df["similarity_score"] = similarity_scores

    # Step 1: Filter by minimum similarity threshold
    similarity_matches = scored_df[scored_df["similarity_score"] >= min_threshold]

    if similarity_matches.empty:
        return pd.DataFrame(), "NO_MATCH"

    # Step 2: Apply preparation-time filter (minutes <= max_prep_time)
    filtered_df = similarity_matches[similarity_matches["minutes"] <= max_prep_time]

    if filtered_df.empty:
        return pd.DataFrame(), "NO_MATCH_TIME"

    # Step 3: Re-rank matching recipes using selected dietary goal
    ranked_df = rank_recipes_by_goal(filtered_df, goal)

    return ranked_df.head(top_n), "SUCCESS"



# ==============================================================================
# Section 1: Dataset
# ==============================================================================
def render_dataset_section():
    """Section for uploading, loading, inspecting, and managing recipe datasets."""
    st.header("1. Dataset")

    # Check for local file availability
    local_exists = os.path.exists(LOCAL_CSV_FILENAME)

    col_upload, col_local = st.columns([2, 1])
    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload Food.com RAW_recipes.csv",
            type=["csv"],
            help="Upload the Food.com RAW_recipes.csv file",
        )
    with col_local:
        st.write("**Local Dataset Detection**")
        if local_exists:
            st.success(f"Found `{LOCAL_CSV_FILENAME}` in project directory.")
        else:
            st.caption(f"`{LOCAL_CSV_FILENAME}` not detected in local directory.")

    # Determine file source
    file_source = None
    source_label = None

    if uploaded_file is not None:
        file_source = uploaded_file
        source_label = f"Uploaded file (`{uploaded_file.name}`)"
    elif local_exists:
        file_source = LOCAL_CSV_FILENAME
        source_label = f"Local file (`{LOCAL_CSV_FILENAME}`)"

    # If no file is available, show instructional message and exit early
    if file_source is None:
        if "current_dataset_source" in st.session_state:
            del st.session_state["current_dataset_source"]
        st.session_state["recommendation_status"] = None
        st.session_state["recommendation_results"] = None
        st.warning(
            "⚠️ No dataset found. Please upload `RAW_recipes.csv` using the file uploader above, "
            "or place `RAW_recipes.csv` in the project root directory."
        )
        return None

    # Load and preprocess dataset with clear error handling
    try:
        with st.spinner("Loading and preprocessing dataset..."):
            df = load_and_preprocess_dataset(file_source)

        # Invalidate cached recommendations if dataset source changed
        if st.session_state.get("current_dataset_source") != source_label:
            st.session_state["current_dataset_source"] = source_label
            st.session_state["recommendation_status"] = None
            st.session_state["recommendation_results"] = None

        st.success(f"✅ Dataset loaded successfully from {source_label}!")

        # Dataset summary metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total Recipes", f"{len(df):,}")
        with m2:
            st.metric("Total Columns", f"{len(df.columns)}")
        with m3:
            st.metric("Sampling State", f"Reproducible (seed {RANDOM_STATE})")

        # Data Preview
        st.subheader("Dataset Preview")
        preview_columns = [
            c for c in [
                "name", "minutes", "cleaned_ingredients", "calories",
                "protein", "carbohydrates", "fat", "sugar", "sodium"
            ] if c in df.columns
        ]
        st.dataframe(df[preview_columns].head(5), use_container_width=True)

        return df

    except ValueError as val_err:
        st.error(f"⚠️ Validation Error: {val_err}")
        return None
    except Exception as err:
        st.error(f"❌ Failed to parse and load the dataset: {err}")
        return None


# ==============================================================================
# Section 2: User Ingredients
# ==============================================================================
def render_user_ingredients_section(df):
    """Section for capturing user-available ingredients."""
    st.header("2. User Ingredients")

    if df is None:
        st.info("ℹ️ Please load or upload a recipe dataset in Section 1 to search for recipes.")
        return ""

    user_input = st.text_input(
        "Enter ingredients you have (separated by commas):",
        placeholder="chicken, rice, onion, tomato, garlic",
        help="Type the ingredients available in your pantry, separated by commas.",
        key="user_ingredient_input",
    )
    return user_input


# ==============================================================================
# Section 3: Dietary Goal
# ==============================================================================
def render_dietary_goal_section(df, user_input):
    """
    Section for setting dietary goals, preparation-time filters, and triggering recommendation.
    """
    st.header("3. Dietary Goal")

    if df is None:
        st.info("ℹ️ Load a dataset in Section 1 to set dietary goals and filters.")
        return

    col1, col2 = st.columns(2)
    with col1:
        goal = st.radio(
            "Select your dietary goal:",
            options=["Balanced", "High Protein", "Low Calorie", "Quick Prep"],
            horizontal=False,
            help="Prioritizes nutritional balance and prep time according to your chosen goal.",
            key="dietary_goal_radio",
        )
    with col2:
        max_time = st.slider(
            "Maximum preparation time (minutes):",
            min_value=5,
            max_value=300,
            value=60,
            step=5,
            help="Filter out recipes that take longer than this to prepare.",
            key="prep_time_slider",
        )

    find_button = st.button("🔍 Find Matching Recipes", type="primary", key="find_recipes_btn")

    if find_button:
        cleaned_query = clean_user_input(user_input)
        if not cleaned_query:
            st.session_state["recommendation_status"] = "EMPTY_INPUT"
            st.session_state["recommendation_results"] = None
        else:
            with st.spinner("Finding and ranking best recipe matches..."):
                vectorizer, tfidf_matrix = build_tfidf_model(df["cleaned_ingredients"])
                recs, status = recommend_recipes(
                    user_query=cleaned_query,
                    df=df,
                    vectorizer=vectorizer,
                    tfidf_matrix=tfidf_matrix,
                    goal=goal,
                    max_prep_time=max_time,
                    top_n=3,
                    min_threshold=MIN_SIMILARITY_THRESHOLD,
                )
                st.session_state["recommendation_status"] = status
                st.session_state["recommendation_results"] = recs
                st.session_state["selected_goal"] = goal
                st.session_state["selected_max_time"] = max_time
                st.session_state["user_input_query"] = user_input



# ==============================================================================
# Section 4: Recipe Recommendations
# ==============================================================================
def render_recipe_recommendations_section(df):
    """Section for displaying top 3 matched and re-ranked recipes with nutritional details."""
    st.header("4. Recipe Recommendations")

    if df is None:
        st.info("ℹ️ Recipe recommendations will be displayed here once a dataset is loaded.")
        return

    status = st.session_state.get("recommendation_status")
    results = st.session_state.get("recommendation_results")

    if status is None:
        st.info("💡 Enter your ingredients in Section 2, select your dietary goal, and click **Find Matching Recipes**.")
    elif status == "EMPTY_INPUT":
        st.warning("⚠️ Please enter at least one ingredient to find matching recipes.")
    elif status == "VOCAB_MISMATCH":
        st.warning(
            "⚠️ None of the entered ingredients were recognized in the recipe dataset vocabulary. "
            "Please check your spelling or try common ingredients (e.g., chicken, rice, onion, tomato, garlic)."
        )
    elif status == "NO_MATCH":
        st.warning(
            "⚠️ No recipes matched your ingredients above the minimum similarity threshold. "
            "Try adding different or more common ingredients."
        )
    elif status == "NO_MATCH_TIME":
        max_t = st.session_state.get("selected_max_time", 60)
        st.warning(
            f"⚠️ Matching recipes were found, but none have a preparation time under {max_t} minutes. "
            "Try increasing the maximum preparation time slider."
        )
    elif status == "SUCCESS" and results is not None and not results.empty:
        goal = st.session_state.get("selected_goal", "Balanced")
        top_candidates = results.head(3)
        count = len(top_candidates)
        rec_title = f"Top {count} Recipe Recommendations" if count > 1 else "Top Recipe Recommendation"
        st.success(f"🎉 {rec_title} for Goal: **{goal}**")

        for idx, (_, row) in enumerate(top_candidates.iterrows(), start=1):
            with st.container():
                final_pct = f"{row['final_score'] * 100:.1f}%" if pd.notna(row.get('final_score')) else "N/A"
                sim_pct = f"{row['similarity_score'] * 100:.1f}%" if pd.notna(row.get('similarity_score')) else "N/A"
                rank_badge = f"#{idx}"
                recipe_name = str(row['name']).title() if pd.notna(row.get('name')) else "Untitled Recipe"

                # Recipe Rank & Name
                st.subheader(f"{rank_badge} {recipe_name}")

                # Key Metric Badges: Final Score, Similarity Score, Prep Time
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🏆 Final Score", final_pct)
                with col2:
                    st.metric("🎯 Ingredient Similarity", sim_pct)
                with col3:
                    prep_mins = f"{int(row['minutes'])} mins" if pd.notna(row.get('minutes')) else "N/A"
                    st.metric("⏱️ Preparation Time", prep_mins)

                # Nutritional Breakdown (Safe formatting with NaN handling)
                n1, n2, n3, n4 = st.columns(4)
                with n1:
                    cal_str = f"{row['calories']:.0f} kcal" if pd.notna(row.get('calories')) else "N/A"
                    st.write(f"🔥 **Calories:** {cal_str}")
                with n2:
                    prot_str = f"{row['protein']:.1f} g" if pd.notna(row.get('protein')) else "N/A"
                    st.write(f"🥩 **Protein:** {prot_str}")
                with n3:
                    carb_str = f"{row['carbohydrates']:.1f} g" if pd.notna(row.get('carbohydrates')) else "N/A"
                    st.write(f"🍞 **Carbohydrates:** {carb_str}")
                with n4:
                    fat_str = f"{row['fat']:.1f} g" if pd.notna(row.get('fat')) else "N/A"
                    st.write(f"🥑 **Fat:** {fat_str}")

                # Cleaned ingredients
                st.write(f"🥗 **Ingredients:** {row.get('cleaned_ingredients', 'N/A')}")

                # Preparation Instructions Expander
                with st.expander("👨‍🍳 Preparation Instructions"):
                    steps_data = row.get("steps", None) if pd.notna(row.get("steps", None)) else None
                    step_items = parse_steps(steps_data)
                    if step_items:
                        for step_num, step_text in enumerate(step_items, start=1):
                            st.markdown(f"**Step {step_num}:** {step_text}")
                    else:
                        st.info("No preparation instructions available for this recipe.")

                # Visual separation between recipe cards
                st.divider()



# ==============================================================================
# Section 5: Nutrition Analysis
# ==============================================================================
def render_nutrition_analysis_section():
    """Displays nutritional charts, summary metrics, and goal-based analytics for top 3 recipes."""
    st.header("5. Nutrition Analysis")

    status = st.session_state.get("recommendation_status")
    results = st.session_state.get("recommendation_results")
    goal = st.session_state.get("selected_goal", "Balanced")

    if status != "SUCCESS" or results is None or results.empty:
        st.info("💡 Nutritional analytics will appear here after recipe recommendations are generated.")
        return

    top3 = results.head(3).copy()
    count = len(top3)

    # Step 1: Summary Average Metrics
    header_count = f"Top {count} Recommendations" if count > 1 else "Top Recommendation"
    st.subheader(f"Nutritional Averages ({header_count})")
    avg_cal = top3["calories"].mean()
    avg_prot = top3["protein"].mean()
    avg_carbs = top3["carbohydrates"].mean()
    avg_fat = top3["fat"].mean()

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Avg Calories", f"{avg_cal:.0f} kcal" if pd.notna(avg_cal) else "N/A")
    with m2:
        st.metric("Avg Protein", f"{avg_prot:.1f} g" if pd.notna(avg_prot) else "N/A")
    with m3:
        st.metric("Avg Carbs", f"{avg_carbs:.1f} g" if pd.notna(avg_carbs) else "N/A")
    with m4:
        st.metric("Avg Fat", f"{avg_fat:.1f} g" if pd.notna(avg_fat) else "N/A")

    st.divider()

    # Step 2: Nutrition Comparison Charts
    st.subheader("Nutritional Comparison")
    chart_data = top3[["name", "calories", "protein", "carbohydrates", "fat"]].copy()
    chart_data["short_name"] = chart_data["name"].apply(
        lambda n: str(n).title()[:22] + "..." if len(str(n)) > 22 else str(n).title()
    )

    col_cal, col_macro = st.columns(2)
    with col_cal:
        st.write("##### ⚡ Calories (kcal)")
        cal_df = chart_data.set_index("short_name")[["calories"]].fillna(0)
        cal_df.columns = ["Calories"]
        st.bar_chart(cal_df, use_container_width=True)

    with col_macro:
        st.write("##### ⚖️ Macronutrients (grams)")
        macro_df = chart_data.set_index("short_name")[["protein", "carbohydrates", "fat"]].fillna(0)
        macro_df.columns = ["Protein (g)", "Carbs (g)", "Fat (g)"]
        st.bar_chart(macro_df, use_container_width=True)

    st.divider()

    # Step 3: Standout Highlights & Dynamic Goal Interpretation
    st.subheader("Recipe Highlights & Goal Insights")

    valid_prot = top3.dropna(subset=["protein"])
    valid_cal = top3.dropna(subset=["calories"])
    valid_time = top3.dropna(subset=["minutes"])

    best_prot_name, best_prot_val = ("N/A", "N/A")
    if not valid_prot.empty:
        p_row = valid_prot.loc[valid_prot["protein"].idxmax()]
        best_prot_name = str(p_row["name"]).title()
        best_prot_val = f"{p_row['protein']:.1f} g"

    best_cal_name, best_cal_val = ("N/A", "N/A")
    if not valid_cal.empty:
        c_row = valid_cal.loc[valid_cal["calories"].idxmin()]
        best_cal_name = str(c_row["name"]).title()
        best_cal_val = f"{c_row['calories']:.0f} kcal"

    best_time_name, best_time_val = ("N/A", "N/A")
    if not valid_time.empty:
        t_row = valid_time.loc[valid_time["minutes"].idxmin()]
        best_time_name = str(t_row["name"]).title()
        best_time_val = f"{int(t_row['minutes'])} mins"

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("🥩 **Highest Protein**")
        st.write(f"**{best_prot_name}**")
        st.caption(f"Protein: `{best_prot_val}`")

    with c2:
        st.markdown("🥗 **Lowest Calories**")
        st.write(f"**{best_cal_name}**")
        st.caption(f"Calories: `{best_cal_val}`")

    with c3:
        st.markdown("⏱️ **Shortest Prep Time**")
        st.write(f"**{best_time_name}**")
        st.caption(f"Time: `{best_time_val}`")

    st.write("##### 🎯 Dynamic Goal Interpretation")
    if goal == "High Protein":
        if best_prot_name != "N/A":
            st.info(
                f"🏋️ **High Protein Focus**: **{best_prot_name}** provides the highest protein density "
                f"at **{best_prot_val}**, making it an optimal selection for post-workout recovery and muscle building."
            )
        else:
            st.info("🏋️ **High Protein Focus**: Protein metrics were not available for the recommended recipes.")
    elif goal == "Low Calorie":
        if best_cal_name != "N/A":
            st.info(
                f"🌿 **Low Calorie Focus**: **{best_cal_name}** is the lightest meal with only "
                f"**{best_cal_val}**, allowing you to enjoy a flavorful meal while maintaining a caloric deficit."
            )
        else:
            st.info("🌿 **Low Calorie Focus**: Calorie metrics were not available for the recommended recipes.")
    elif goal == "Quick Prep":
        if best_time_name != "N/A":
            st.info(
                f"⚡ **Quick Prep Focus**: **{best_time_name}** is your fastest option ready in just "
                f"**{best_time_val}**, perfectly tailored for busy schedules without compromising on taste."
            )
        else:
            st.info("⚡ **Quick Prep Focus**: Preparation times were not available for the recommended recipes.")
    else:  # Balanced
        cal_str = f"**{avg_cal:.0f} kcal**" if pd.notna(avg_cal) else "moderate calories"
        prot_str = f"**{avg_prot:.1f}g protein**" if pd.notna(avg_prot) else "balanced protein"
        carb_str = f"**{avg_carbs:.1f}g carbs**" if pd.notna(avg_carbs) else "moderate carbs"
        fat_str = f"**{avg_fat:.1f}g fat**" if pd.notna(avg_fat) else "healthy fats"
        st.info(
            f"🥗 **Balanced Nutrition Focus**: The top recipes maintain an average of {cal_str} "
            f"with {prot_str}, {carb_str}, and {fat_str}, "
            f"offering steady macronutrient distribution for sustained daily energy."
        )


# ==============================================================================
# Section 6: Health & Food Waste Insights
# ==============================================================================
def check_ingredient_match(user_item, recipe_text):
    """
    Checks if a user ingredient token appears within recipe ingredients text.
    Handles word boundaries and basic singular/plural forms to avoid false partial matches.
    """
    u = user_item.strip().lower()
    text = recipe_text.strip().lower()
    if not u or not text:
        return False

    variants = [u]
    if u.endswith("es") and len(u) > 4:
        variants.append(u[:-2])
    elif u.endswith("s") and len(u) > 3:
        variants.append(u[:-1])
    else:
        variants.append(u + "s")
        variants.append(u + "es")

    for v in variants:
        if re.search(r"\b" + re.escape(v) + r"\b", text):
            return True
    return False


def render_health_and_food_waste_insights_section(user_input=None):
    """
    Section 6: Health and Food Waste Insights
    Calculates dynamic food waste reduction metrics and non-medical nutritional insights
    based on user ingredients, top 3 recommendations, and the selected dietary goal.
    """
    st.header("6. Health & Food Waste Insights")

    status = st.session_state.get("recommendation_status")
    results = st.session_state.get("recommendation_results")
    goal = st.session_state.get("selected_goal", "Balanced")

    # Retrieve user ingredients from parameter or session state
    raw_query = (
        user_input
        or st.session_state.get("user_input_query", "")
        or st.session_state.get("user_ingredient_input", "")
    )

    if status != "SUCCESS" or results is None or results.empty or not str(raw_query).strip():
        st.info("💡 Dynamic health and food waste insights will appear here once recipes are recommended.")
        return

    top3 = results.head(3).copy()

    # Parse individual user ingredients
    user_items = [item.strip().lower() for item in str(raw_query).split(",") if item.strip()]

    if not user_items:
        st.info("💡 Please enter at least one ingredient to generate food waste and nutrition insights.")
        return

    # ==========================================================================
    # 1. Food Waste Reduction Insight
    # ==========================================================================
    st.subheader("♻️ Food Waste Reduction & Pantry Utilization")

    # Combine ingredients of all top 3 recipes for aggregate utilization
    all_recipe_ingredients = " ".join(top3["cleaned_ingredients"].astype(str).tolist()).lower()

    used_ingredients = []
    unused_ingredients = []
    for item in user_items:
        if check_ingredient_match(item, all_recipe_ingredients):
            used_ingredients.append(item)
        else:
            unused_ingredients.append(item)

    total_count = len(user_items)
    used_count = len(used_ingredients)
    utilization_pct = (used_count / total_count * 100) if total_count > 0 else 0.0

    # Display Metrics
    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1:
        st.metric("Pantry Utilization", f"{utilization_pct:.0f}%")
    with col_w2:
        st.metric("Ingredients Utilized", f"{used_count} of {total_count}")
    with col_w3:
        st.metric("Leftover Items", f"{len(unused_ingredients)}")

    # Detailed Utilization Breakdown
    c_used, c_unused = st.columns(2)
    with c_used:
        st.write("**Ingredients Utilized in Top Recipes:**")
        if used_ingredients:
            st.success(", ".join([f"`{ing}`" for ing in used_ingredients]))
        else:
            st.write("None of your listed ingredients matched the top recipes.")

    with c_unused:
        st.write("**Ingredients Not Utilized (Potential Leftovers):**")
        if unused_ingredients:
            st.warning(", ".join([f"`{ing}`" for ing in unused_ingredients]))
        else:
            st.success("None! All of your pantry items are put to use.")

    # Find which individual recipe uses the highest number of user ingredients
    def count_matched(row_ingredients):
        return sum(1 for item in user_items if check_ingredient_match(item, str(row_ingredients)))

    top3["user_items_used"] = top3["cleaned_ingredients"].apply(count_matched)
    best_utilizer_idx = top3["user_items_used"].idxmax()
    best_utilizer_row = top3.loc[best_utilizer_idx]
    best_recipe_used_count = int(best_utilizer_row["user_items_used"])

    if best_recipe_used_count > 0:
        st.success(
            f"🌟 **Top Pantry Champion**: **{str(best_utilizer_row['name']).title()}** uses "
            f"**{best_recipe_used_count} of your {total_count}** ingredients, making it the most "
            f"efficient choice for minimizing leftover groceries."
        )

    if unused_ingredients:
        unused_str = ", ".join([f"**{item}**" for item in unused_ingredients])
        st.warning(
            f"💡 **Food Waste Tip**: Consider using {unused_str} in a side dish, soup, "
            f"or storing/freezing properly to prevent unnecessary spoilage."
        )
    else:
        st.success("🎉 **Zero Waste Potential**: Every single ingredient you entered is utilized in these recommendations!")

    st.divider()

    # ==========================================================================
    # 2. Health & Nutrition Insights (Non-Medical)
    # ==========================================================================
    st.subheader("🥗 Health & Nutrition Guidance")

    valid_prot = top3.dropna(subset=["protein"])
    valid_cal = top3.dropna(subset=["calories"])
    valid_time = top3.dropna(subset=["minutes"])

    # Extract standout recipes for dynamic insights
    best_prot_row = valid_prot.loc[valid_prot["protein"].idxmax()] if not valid_prot.empty else None
    best_cal_row = valid_cal.loc[valid_cal["calories"].idxmin()] if not valid_cal.empty else None
    fastest_row = valid_time.loc[valid_time["minutes"].idxmin()] if not valid_time.empty else None

    # Goal-Specific Dynamic Observation
    if goal == "High Protein" and best_prot_row is not None:
        p_val = best_prot_row["protein"]
        st.success(
            f"💪 **Protein Density Highlight**: **{str(best_prot_row['name']).title()}** provides "
            f"**{p_val:.1f}g of protein**, delivering the highest protein concentration among your recommendations "
            f"to support muscle recovery and prolonged satiety."
        )

    elif goal == "Low Calorie" and best_cal_row is not None:
        cal_val = best_cal_row["calories"]
        st.success(
            f"🍃 **Calorie Management Highlight**: **{str(best_cal_row['name']).title()}** has the lowest "
            f"energy density at **{cal_val:.0f} kcal**, providing a nutrient-rich, satisfying meal that "
            f"easily fits into calorie-conscious dietary goals."
        )

    elif goal == "Quick Prep" and fastest_row is not None:
        time_val = int(fastest_row["minutes"])
        st.success(
            f"⏱️ **Efficiency & Fresh Eating**: **{str(fastest_row['name']).title()}** is ready in just "
            f"**{time_val} minutes**, making home-cooked dining realistic on busy schedules without relying on processed fast food."
        )

    else:  # Balanced
        avg_cal = top3["calories"].mean()
        avg_prot = top3["protein"].mean()
        avg_carbs = top3["carbohydrates"].mean()
        avg_fat = top3["fat"].mean()
        cal_str = f"**{avg_cal:.0f} kcal**" if pd.notna(avg_cal) else "balanced calories"
        prot_str = f"**{avg_prot:.1f}g protein**" if pd.notna(avg_prot) else "balanced protein"
        carb_str = f"**{avg_carbs:.1f}g carbohydrates**" if pd.notna(avg_carbs) else "moderate carbohydrates"
        fat_str = f"**{avg_fat:.1f}g dietary fat**" if pd.notna(avg_fat) else "healthy fats"
        st.success(
            f"⚖️ **Balanced Macronutrient Profile**: Across your top recommendations, the average meal provides "
            f"{cal_str} with {prot_str}, {carb_str}, and {fat_str}, "
            f"offering steady glycemic release and comprehensive nutrition."
        )

    # Actionable Nutrition Guidance
    if best_prot_row is not None and best_cal_row is not None:
        st.info(
            f"📌 **Actionable Advice**: If your priority today is muscle repair or prolonged fullness, select "
            f"**{str(best_prot_row['name']).title()}** ({best_prot_row['protein']:.1f}g protein). "
            f"If you prefer a lighter, lower-calorie meal, go with **{str(best_cal_row['name']).title()}** "
            f"({best_cal_row['calories']:.0f} kcal)."
        )


# ==============================================================================
# Main Application Flow
# ==============================================================================
def main():
    df = render_dataset_section()
    user_input = render_user_ingredients_section(df)
    render_dietary_goal_section(df, user_input)
    render_recipe_recommendations_section(df)
    render_nutrition_analysis_section()
    render_health_and_food_waste_insights_section(user_input)


if __name__ == "__main__":
    main()


