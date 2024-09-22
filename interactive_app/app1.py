# Streamlit imports
import streamlit as st

# Standard imports
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lime
import lime.lime_tabular

# Set page configuration
st.set_page_config(page_title="Who's likely to love their job? A Predictive Tool for Job Satisfaction and Key Drivers", layout="wide")

# Load models and datasets
rf_model = joblib.load('./models/best_model_rf.pkl')  # Random Forest model
best_model_logit = joblib.load('./models/best_model_logit.pkl')  # Logit model
X_test_readable = joblib.load('./preprocessed_variables/X_test_readable.pkl')  # Readable test data
X_train_resampled_df = joblib.load('./preprocessed_variables/X_train_resampled_df.pkl')  # Resampled train data
X_test_scaled_df = joblib.load('./preprocessed_variables/X_test_scaled_df.pkl')  # Scaled test data

# Ensure X_test_readable and X_test_scaled_df are loaded
if X_test_readable is None or X_test_scaled_df is None:
    st.error("Error loading the data. Please check if the files exist.")
    st.stop()

X_test_readable = X_test_readable.reset_index(drop=True)
X_test_scaled_df = X_test_scaled_df.reset_index(drop=True)

# Logistic Regression Feature Name Mapping (for odds ratios)
feature_name_mapping_logit = {
    'satisfaction_advancement': 'Satisfaction with Career Advancement',
    'job_duration_months': 'Job Duration (Months)',
    'job_degree_relation': 'Job Related to Degree',
    'salary': 'Salary',
    'marital_status_Married': 'Marital Status: Married',
    'employer_sector_Government': 'Employer Sector: Government',
    'employer_sector_Educational Institution': 'Employer Sector: Educational Institution',
    'marital_status_Widowed': 'Marital Status: Widowed',
    'occupation_name_Psychologists, including clinical': 'Occupation: Psychologists (Including Clinical)',
    'employer_size_category_Small': 'Employer Size: Small',
    'occupation_name_Clergy and other religious workers': 'Occupation: Clergy and Religious Workers',
    'ethnicity_White': 'Ethnicity: White',
    'recent_degree_field_Physical therapy and other rehabilitation/therapeutic services': 'Degree in Physical Therapy or Rehabilitation Services',
    'occupation_name_RNs, pharmacists, dieticians, therapists, physician assistants, nurse practitioners': 'Occupation: RNs, Pharmacists, Therapists, etc.',
    'occupation_name_Electrical and electronics engineers': 'Occupation: Electrical and Electronics Engineers',
    'children_in_household': 'Children in Household',
    'recent_degree_field_Audiology and speech pathology': 'Degree in Audiology and Speech Pathology',
    'occupation_name_Computer system analysts': 'Occupation: Computer System Analysts',
    'recent_degree_field_Atmospheric sciences and meteorology': 'Degree in Atmospheric Sciences and Meteorology',
    'occupation_name_Electrical, electronic, industrial, and mechanical technicians': 'Occupation: Electrical, Industrial, and Mechanical Technicians'
}

# MDI Feature Name Mapping (for Random Forest)
feature_name_mapping_top_10 = {
    'num__satisfaction_advancement': 'Satisfaction with Career Advancement',
    'num__satisfaction_challenges': 'Satisfaction with Intellectual Challenge during Job',
    'num__satisfaction_salary': 'Satisfaction with Salary',
    'num__satisfaction_security': 'Satisfaction with Job Security',
    'num__satisfaction_location': 'Satisfaction with Job Location',
    'num__salary': 'Salary',
    'num__age': 'Age',
    'cat__survey_year_2021-01-01': 'Survey Year: 2021',
    'num__satisfaction_benefits': 'Satisfaction with Job Benefits',
    'num__job_duration_months': 'Job Duration (Months)'
}

# Home Page Function
def home_page():
    st.title("Who's likely to love their job? A Predictive Tool for Job Satisfaction and Key Drivers")

    st.markdown("""
    ### Welcome!

    This tool is designed to help you understand the factors that drive job satisfaction based on various job-related attributes. 

    #### Objectives:
    - Predict job satisfaction using machine learning models.
    - Explore key drivers of job satisfaction through logistic regression and random forest models.
    - Gain insights into how different job attributes contribute to job satisfaction.

    #### Navigation:
    - **Logit Predictors**: Explore the top predictors of job satisfaction using logistic regression.
    - **Random Forest MDI**: Discover feature importance using the Mean Decrease in Impurity (MDI) technique.
    - **Profile Predictions**: Select a specific employee profile and see personalized predictions, including LIME-based explanations.
    """)
### Logistic Regression Odds Ratios (First Section)
def logit_odds_ratios():
    st.subheader("Top Predictors of High Job Satisfaction (Logit Regression)")

    num_top_features_logit = st.slider("Select number of top features to display (Logit)", 1, 20, 10)

    significant_params = best_model_logit.params[best_model_logit.pvalues < 0.05]
    top_features = significant_params.sort_values(ascending=False).head(num_top_features_logit)
    top_odds_ratios = np.exp(top_features)

    top_features_readable = [feature_name_mapping_logit.get(feat, feat) for feat in top_features.index]

    plot_data = pd.DataFrame({
        'Feature': top_features_readable,
        'Odds Ratio': top_odds_ratios
    }).sort_values(by='Odds Ratio', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Odds Ratio', y='Feature', data=plot_data, palette='Blues_d')
    for index, value in enumerate(plot_data['Odds Ratio']):
        plt.text(value + 0.05, index, f'{value:.2f}', color='blue', va="center", fontsize=12)
    plt.title(f"Top {num_top_features_logit} Positive Predictors of High Job Satisfaction (Odds Ratios)", fontsize=18)
    plt.xlabel("Odds Ratio", fontsize=14)
    plt.ylabel("Top Features", fontsize=14)
    st.pyplot(plt.gcf())

### Random Forest MDI Section
def random_forest_mdi():
    st.subheader("Top Feature Importance (Random Forest MDI)")

    num_top_features_mdi = st.slider("Select number of top features to display (MDI)", 1, 10, 5)

    feature_importances = rf_model.named_steps['model'].feature_importances_
    sorted_indices = np.argsort(feature_importances)[::-1][:num_top_features_mdi]

    readable_feature_names = [feature_name_mapping_top_10.get(X_train_resampled_df.columns[i], X_train_resampled_df.columns[i]) for i in sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.barh(range(num_top_features_mdi), feature_importances[sorted_indices], align='center', color='skyblue')
    plt.yticks(range(num_top_features_mdi), readable_feature_names, fontsize=12)
    plt.gca().invert_yaxis()
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Top {num_top_features_mdi} Feature Importance (MDI)', fontsize=16)
    st.pyplot(plt.gcf())

# Helper function to get profile indices based on relaxed filtering logic
def get_profile_indices(profile, percentiles):
    low_percentile = 0.15  # Further relaxed from 0.25
    high_percentile = 0.85  # Further relaxed from 0.75

    if profile == 'Fast Career Climber':
        indices = X_test_readable[
            (X_test_readable['Your Satisfaction with Career Advancement'] >= percentiles.loc[low_percentile, 'Your Satisfaction with Career Advancement']) &
            (X_test_readable['Your Satisfaction with Salary'] <= percentiles.loc[high_percentile, 'Your Satisfaction with Salary']) &
            (X_test_readable['Your Satisfaction with Intellectual Challenge on the Job'] >= percentiles.loc[low_percentile, 'Your Satisfaction with Intellectual Challenge on the Job'])
        ].index
    elif profile == 'Security Seeker':
        indices = X_test_readable[
            (X_test_readable['Your Satisfaction with Job Security'] >= percentiles.loc[low_percentile, 'Your Satisfaction with Job Security']) &
            (X_test_readable['Your Satisfaction with Salary'] <= percentiles.loc[high_percentile, 'Your Satisfaction with Salary']) &
            (X_test_readable['Your Satisfaction with Job Location'].between(percentiles.loc[low_percentile, 'Your Satisfaction with Job Location'], percentiles.loc[high_percentile, 'Your Satisfaction with Job Location']))
        ].index
    elif profile == 'Balanced Performer':
        indices = X_test_readable[
            (X_test_readable['Your Satisfaction with Salary'].between(percentiles.loc[low_percentile, 'Your Satisfaction with Salary'], percentiles.loc[high_percentile, 'Your Satisfaction with Salary'])) &
            (X_test_readable['Your Satisfaction with Benefits'].between(percentiles.loc[low_percentile, 'Your Satisfaction with Benefits'], percentiles.loc[high_percentile, 'Your Satisfaction with Benefits'])) &
            (X_test_readable['Your Job-Degree Alignment'].between(percentiles.loc[low_percentile, 'Your Job-Degree Alignment'], percentiles.loc[high_percentile, 'Your Job-Degree Alignment']))
        ].index
    elif profile == 'High Performer':
        indices = X_test_readable[
            (X_test_readable['Your Satisfaction with Salary'] >= percentiles.loc[low_percentile, 'Your Satisfaction with Salary']) &
            (X_test_readable['Your Satisfaction with Job Security'] >= percentiles.loc[low_percentile, 'Your Satisfaction with Job Security']) &
            (X_test_readable['Your Job-Degree Alignment'] >= percentiles.loc[low_percentile, 'Your Job-Degree Alignment'])
        ].index
    elif profile == 'Undervalued High Achiever':
        indices = X_test_readable[
            (X_test_readable['Your Satisfaction with Salary'] <= percentiles.loc[high_percentile, 'Your Satisfaction with Salary']) &
            (X_test_readable['Your Satisfaction with Career Advancement'] >= percentiles.loc[low_percentile, 'Your Satisfaction with Career Advancement']) &
            (X_test_readable['Your Satisfaction with Job Security'] >= percentiles.loc[low_percentile, 'Your Satisfaction with Job Security'])
        ].index

    return indices

# Prediction Page (Interactive Employee Profiles)
def prediction_page():
    st.title("Interactive Employee Profile Tool")

    st.markdown("""
    ### How to Use:
    1. Select an **Employee Profile** to explore different satisfaction levels.
    2. Select a specific **Test Instance** to explore how the model predicts their job satisfaction.
    3. Generate explanations using **local interpretable model-agnostic explanations (LIME)**.
    """)

    # Profile selection options
    profile_options = ['Fast Career Climber', 'Security Seeker', 'Balanced Performer', 'High Performer', 'Undervalued High Achiever']
    selected_profile = st.selectbox("Choose a profile:", profile_options)

    # Calculate percentiles for key satisfaction metrics
    low_percentile = 0.15  # Loosened even more
    high_percentile = 0.85

    # Calculate percentiles
    percentiles = X_test_readable.quantile([low_percentile, high_percentile])

    # # Debugging: Print percentiles to check
    # st.write("Calculated Percentiles:", percentiles)

    # Get indices for the selected profile
    indices = get_profile_indices(selected_profile, percentiles)

    # Debugging: Print indices to check if any instances match the selected profile
    st.write(f"Found {len(indices)} instances for profile: {selected_profile}")

    # Handle case when no matching instances are found
    if len(indices) == 0:
        st.error(f"No instances found for profile: {selected_profile}")
        return

    # Show available instances for the selected profile
    selected_instance = st.selectbox(f"Choose a test instance for {selected_profile}:", indices)

    # Prepare test instance from scaled data (ensure it's in correct format)
    test_instance = X_test_scaled_df.loc[selected_instance].values

    # # Debugging: Print test instance values
    # st.write("Test instance selected:", test_instance)

    # Initialize LIME explainer (consistent with your previous logic)
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_test_scaled_df.values,
        feature_names=X_test_readable.columns,
        class_names=['Low Satisfaction', 'High Satisfaction'],
        mode='classification',
        discretize_continuous=False,  # Use continuous values
        sample_around_instance=True  # Sample around instance as in your previous logic
    )

    # Generate LIME explanation for the selected instance
    explanation = explainer.explain_instance(
        data_row=test_instance,
        predict_fn=rf_model.predict_proba,
        num_features=10,
        num_samples=5000  # Adjusted to balance speed and performance
    )

    # Get predicted probabilities for the current test instance
    probs = rf_model.predict_proba([test_instance])[0]

    # Create a mini bar plot to visualize the predicted probabilities
    st.write("Predicted Probability of Job Satisfaction")
    fig, ax = plt.subplots(figsize=(8, 3))  # Increase the figure size
    bars = ax.bar(['Low Satisfaction', 'High Satisfaction'], probs, color=['red', 'blue'])

    # Add labels and titles to make it clear
    ax.set_title("Your Predicted Probability of Job Satisfaction", fontsize=14, fontweight='bold')
    ax.set_ylabel('Probability')
    ax.set_ylim(0, 1)  # Limit y-axis to 0-1 to reflect probabilities

    # Add probability labels inside the bars
    for bar, p in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.05, f'{p:.2f}', ha='center', va='top', color='white', fontsize=12)

    # Show the probability plot
    st.pyplot(fig)

    # Plot the LIME explanation for the selected instance
    explanation_fig = explanation.as_pyplot_figure()
    plt.title("What Factors Could Be Positively or Negatively Impacting Your Job Satisfaction?", fontsize=14, fontweight='bold')

    # Display the LIME explanation plot
    st.pyplot(explanation_fig)


#######

# Navigation Bar
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page", ["Home", "Logit Predictors", "Random Forest MDI", "Profile Predictions"])

    if page == "Home":
        home_page()
    elif page == "Logit Predictors":
        logit_odds_ratios()
    elif page == "Random Forest MDI":
        random_forest_mdi()
    elif page == "Profile Predictions":
        prediction_page()

# Call the main function to run the app
if __name__ == '__main__':
    main()

