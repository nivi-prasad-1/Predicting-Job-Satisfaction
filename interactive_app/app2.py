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
st.set_page_config(page_title="Predictive Tool for Job Satisfaction", layout="wide")

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

#################################################HOME PAGE######
def home_page():
    st.title("Who's likely to love their job? A Predictive Tool for Job Satisfaction, and Key Drivers")

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

def explore_model_results():
    st.title("Explore Model Results")

    st.markdown("""
    ### Logit Predictors and Random Forest Feature Importance
    This section allows you to explore key drivers of job satisfaction using logistic regression and random forest models.
    """)

    # Logit Predictors Section
    st.subheader("Top Predictors of High Job Satisfaction (Logit Regression)")
    num_top_features_logit = st.slider("Select number of top features to display (Logit)", 1, 20, 10)

    significant_params = best_model_logit.params[best_model_logit.pvalues < 0.05]
    top_features = significant_params.sort_values(ascending=False).head(num_top_features_logit)
    top_odds_ratios = np.exp(top_features)

    top_features_readable = [feature_name_mapping_logit.get(feat, feat) for feat in top_features.index]

    plot_data_logit = pd.DataFrame({
        'Feature': top_features_readable,
        'Odds Ratio': top_odds_ratios
    }).sort_values(by='Odds Ratio', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Odds Ratio', y='Feature', data=plot_data_logit)  # No color argument for default colors
    plt.title(f"Top {num_top_features_logit} Positive Predictors of High Job Satisfaction (Odds Ratios)", fontsize=18)
    plt.xlabel("Odds Ratio", fontsize=14)
    plt.ylabel("Top Features", fontsize=14)
    st.pyplot(plt.gcf())

    # Random Forest MDI Section
    st.subheader("Top Feature Importance (Random Forest MDI)")
    num_top_features_mdi = st.slider("Select number of top features to display (MDI)", 1, 10, 5)

    feature_importances = rf_model.named_steps['model'].feature_importances_
    sorted_indices = np.argsort(feature_importances)[::-1][:num_top_features_mdi]

    readable_feature_names = [feature_name_mapping_top_10.get(X_train_resampled_df.columns[i], X_train_resampled_df.columns[i]) for i in sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.barh(range(num_top_features_mdi), feature_importances[sorted_indices], align='center')  # No color specified
    plt.yticks(range(num_top_features_mdi), readable_feature_names, fontsize=12)
    plt.gca().invert_yaxis()
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Top {num_top_features_mdi} Feature Importance (MDI)', fontsize=16)
    st.pyplot(plt.gcf())


###### PREDICTION TOOL #####

# Helper function to get profile indices based on relaxed percentile filtering logic
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


def prediction_page():
    st.title("Interactive Employee Profile Tool")

    st.markdown("""
    ### Welcome to the Profile Prediction Tool

    This tool categorizes employee profiles based on various job satisfaction metrics. The profiles are created using percentile thresholds across several key factors, such as satisfaction with career advancement, salary, job security, etc.

    Below is a summary of the profiles and how they are created:
    """)

    # Creating a table to explain the profiles
    profile_table_data = {
        'Profile Name': ['Career Climber', 'Security Seeker', 'Balanced Performer', 'High Achiever', 'Undervalued Achiever'],
        'Description': [
            'This profile has high satisfaction with career advancement and intellectual challenge but is relatively less satisfied with salary.',
            'Prioritises job security and location, with moderate or lower salary satisfaction.',
            'This group has balanced satisfaction across salary, benefits, and job-degree alignment.',
            'Highly satisfied with salary, job security, and job-degree alignment.',
            'Strong satisfaction with career advancement and job security, but less satisfied with salary.'
        ],
        'High Satisfaction': [
            'Career Advancement, Intellectual Challenge',
            'Job Security, Location',
            'Salary, Benefits, Job-Degree Alignment',
            'Salary, Job Security, Job-Degree Alignment',
            'Career Advancement, Job Security'
        ],
        'Low Satisfaction': [
            'Salary',
            'Salary',
            'None (Balanced)',
            'None',
            'Salary'
        ]
    }

    # Display the table in the app
    profile_df = pd.DataFrame(profile_table_data)
    st.table(profile_df)

    st.markdown("""
    ### How to Use:
    1. Select an **Employee Profile** to explore different satisfaction levels.
    2. Click on the **Random Instance Selector** to explore how the model predicts job satisfaction.
    3. Adjust job attributes using the **What-If Scenario Tool** to see how predictions change.
    """)

    # Profile selection options
    profile_options = ['Career Climber', 'Security Seeker', 'Balanced Performer', 'High Performer', 'Undervalued High Achiever']
    selected_profile = st.selectbox("Choose a profile:", profile_options)

    # Calculate percentiles for key satisfaction metrics
    low_percentile = 0.15  # Loosened even more
    high_percentile = 0.85

    # Calculate percentiles
    percentiles = X_test_readable.quantile([low_percentile, high_percentile])

    # Get indices for the selected profile
    indices = get_profile_indices(selected_profile, percentiles)

    st.write(f"Found {len(indices)} instances for profile: {selected_profile}")

    # Handle case when no matching instances are found
    if len(indices) == 0:
        st.error(f"No instances found for profile: {selected_profile}")
        return

    # Button to randomly select a test instance
    if st.button("Select a Random Instance"):
        selected_instance = np.random.choice(indices)
        st.write(f"Randomly selected test instance: {selected_instance}")
    else:
        st.warning("Please click the button to select a test instance.")
        return

    # Prepare test instance from scaled data (ensure it's in correct format)
    test_instance = X_test_scaled_df.loc[selected_instance].values

    # Initialize LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_test_scaled_df.values,
        feature_names=X_test_readable.columns,
        class_names=['Low Satisfaction', 'High Satisfaction'],
        mode='classification',
        discretize_continuous=False,
        sample_around_instance=True
    )

    # Generate LIME explanation for the selected instance
    explanation = explainer.explain_instance(
        data_row=test_instance,
        predict_fn=rf_model.predict_proba,
        num_features=10,
        num_samples=5000
    )

    # Get predicted probabilities for the current test instance
    probs = rf_model.predict_proba([test_instance])[0]

    # Create a mini bar plot to visualize the predicted probabilities
    st.write("Predicted Probability of Job Satisfaction")
    fig, ax = plt.subplots(figsize=(8, 3))
    bars = ax.bar(['Low Satisfaction', 'High Satisfaction'], probs)

    # Add labels and titles to make it clear
    ax.set_title("Your Predicted Probability of Job Satisfaction", fontsize=14, fontweight='bold')
    ax.set_ylabel('Probability')
    ax.set_ylim(0, 1)  # Limit y-axis to 0-1 to reflect probabilities

    # Add probability labels inside the bars
    for bar, p in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.05, f'{p:.2f}', ha='center', va='top', color='white', fontsize=12)

    st.pyplot(fig)

    # Plot the LIME explanation for the selected instance
    explanation_fig = explanation.as_pyplot_figure()
    plt.title("What Factors Could Be Positively or Negatively Impacting Your Job Satisfaction?", fontsize=14, fontweight='bold')

    # Display the LIME explanation plot
    st.pyplot(explanation_fig)

    # What-If Scenario Modeling
    st.write("### What-If Scenario: Adjust Job Attributes")

    # Example sliders for adjusting job-related attributes
    adjusted_salary = st.slider("Salary", min_value=20000, max_value=150000, step=5000, value=int(X_test_readable.loc[selected_instance, 'Salary']))
    adjusted_advancement = st.slider("Your Satisfaction with Career Advancement", min_value=1, max_value=10, step=1, value=int(X_test_readable.loc[selected_instance, 'Your Satisfaction with Career Advancement']))
    adjusted_security = st.slider("Your Satisfaction with Job Security", min_value=1, max_value=10, step=1, value=int(X_test_readable.loc[selected_instance, 'Your Satisfaction with Job Security']))

    # Update test instance with adjusted values
    adjusted_instance = test_instance.copy()

    # Print the column names and their index positions
    st.write("Feature names and their indices in the test_instance array:")
    for idx, col in enumerate(X_test_readable.columns):
        st.write(f"{idx}: {col}")

    # Use .get_loc() to find specific feature positions
    salary_idx = X_test_readable.columns.get_loc('Salary')
    advancement_idx = X_test_readable.columns.get_loc('Your Satisfaction with Career Advancement')
    security_idx = X_test_readable.columns.get_loc('Your Satisfaction with Job Security')

    st.write(f"Salary index: {salary_idx}")
    st.write(f"Satisfaction with Career Advancement index: {advancement_idx}")
    st.write(f"Job Security index: {security_idx}")

    # Now use the correct indices in the What-If scenario
    adjusted_instance[salary_idx] = adjusted_salary
    adjusted_instance[advancement_idx] = adjusted_advancement
    adjusted_instance[security_idx] = adjusted_security


    # Get predictions for the adjusted instance
    adjusted_probs = rf_model.predict_proba([adjusted_instance])[0]

    st.write("Updated Probability of Job Satisfaction (after adjustments)")
    fig, ax = plt.subplots(figsize=(8, 3))
    bars = ax.bar(['Low Satisfaction', 'High Satisfaction'], adjusted_probs)

    # Add labels and titles to make it clear
    ax.set_title("Updated Predicted Probability of Job Satisfaction", fontsize=14, fontweight='bold')
    ax.set_ylabel('Probability')
    ax.set_ylim(0, 1)  # Limit y-axis to 0-1 to reflect probabilities

    # Add probability labels inside the bars
    for bar, p in zip(bars, adjusted_probs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.05, f'{p:.2f}', ha='center', va='top', color='white', fontsize=12)

    st.pyplot(fig)

    st.write("The What-If Scenario allows you to tweak job-related features (salary, advancement, security) and instantly see how they impact predicted job satisfaction.")

    # Navigation Bar
    def main():
        st.sidebar.title("Navigation")
        # Sidebar radio buttons to navigate between pages
        page = st.sidebar.radio(
            "Choose a page",
            ["Home", "Explore Model Results", "Profile Predictions"]
        )

        # Conditional rendering of each page based on user's selection
        if page == "Home":
            home_page()
        elif page == "Explore Model Results":
            explore_model_results()  # Combined page for both logit and random forest
        elif page == "Profile Predictions":
            prediction_page()

    # Call the main function to run the app
    if __name__ == '__main__':
        main()
