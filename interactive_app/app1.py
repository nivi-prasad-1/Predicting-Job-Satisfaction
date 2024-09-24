# Streamlit imports
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

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

def home_page():
    st.title("Who's likely to love their job? A Predictive Tool for Job Satisfaction and Key Drivers")

    # Display images at the top for a more engaging look
    col1, col2 = st.columns(2)
    
    with col1:
        st.image("jobsat.png", use_column_width=True)
    
    with col2:
        st.image("jobsat2.png", use_column_width=True)

    st.markdown("""
    ### Welcome!

    This tool is designed to help jobseekers, employers, and policymakers understand the factors that drive job satisfaction based on various job-related attributes.

    #### Key Objectives:
    - **Predict Job Satisfaction**: Use machine learning models to predict an individual's likelihood of job satisfaction.
    - **Understand Key Drivers**: Explore the top factors contributing to job satisfaction through logistic regression and random forest models.
    - **Gain Valuable Insights**: Dive into job attributes like salary, career advancement, job security, and more to understand their impact on satisfaction levels.

    #### How to Navigate:
    - **Logit Predictors**: Explore the top predictors of job satisfaction using logistic regression.
    - **Random Forest MDI**: Discover the key factors impacting satisfaction using the Mean Decrease in Impurity (MDI) technique.
    - **Profile Predictions**: Select a specific employee profile and see personalized predictions, with detailed explanations provided by LIME.
    """)

    # Create sections for Jobseekers and Policymakers
    st.markdown("## Value for Different Audiences")

    col1, col2 = st.columns(2)

    with col1:
        st.image("./interactive_app/jobseekers.png", use_column_width=True)
        # Value for Jobseekers
        st.markdown("""
        ### Value for Jobseekers

        Curious about what drives job satisfaction for employees with profiles similar to yours? On the **Predictive Tool - Employee Profiles Explorer** page, you can explore different employee profiles from our test data and see what factors most impact their job satisfaction.

        - Use the insights from employees matching certain profiles to **learn which job attributes drive satisfaction**, such as career advancement, salary, job security, and more.
        - By understanding the profiles that align with your own experiences, you can tweak your career path or negotiate for factors that matter most to you.
        
        Although the predictions aren't fully personalised, you can explore and learn from **real-world data** on what impacts satisfaction for employees like you.
        """)
    
    with col2:
        st.image("./interactive_app/policymakers.png", use_column_width=True)
        # Value for Policymakers
        st.markdown("""
        ### Value for Policymakers

        For policymakers aiming to improve workforce satisfaction and retention, this tool offers valuable insights based on key job attributes.

        - On the **Key Predictors using Logistic Regression** and **Feature Importance using Random Forest** pages, you can explore the **key drivers** of job satisfaction, such as salary, job security, and career advancement. These insights allow for **evidence-based policy formulation** that can address areas needing improvement across sectors.
        - The **Predictive Tool - Employee Profiles Explorer** page helps illustrate how employees with different profiles respond to these key factors, offering an opportunity to understand broader trends in workforce satisfaction.
        - Policymakers can use this tool to **identify areas of focus** for workforce wellbeing initiatives, leading to more **targeted policies** that promote job satisfaction and productivity.

        Whether you are focused on shaping national employment policies or specific sector-based reforms, this tool provides the **data-backed insights** necessary to support impactful, long-lasting improvements.
        """)

    # call to action
    st.markdown("""
    ### Ready to Explore?
    Head to the **Profile Predictions** page to explore different employee profiles from our test data. 
    - Use our Random Forest model to understand the likelihood of job satisfaction based on key job attributes like salary, career advancement, and job security.
    - Although the profiles are based on real data, they provide a general prediction for common employee profiles, rather than personalised predictions for individual users.
    """)



### Logistic Regression Odds Ratios (First Section)
# def logit_odds_ratios():
#     st.subheader("Top Predictors of High Job Satisfaction (Logit Regression)")

#     num_top_features_logit = st.slider("Select number of top features to display (Logit)", 1, 20, 10)

#     significant_params = best_model_logit.params[best_model_logit.pvalues < 0.05]
#     top_features = significant_params.sort_values(ascending=False).head(num_top_features_logit)
#     top_odds_ratios = np.exp(top_features)

#     top_features_readable = [feature_name_mapping_logit.get(feat, feat) for feat in top_features.index]

#     plot_data = pd.DataFrame({
#         'Feature': top_features_readable,
#         'Odds Ratio': top_odds_ratios
#     }).sort_values(by='Odds Ratio', ascending=False)

#     plt.figure(figsize=(12, 8))
#     sns.barplot(x='Odds Ratio', y='Feature', data=plot_data, palette='Blues_d')
#     for index, value in enumerate(plot_data['Odds Ratio']):
#         plt.text(value + 0.05, index, f'{value:.2f}', color='blue', va="center", fontsize=12)
#     plt.title(f"Top {num_top_features_logit} Positive Predictors of High Job Satisfaction (Odds Ratios)", fontsize=18)
#     plt.xlabel("Odds Ratio", fontsize=14)
#     plt.ylabel("Top Features", fontsize=14)
#     st.pyplot(plt.gcf())

import plotly.express as px

import plotly.express as px

def logit_odds_ratios():
    st.subheader("Top Predictors of High Job Satisfaction (Logit Regression)")

    # Slider to select the number of top features to display
    num_top_features_logit = st.slider("Select number of top features to display (Logit)", 1, 20, 10)

    # Get significant features based on p-values and exclude the constant term
    significant_params = best_model_logit.params[best_model_logit.pvalues < 0.05]
    significant_params = significant_params.drop('const', errors='ignore')  # Drop the constant if present
    top_features = significant_params.sort_values(ascending=False).head(num_top_features_logit)
    top_odds_ratios = np.exp(top_features)  # Convert log-odds to odds ratios

    # Map feature names to more readable labels
    top_features_readable = [feature_name_mapping_logit.get(feat, feat) for feat in top_features.index]

    # Create a DataFrame for Plotly and reverse the order for correct bar display
    plot_data_logit = pd.DataFrame({
        'Feature': top_features_readable,
        'Odds Ratio': top_odds_ratios
    }).sort_values(by='Odds Ratio', ascending=True)  # Reversed order for Plotly

    # Create the bar chart using Plotly
    fig_logit = px.bar(
        plot_data_logit,
        x='Odds Ratio',
        y='Feature',
        orientation='h',  # Horizontal bar chart
        title=f'Top {num_top_features_logit} Positive Predictors of High Job Satisfaction (Odds Ratios)',
        color_discrete_sequence=['blue']  # Set color for bars
    )

    # Customize layout
    fig_logit.update_layout(
        title_font_size=24, 
        # title_font_family='Roboto', 
        title_font_color='white',
        xaxis_title_font_size=18,
        yaxis_title_font_size=18,
        xaxis=dict(
            tickfont=dict(size=16),  # Font size for x-axis ticks
            showgrid=False
        ),
        yaxis=dict(
            tickfont=dict(size=16),  # Font size for y-axis ticks
            showgrid=False
        ),
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        paper_bgcolor='rgba(0,0,0,0)',
        bargap=0.15,  # Space between bars
        legend=dict(
            font=dict(size=16)
        )
    )

    # Update hover text to display two decimal places
    fig_logit.update_traces(
        texttemplate=None,  # No data labels inside the bars
        hovertemplate='<b>%{y}</b><br>Odds Ratio: %{x:.2f}<extra></extra>'  # Format hover text
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig_logit)

    # Methodological note at the bottom of the odds ratio section
    st.markdown("""
    **Methodological Note:**
    - The results displayed above are based on a **logistic regression** model fitted using **Statsmodels**.
    - Odds ratios represent the likelihood of high job satisfaction associated with each feature, holding other factors constant. 
    - For example, an odds ratio greater than 1 indicates that a one standard deviation increase in that feature increases the likelihood of high job satisfaction. Conversely, an odds ratio less than 1 suggests a decrease in that likelihood.
    - Only features with statistically significant effects (p-values < 0.05) are included in the plot. These features are the strongest predictors of job satisfaction in our model.
    """)

### Random Forest MDI Section
# def random_forest_mdi():
#     st.subheader("Top Feature Importance (Random Forest MDI)")

#     num_top_features_mdi = st.slider("Select number of top features to display (MDI)", 1, 10, 5)

#     feature_importances = rf_model.named_steps['model'].feature_importances_
#     sorted_indices = np.argsort(feature_importances)[::-1][:num_top_features_mdi]

#     readable_feature_names = [feature_name_mapping_top_10.get(X_train_resampled_df.columns[i], X_train_resampled_df.columns[i]) for i in sorted_indices]

#     plt.figure(figsize=(10, 6))
#     plt.barh(range(num_top_features_mdi), feature_importances[sorted_indices], align='center', color='skyblue')
#     plt.yticks(range(num_top_features_mdi), readable_feature_names, fontsize=12)
#     plt.gca().invert_yaxis()
#     plt.xlabel('Importance', fontsize=12)
#     plt.title(f'Top {num_top_features_mdi} Feature Importance (MDI)', fontsize=16)
#     st.pyplot(plt.gcf())

import plotly.express as px

def random_forest_mdi():
    st.subheader("Top Feature Importance (Random Forest MDI)")

    # Slider to allow selection of the number of top features to display
    num_top_features_mdi = st.slider("Select number of top features to display (MDI)", 1, 10, 5)

    feature_importances = rf_model.named_steps['model'].feature_importances_
    sorted_indices = np.argsort(feature_importances)[::-1][:num_top_features_mdi]

    readable_feature_names = [feature_name_mapping_top_10.get(X_train_resampled_df.columns[i], X_train_resampled_df.columns[i]) for i in sorted_indices]
    
    # Prepare the data for Plotly
    plot_data_mdi = pd.DataFrame({
        'Feature': readable_feature_names,
        'Importance': feature_importances[sorted_indices]
    }).sort_values(by='Importance', ascending=True)

    # Create the bar chart using Plotly
    fig_mdi = px.bar(
        plot_data_mdi,
        x='Importance',
        y='Feature',
        orientation='h',  # Horizontal bar chart
        text='Importance',
        title=f'Top {num_top_features_mdi} Feature Importance (MDI)',
        color_discrete_sequence=['green']  # Set a color for the bars
    )

    # Create the bar chart using Plotly
    fig_mdi = px.bar(
        plot_data_mdi,
        x='Importance',
        y='Feature',
        orientation='h',  # Horizontal bar chart
        title=f'Top {num_top_features_mdi} Feature Importance (MDI)',
        color_discrete_sequence=['green']  # Set a color for the bars
    )

    # Customize layout
    fig_mdi.update_layout(
        title_font_size=24, 
        title_font_color='white',
        xaxis_title_font_size=18,
        yaxis_title_font_size=16,
        xaxis=dict(
            tickfont=dict(size=16),  # Font size for x-axis ticks
            showgrid=False
        ),
        yaxis=dict(
            tickfont=dict(size=16),  # Font size for y-axis ticks
            showgrid=False
        ),
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        paper_bgcolor='rgba(0,0,0,0)',
        bargap=0.15,  # Space between bars
        legend=dict(
            font=dict(size=16)
        )
    )

    # Add hover template for displaying importance with 2 decimal places and remove data labels
    fig_mdi.update_traces(
        texttemplate=None,  # Ensure no data labels are displayed
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.2f}<extra></extra>'  # Format hover text with 2 decimal places
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig_mdi)

    # Methodological note at the bottom of the feature importance section
    st.markdown("""
    **Methodological Note:**
    - This Random Forest model is trained with **400 decision trees**, and hyperparameter optimisation was performed to improve the model’s accuracy and generalisability.
    - The bars above represent the **Mean Decrease in Impurity (MDI)**, which is a measure of how much each feature contributes to reducing uncertainty in the predictions. 
    - A higher MDI value means the feature plays a more important role in the model's decision-making process.
    - Random Forest is an ensemble learning method that builds multiple decision trees and averages their predictions to improve accuracy and reduce overfitting.
    """)

# Helper function to get profile indices based on relaxed filtering logic
def get_profile_indices(profile, percentiles):
    low_percentile = 0.15  # Further relaxed from 0.25
    high_percentile = 0.85  # Further relaxed from 0.75

    if profile == 'Career Climber':
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

    # Create a dictionary to hold the explanation of each profile
    profile_explanations = {
        'Career Climber': {
            'Description': 'Focused on rapid career growth. Highly satisfied with career advancement and intellectual challenge but may be less satisfied with salary.',
            'High Satisfaction': 'Career Advancement, Intellectual Challenge',
            'Low Satisfaction': 'Salary'
        },
        'Security Seeker': {
            'Description': 'Values job security and stability. Highly satisfied with job security and location, but less satisfied with salary.',
            'High Satisfaction': 'Job Security, Location',
            'Low Satisfaction': 'Salary'
        },
        'Balanced Performer': {
            'Description': 'Balanced satisfaction across key job areas. No significant high or low satisfaction in any area.',
            'High Satisfaction': 'Salary, Benefits, Job-Degree Alignment',
            'Low Satisfaction': 'None (Balanced)'
        },
        'High Performer': {
            'Description': 'High achiever across all metrics. Satisfied with salary, job security, and job-degree alignment.',
            'High Satisfaction': 'Salary, Job Security, Job-Degree Alignment',
            'Low Satisfaction': 'None'
        },
        'Undervalued High Achiever': {
            'Description': 'Feels undervalued in terms of salary but is otherwise satisfied with career advancement and job security.',
            'High Satisfaction': 'Career Advancement, Job Security',
            'Low Satisfaction': 'Salary'
        }
    }

    # Convert the dictionary into a DataFrame for display
    profile_df = pd.DataFrame.from_dict(profile_explanations, orient='index')

    st.markdown("""
    #### Employee Profiles Explained:
    Below is a breakdown of each profile type. These profiles reflect different types of employees based on their satisfaction levels in various areas of their job.
    """)

    # Display the DataFrame as a table
    st.table(profile_df)

    st.markdown("""
    Now, select a profile to see how the model predicts job satisfaction for a random employee in this category.
    """)

    # Profile selection options
    profile_options = ['Career Climber', 'Security Seeker', 'Balanced Performer', 'High Performer', 'Undervalued High Achiever']

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

    # # Debugging: Print indices to check if any instances match the selected profile
    # st.write(f"Found {len(indices)} instances for profile: {selected_profile}")

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
    
    st.markdown("""
    #### What is a "Test Instance"?
    A test instance represents a randomly selected employee from our test data that matches the chosen profile. Based on their attributes (like salary, satisfaction with job security), the model will predict their job satisfaction.
    """)

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

    st.markdown("""
    #### Predicted Probability of Job Satisfaction:
    The bar chart below shows the predicted probability that the randomly selected employee falls into either the "High Satisfaction" or "Low Satisfaction" category based on the model's predictions.
    """)

    # # Create a mini bar plot to visualize the predicted probabilities
    # st.write("Predicted Probability of Job Satisfaction")
    # fig, ax = plt.subplots(figsize=(8, 3))  # Increase the figure size
    # bars = ax.bar(['Low Satisfaction', 'High Satisfaction'], probs)

    # # Add labels and titles to make it clear
    # ax.set_title("Your Predicted Probability of Job Satisfaction", fontsize=14, fontweight='bold')
    # ax.set_ylabel('Probability')
    # ax.set_ylim(0, 1)  # Limit y-axis to 0-1 to reflect probabilities

    # # Add probability labels inside the bars
    # for bar, p in zip(bars, probs):
    #     ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.05, f'{p:.2f}', ha='center', va='top', fontsize=12)

    # # Show the probability plot
    # st.pyplot(fig)


    #     # Create a predicted probability bar plot using Plotly
    # st.write("Predicted Probability of Job Satisfaction")

    # Prepare data for Plotly
    prob_df = pd.DataFrame({
        'Satisfaction': ['Low Satisfaction', 'High Satisfaction'],
        'Probability': probs
    })

    # Create a bar chart using Plotly Express
    fig_prob = px.bar(prob_df, x='Satisfaction', y='Probability', text='Probability',
                    title='Your Predicted Probability of Job Satisfaction',
                    labels={'Satisfaction': 'Job Satisfaction', 'Probability': 'Predicted Probability'},
                    color='Satisfaction',  # Optional: color based on satisfaction level
                    color_discrete_sequence=['blue', 'green'])  # Set colors for bars

    # Customize appearance
    fig_prob.update_traces(
        texttemplate='%{text:.2f}',  # For the text on the bars
        textposition='auto',
        marker=dict(line=dict(width=2, color='white')),
        textfont_size=20,  # Adjust this value to change the font size of the data labels
        hovertemplate='%{y:.2f}'  # Format hover text to show probabilities to 2 decimal places
    )

    fig_prob.update_layout(
        title_font_size=24, 
        # title_font_family='Roboto', 
        title_font_color='white',
        xaxis_title_font_size=18, 
        yaxis_title_font_size=14, 
        yaxis_range=[0, 1],
        xaxis=dict(
            tickfont=dict(size=18)  # Adjust x-axis tick label font size
        ),
        yaxis=dict(
            tickfont=dict(size=18)  # Adjust y-axis tick label font size
        )
    )

    # Display Plotly figure
    st.plotly_chart(fig_prob)


    # # Plot the LIME explanation for the selected instance
    # explanation_fig = explanation.as_pyplot_figure()
    # plt.title("What Factors Could Be Positively or Negatively Impacting Your Job Satisfaction?", fontsize=14, fontweight='bold')

    # # Display the LIME explanation plot
    # st.pyplot(explanation_fig)

    import plotly.graph_objects as go

    st.markdown("""
    #### LIME Explanation:
    LIME helps explain **why** the model made the prediction above by analysing the factors that contributed the most to the employee's job satisfaction prediction.
    
    Here are the factors and their respective contributions (positive or negative) to the predicted satisfaction level.
    """)

    # Extract LIME explanation as a dictionary or list of tuples
    lime_explanation = explanation.as_list()  # Extracts [(feature, importance), ...]

    # Separate features and importance for plotting
    features = [x[0] for x in lime_explanation]
    importance = [x[1] for x in lime_explanation]

    # Assign colors based on positive or negative importance
    colors = ['green' if val > 0 else 'blue' for val in importance]

    # Create a bar chart using plotly.graph_objects
    fig_lime = go.Figure(go.Bar(
        x=importance,
        y=features,
        orientation='h',  # Horizontal bar chart
        marker=dict(color=colors, line=dict(width=1.5, color='black')),  # Add color and border
        # text=[f'{val:.2f}' for val in importance],  # Display importance as text
        # textposition='auto'  # Position text on the bars
    ))

    # Update layout to match the style
    fig_lime.update_layout(
        title="Factors Positively or Negatively Impacting Job Satisfaction",
        title_font_size=24, 
        # title_font_family='Roboto', 
        title_font_color='white',
        xaxis_title="Importance",
        xaxis_title_font_size=18,
        yaxis_title_font_size=16,
        xaxis=dict(
            tickfont=dict(size=18),  # Font size for x-axis ticks
            showgrid=True  # Optional: remove gridlines
        ),
        yaxis=dict(
            tickfont=dict(size=14),  # Font size for y-axis ticks
            showgrid=False  # Optional: remove gridlines
        ),
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent overall background
        bargap=0.15  # Space between bars
    )

    # Add hover template for limiting decimals in hover text
    fig_lime.update_traces(hovertemplate='%{x:.5f}')

    # Display the plot in Streamlit
    st.plotly_chart(fig_lime)

    st.markdown("""
    In the LIME chart above, the green bars represent the kind of factors that could **positively** impact job satisfaction, while the blue bars represent factors that **negatively** impact it. 
    These factors help you understand the model’s reasoning behind the prediction.
    """)

        # Add Methodological Note:
    st.markdown("""
    ### Methodological Note:

    The employee profiles are generated using **percentile thresholds** on key job satisfaction metrics (e.g., career advancement, salary, job security). We group employees into profiles based on their satisfaction levels in various areas. For instance, a 'Career Climber' might be highly satisfied with career advancement but less satisfied with salary.

    These predictions are based on the **test data** – a portion of the overall dataset that the model hasn't seen during training, allowing us to evaluate how well it generalises to new employees.

    **How the LIME Explanation Works**:
    
    1. **Test Instance**: Using the random test instance generator, random employee is selected from the test data that matches the chosen profile 
    2. **This test instance passes through the model and outputs predicted probabilities of 'Low Satisfaction' and 'High Satisfaction'
    2. **Synthetic Data Generation**: LIME generates synthetic data points around this employee by slightly tweaking the features (like salary or job security).
    3. **Model Predictions**: The model predicts job satisfaction for these synthetic employees.
    4. **Weighting**: LIME gives higher importance to data points that are closer to the original employee/row of data
    5. **Local Model**: Using the weighted data, LIME trains a simple, interpretable model to approximate how the complex model made its prediction for the specific employee.
    6. **Explanation**: The resulting explanation shows which features (like salary or career advancement) had the most positive or negative impact on the employee’s job satisfaction.

    This process helps break down the complex decision-making of the model into easy-to-understand pieces, giving you a clear sense of why the model made its prediction.
    """)

#######

# Navigation Bar
def main():
    st.sidebar.title("Explore insights")
    page = st.sidebar.radio("Choose a page", ["Welcome", "Key Predictors using Logistic Regression", "Feature Importance usng Random Forest", "Predictive Tool - Employee Profiles Explorer"])

    if page == "Welcome":
        home_page()
    elif page == "Key Predictors using Logistic Regression":
        logit_odds_ratios()
    elif page == "Feature Importance usng Random Forest":
        random_forest_mdi()
    elif page == "Predictive Tool - Employee Profiles Explorer":
        prediction_page()

# Call the main function to run the app
if __name__ == '__main__':
    main()

