# BrainStation Data Science Capstone Template

This is a template repository for setting up your capstone project: it includes a simple folder structure and placeholder files for the most important assets you will be creating.

## Usage

1. Start a new GitHub repo using this template.
2. Update your `LICENSE` file with date and owner.
3. Update your `README.md` file to reflect the project - see a sample structure below and please refer to Synapse on what needs to be included here. 
4. Set up and activate your conda environment:
    - Create a new `conda` environment for your capstone project.
    - Activate the environment and export:
        ```bash
        conda env export > conda.yml
        ```
    - Make sure re-export every time after you update the environment.
    - You can reset your conda environment by running:
        ```bash
        conda env create -f conda.yml
        conda activate <your-env-name>
        ```
5. Add your own notebooks in `./notebooks/` and remove placeholders.
6. Add your own data in `./data/` and remove placeholder. Note: `.gitignore` will ignore the data folder when you push to github, save a copy of your raw and processed data, pickled models in a Google Drive folder and add the link in the `data_links.md` file.
7. Add your project documents, figures, reports, presentation pdf's in the `./docs` and remove placeholders.
8. Add your references (tutorials, papers, books etc.) in `./references`. 
9. Add your own scripts in `./src/` and remove unnecessary folders.

Feel free to rename the folder and customize the project structure to best fit your work - this template is just the starting point.

------------------------------------------------------------------------------

## Project Title
=========================

### Executive Summary

... Define the problem

* Job matching continues to be a critical issue in today’s labour market (and something many of us can relate to, more personally!). 
* We talk a lot about the skills we need to be the right fit for a job, our work-life balance, and levels of financial wellbeing we aspire to. 
* But it continues to be challenge to get the ‘right fit’, as we know from stark unemployment figures
* Often studies and articles focus on salary as the key outcome for a successful job match
* We can go beyond this to consider aspects of job satisfaction, career growth and skills utilisation

... What is the data science opportunity
* How might we use machine learning to better predict the likelihood of a successful job match for jobseekers?
* How can we define a ‘successful job match’?
* How can we use ML classification to predict a ‘successful job match’?
* Using ML, we have the opportunity to use our insights from classification modelling to develop recommendations for job matches and support job seekers

... Key takeaways

* We are looking at the National Survey of College Graduates
* Each dataset is a snapshot of the U.S. college graduate population, at a specific point in time
* Given the richness of the data - this focuses our scope on college graduates


... Description of dataset
The U.S. National Survey of College Graduates (NSCG) is a recurring survey conducted by the National Science Foundation (NSF) that collects detailed information on the educational background, employment status, and career paths of individuals with at least a bachelor's degree in the United States. 

* The NSCG is used to assess trends in the labor market, particularly within science and engineering fields, and to inform policies related to education, workforce development, and economic competitiveness. 
* The survey provides valuable insights into the experiences and outcomes of college graduates, including factors like job satisfaction, salary, and the relevance of education to employment.

The NSCG dataset is collected in a cyclical manner, with major surveys conducted every two to three years. 

This means that each dataset provides a **snapshot of the U.S. college graduate population at a specific point in time**, allowing for both cross-sectional analysis and, to some extent, longitudinal insights when comparing data across different survey years.



... Emerging insights
* Salary distributions amongst graduate survey respondents
* Overall salary distribution is heavily right-skewed (longer tail corresponding to higher values)
* Distribution by gender highlighting pay gaps; lower median salary as well as interquartile range
* Exploration of a previously unexplored variable - usually unavailable in other datasets
* Self-reported perceptions of an individual’s job’s importance to society - over 50% of respondents rate this as ‘Very Important’
* Perceptions of extent to which job is related to their degree - some variability, most most find it’s ‘closely related’
* This is a helpful proxy of ‘skills matching - and we’ll want explore this distribution more by different demographics, and by degree subject

### Data Dictionary
| Raw Data - Variable_Name | Variable Name        | Description                                              | Data Structure                                                                                                           |
|--------------------------|----------------------|----------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| JOBSATIS                | job_satisfaction     | Job satisfaction                                         | 1: Very satisfied, 2: Somewhat satisfied, 3: Somewhat dissatisfied, 4: Very dissatisfied, L: Logical Skip                |
| SATSAL                  | satisfaction_salary  | Satisfaction with job's salary                           | 1: Very satisfied, 2: Somewhat satisfied, 3: Somewhat dissatisfied, 4: Very dissatisfied, L: Logical Skip                |
| SATADV                  | satisfaction_advancement | Satisfaction with advancement opportunities          | 1: Very satisfied, 2: Somewhat satisfied, 3: Somewhat dissatisfied, 4: Very dissatisfied, L: Logical Skip                |
| SATBEN                  | satisfaction_benefits    | Satisfaction with benefits                           | 1: Very satisfied, 2: Somewhat satisfied, 3: Somewhat dissatisfied, 4: Very dissatisfied, L: Logical Skip                |
| SATCHAL                 | satisfaction_challenges  | Satisfaction with job's intellectual challenge                          | 1: Very satisfied, 2: Somewhat satisfied, 3: Somewhat dissatisfied, 4: Very dissatisfied, L: Logical Skip                |
| SATLOC                  | satisfaction_location    | Satisfaction with job location                        | 1: Very satisfied, 2: Somewhat satisfied, 3: Somewhat dissatisfied, 4: Very dissatisfied, L: Logical Skip                |
| SATSEC                  | satisfaction_security    | Satisfaction with job security                        | 1: Very satisfied, 2: Somewhat satisfied, 3: Somewhat dissatisfied, 4: Very dissatisfied, L: Logical Skip                |
| SALARY                  | salary                | Salary (annualized)                                      | 0-9999996: Value, 9999998: Logical Skip                                                                                   |
| FACSOC                  | importance_society_contribution | Self-rated assessment / view of their job's contribution to society | 1: Very important, 2: Somewhat important, 3: Somewhat unimportant, 4: Not important at all, L: Logical Skip             |
| OCEDRLP                 | job_degree_relation   | Extent that principal job is related to highest degree   | 1: Closely related, 2: Somewhat related, 3: Not related, L: Logical Skip                                                  |
| N2MRMED                 | recent_degree_field   | Field of study of major for most recent degree           | Code for field of study during reference week - best code                                                                 |
| MRDG                | recent_degree_type  | Type of degree for most recent degree           | 1 - Bachelor's, 2 - Masters, 3 - Doctorate, 4 -Professional code                                                                 |
| GENDER                    | gender               | Gender                                                   | F: Female, M: Male                                                                                                        |
| AGE                       | age                  | Age                                                      | Age number                                                                                                               |
| AGEGR                     | age_group            | Age group (5 year intervals)                             | 20: Ages 24 or younger, 25: Ages 25-29, 30: Ages 30-34, etc.                                                              |            |
| RACEM                     | race                 | Race                                                     | 1: Asian ONLY, 2: American Indian/Alaska Native ONLY, 3: Black ONLY, 4: White ONLY, 5: Native Hawaiian/Other Pacific Islander ONLY, 6: Multiple Race |
| EMSECDT                   | employer_sector_detailed | Employer sector (detailed codes)                     | 11: 4-yr coll/univ; med schl; univ. res. inst., 12: 2-yr coll/pre-college institutions, 21: Bus/Ind, for-profit, etc.   |
| EMSECSM                   | employer_sector_summary | Employer sector (summary codes)                       | 1: Educational Institution, 2: Government, 3: Business/Industry, L: Logical Skip                                          |
| EMSIZE                    | employer_size        | Employer size                                            | 1: 10 or fewer employees, 2: 11-24 employees, 3: 25-99 employees, 4: 100-499 employees, etc.                             |
| EMST                      | employer_location    | State/country code for employer                          | Various codes for U.S. regions, states, and countries, L: Logical Skip                                                    |
| MARSTA                     | marital_status       | Marital Status                                           | 1: Married, 2: Living in a marriage-like relationship, 3: Widowed, 4: Separated, 5: Divorced, 6: Never married           |
| RESPLO3                   | respondent_location_code | 3-Digit respondent location (state/country code)       | Various codes for U.S. regions, states, and countries, L: Logical Skip                                                    |
| CTZUSIN                   | citizenship_status    | U.S. citizenship status                                  | N: Non-U.S. citizen, Y: U.S. citizen                                                                                      |
| CHLVIN                    | children_in_household | Children living in household                             | N: No, Y: Yes                                                                                                             |

### Demo

... Show your work:
...     Data visualizations
...     Interactive demo (e.g., `streamlit` app)
...     Short video of users trying out the solution


### Methodology

... High-level diagrams of entire process:
...     various data processing steps
...     various modelling directions
...     various prototyping directions


### Organization

#### Repository 

* `data` 
    - contains link to copy of the dataset (stored in a publicly accessible cloud storage)
    - saved copy of aggregated / processed data as long as those are not too large (> 10 MB)

* `model`
    - `joblib` dump of final model(s)

* `notebooks`
    - contains all final notebooks involved in the project

* `docs`
    - contains final report, presentations which summarize the project

* `references`
    - contains papers / tutorials used in the project

* `src`
    - Contains the project source code (refactored from the notebooks)

* `.gitignore`
    - Part of Git, includes files and folders to be ignored by Git version control

* `conda.yml`
    - Conda environment specification

* `README.md`
    - Project landing page (this page)

* `LICENSE`
    - Project license

#### Dataset

https://ncses.nsf.gov/surveys/national-survey-college-graduates/2021
https://ncses.nsf.gov/423/assets/0/file/ncses_nscg.pdf
https://ncses.nsf.gov/pubs/nsf23306/assets/nsf23306.pdf
https://ncses.nsf.gov/explore-data/microdata/national-survey-college-graduates

### Credits & References

... Include any personal learning
