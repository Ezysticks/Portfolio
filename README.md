# Data Science Portfolio - Israel Ezema

## About Me

I am a dedicated professional with expertise in Data Science and Supply Chain Material Management. For a comprehensive overview about me, please refer to [My Resume](https://drive.google.com/file/d/11w3H9FAKSr-blFOwwhg5yoUoAVaYDJoi/view?usp=sharing), [My Website](https://sites.google.com/view/e-israel/home), [My Linkedin](https://www.linkedin.com/in/israel-ezema-009530195/),  [Zindi Africa](https://zindi.africa/users/Ezysticks), and [My Kaggle](https://www.kaggle.com/israelezema).

## Projects

Here are a few projects i have been involved in that's is available to the public domain. Kindly explore, and Click on each project to access the full analysis and code. 
---

### 1. [🚀 Full-Stack NLP Preparing Text Data from Emails, PDFs, and Web for AI Training (PUBLIC)](https://github.com/Ezysticks/-Full-Stack-NLP-Preparing-Text-Data-from-Emails-PDFs-and-Web-for-AI-Training)

![Full-Stack NLP Preparing Text Data from Emails, PDFs, and Web for AI Training](Pictures/Fullstack_NLP.png)

This project implements a complete text data pipeline for LLM fine-tuning. It automates the extraction, cleaning, and formatting of text data from multiple sources—emails (IMAP), PDFs (pdfplumber), and web scraping (BeautifulSoup). The processed data is converted into JSONL format for seamless integration with Hugging Face Transformers. Additionally, it includes data manipulation with Pandas and fine-tuning an LLM model for custom AI applications.  
 
 🔹 Key Features: 
    ✅ Extract & clean text from Emails, PDFs, and Web 
    ✅ Convert text to JSONL format for fine-tuning 
    ✅ Load and analyze processed data with Pandas 
    ✅ Fine-tune and test a custom LLM model  
    
📌 **Tech Stack:** Python, NLP, Transformers, IMAP, pdfplumber, BeautifulSoup, Pandas  


---

### 2. [EDSA/AiGlass: Carbon Emission Analysis & Prediction (PUBLIC)](https://github.com/EDSA-Internship-Group-3/co2-emission-analysis)

![Comparison: Top 20 Countries with CO2_Emission_per_Capita](Pictures/CO2_emmision_comparison.png)

As the world becomes even more modernized by the year, it's becoming all the more POLLUTED. UN Official Data States:
1. Over 3 BILLION PEOPLE of the world’s 8 Billion people are affected by degrading ecosystems 
2. Pollution is responsible for some 9 MILLION premature deaths each year
3. Over 1 million plant and animal species risk extinction

With CO2 being a significant air pollutant, It's important to studying how to reduce CO2 emission without curbing economic growth. Hence the need to continue in this research which was built upon an initial analysis on [Kaggle](https://www.kaggle.com/code/lobosi/part-7-co2-emission-analysis). The data source originated from the US Energy Administration and joined together for an easier analysis. Its a collection of some big factors that play into C02 Emissions, with everything from the Production and Consumption of each type of major energy source for each country and its pollution rating each year. It also includes each countries GDP, Population, Energy intensity per capita (person), and Energy intensity per GDP (per person GDP). All the data spans all the way from the 1980's to 2020. 

We further Normalized and enriched the dataset with extra features from the [World Bank Data Website](https://data.worldbank.org/);
1. Rate of population change - To see if a possible change in population of a place will result in change in CO2 emission & to What extent
2. Population density - Does the density of a population have any effect on CO2 Emission?
3. GDP splits - Example, % for agriculture vs manufacturing; Hypothetically, GDP increase due to agricultural/Green activities should oppose the direct correlation of rise in GDP to CO2 Emission
4. Rate of Deforestation - As a result of our research on why the Dip in CO2 Emission of the World occurred in 2009 and the sudden rise in 2010 when Energy Type, Pop, and GDP were Constant.

`Our Best Performing Model was XGBoost with an RMSE of 15.227 MMtonnes CO2 with Energy Consumption, Energy Type & Emission per capita features as the major determinant features.`

**Key Insights from this Project:**
* The correlation between population size and CO2 emissions is not straightforward. Factors such as local activities, cultural practices, and policies play a more significant role in determining CO2 emissions per capita.
* Energy consumption is a critical factor in CO2 emissions. The type of energy sources prevalent in a region significantly influences its carbon footprint.
* While a higher GDP often correlates with increased CO2 emissions, it's essential to recognize that GDP is a multifaceted metric. The primary drivers of CO2 emissions are often related to manufacturing and agricultural activities within a region.
* Globally, coal and petroleum, along with other liquid fuels, have been the dominant sources contributing to CO2 emissions. This underscores the importance of transitioning towards cleaner and more sustainable energy alternatives.
* Coal and Petroleum/other liquids have been the dominant energy source contributing to CO2 Emitted globally.

**Keywords:** CO2_Emission_per_Capita, Energy Consumption, Population, GDP, XGBoost, RMSE.

---


### 3. [EDSA/AI Incorporated: Streamlit-based: Movie Recommender System (PUBLIC)](https://github.com/2110ACDS-T4/unsupervised-predict-streamlit-template/blob/master/Team%204/Notebooks/3.0_EDSA_movie_recommendation_2022_Notebook.ipynb)

![Image header](https://github.com/2110ACDS-T4/unsupervised-predict-streamlit-template/blob/master/resources/imgs/Image_header.png)

This project was part of the requirement to complete the Unsupervised Predict within EDSA's Data Science course. It required us to build and deploy a basic recommender engine based upon the Streamlit web application framework. In the end, I completed the following;

* Constructed several recommendation algorithm based on Content and Collaborative filtering
* Applied Model Versioning with COMET
* Carried out Several Data Processing, Exploratory Data Analysis, and Feature Engineering techniques
* Applied the 'Did you mean?' Algorithm (Levenshtein Distance)
* Deploying our best Model to a Streamlit Web App.

**Keywords:** Recommendation algorithm, Streamlit web deployment, COMET versioning, Exploratory Data Analysis (EDA)

---


### 4. [EDSA/DataWare Solutions - Climate Change Belief Analysis 2022 (PUBLIC)](https://github.com/2110ACDS-T12/classification-predict-streamlit-template/blob/master/Project%20File/5.0%20Advance_Classification_Notebook.ipynb)

![pic](Pictures/climate-change-definition-meaning.jpg)

This project was part of the requirement to complete the Classification Predict within EDSA's Data Science course. Companies are constantly in the push for more sustainable business practices, products and services with many increasingly labeling themselves “eco-friendly”, but along with that, like every other business, feasibility is very important, as well as maximising productivity and profitability. Hence the development through the knowledge of the demand & Supply on her goods & services, being able to classify Tweets, messages and comments of her market to sentiment classes. During the course of the project carried out;

* Data preprocessing
* Natural Language Processing (NLP) 
* Performed data wrangling and exploratory data analysis with Matplotlib and Seaborn
* Applied Data processing techniques such as Text Cleaning, Tokenization, Stemming & Lemming 
* Applied both Oversampling and Undersapmling Class Balancing Techniques
* Applied several classifier models and Evaluated for the Best Prforming Fi-Score
* Hypertuned Best Model
* Deployed on Streamlit Web Based App

**Keywords:** Natural Language Processing (NLP), Class Balancing, Hypertunning, Tokenization

---

### 5. [EDSA Classification Hackathon: South African language Identification (PUBLIC)](https://github.com/Ezysticks/south_african_language_identification)

![pic](https://camo.githubusercontent.com/43ac0c5125127da70afff57a96fbf4044e446209d6961a2f7fb4258951ad53f4/68747470733a2f2f7777772e676f6f676c65617069732e636f6d2f646f776e6c6f61642f73746f726167652f76312f622f6b6167676c652d757365722d636f6e74656e742f6f2f696e626f78253246323230353232322532463766333435343463316231663631643161353934396264646163666438346139253246536f7574685f4166726963615f6c616e6775616765735f323031312e6a70673f67656e65726174696f6e3d3136303433393336363933333930333426616c743d6d65646961)

South Africa is a multicultural society that is characterised by its rich linguistic diversity. Language is an indispensable tool that can be used to deepen democracy and also contribute to the social, cultural, intellectual, economic and political life of the South African society. With such a multilingual population, it is only obvious that the South African systems and devices should also be able to communicate in multi-languages.

Hence, I created a Machine Learning Model which takes text which is in any of South Africa's 11 Official languages and identify which language the text is in. The goal of the model is to determining the natural language that a piece of text is written in.

In this project, I:
* Applied NLP techniques
* Multinomial Naive Bayes classifier as best performing 
* Evaluated Model Performance Using Confusion Matrix
* Extracted and Submitted Test Prediction file to Kaggle

**Keywords:** NLP, EDA, Naive Bayes classifier, Confusion Matrix

---

### 6. [EDSA: Spain Electricity Shortfall Challenge (PUBLIC)](https://github.com/Ezysticks/load-shortfall-regression-predict-api/blob/master/Advanced-Regression-Starter-Data/Versions_Team_9_EDSA/3.1%20starter-notebook-checkpoint_.ipynb)

![pic](Pictures/explore-edsa-image.jpg)

This project was part of the requirement to complete the Advanced Regression Predict within EDSA's Data Science course. The government of Spain was considering an expansion of it's renewable energy resource infrastructure investments. As such, they require information on the trends and patterns of the countries renewable sources and fossil fuel energy generation.I was tasked to:

1. analyse the supplied data;
2. identify potential errors in the data and clean the existing data set;
3. determine if additional features can be added to enrich the data set;
4. build a model that is capable of forecasting the three hourly demand shortfalls;
5. evaluate the accuracy of the best machine learning model;
6. determine what features were most important in the model’s prediction decision, and
7. explain the inner working of the model to a non-technical audience.

It was a Time Series project of which I went through the entire data science lifecycle, including:

* Data wrangling / Data Preprocessing / Handling missing values
* Exploratory Data Analysis / Data Visualization
* Feature Extraction & Engineering
* Modelling and Hypertunning Best Model

**Keywords:** Data Wrangling, EDA, Feature Engineering, Handling missing values, Modelling, Hypertunning

---

**Future Interest:**  
`Demand Forecasting`,  `Predictive Maintenance`,  `Inventory Optimization`, `Supplier Performance Analysis`, `Route Optimization`, `Warehouse Optimization`, `Quality Control and Defect Detection`, `Supply Chain Risk Management`, `Customer Segmentation and Targeting`, `Optimal Sourcing Strategy`, `Return Rate Prediction`, `Sustainability and Green Logistics`.