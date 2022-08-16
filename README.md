# Data Science Portfolio

## About Me

I am a Data Scientist, graduate of the Explore’s Data Science Accelerator [Oct 2021] program, and completed my first degree in Civil Engineering at Nnamdi Azikiwe University Awka. More information about my academic journey can be found in [My Website](https://sites.google.com/view/e-israel/home), [My Linkedin](https://www.linkedin.com/in/israel-ezema-009530195/), [My Resume](https://drive.google.com/file/d/1BLaZAj37Xn0cVAKfArk0Q1p6G0MaVcbC/view?usp=sharing), and [My Kaggle](https://www.kaggle.com/israelezema)

## Projects

This is a list of some machine learning projects I worked on or currently working on. It is updated regularly. Click on the projects to see full analysis and code.

---

### [EDSA/AiGlass: Carbon Emission Analysis & Prediction](https://github.com/EDSA-Internship-Group-3/co2-emission-analysis)

![Top 20 Countries with CO2_Emission_per_Capita](https://github.com/EDSA-Internship-Group-3/co2-emission-analysis/blob/main/notebook_n_data/graphs/Top%2020%20Countries%20with%20Co2_emission_per_Capita.png)

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

What I learned from this project:
* It's entirely wrong to state that the larger the population, the more CO2 the country will be likely to emit. It's more dependent on the Activity, Culture, and policies of a place to reduce the CO2 Emitted per head/capita index.
* The larger the Energy Consumption of a country, the larger the CO2 emission, which is dependent on the prevalent Energy type being consumed in that region.
* The larger the GDP, the more likely the country will have a high CO2 emission, This Is NOT entirely true as the GDP of a place has many contributing factors and the essentials that correlate to CO2 Emission are the Manufacturing and Agricultural activities of the Place.
Coal and Petroleum/other liquids have been the dominant energy source contributing to CO2 Emitted globally.

**Keywords:** CO2_Emission_per_Capita, Energy Consumption, Population, GDP, XGBoost, RMSE.

---


### [EDSA/AI Incorporated: Streamlit-based: Movie Recommender System](https://github.com/2110ACDS-T4/unsupervised-predict-streamlit-template/blob/master/Team%204/Notebooks/3.0_EDSA_movie_recommendation_2022_Notebook.ipynb)

![Image header](https://github.com/2110ACDS-T4/unsupervised-predict-streamlit-template/blob/master/resources/imgs/Image_header.png)

This project was part of the requirement to complete the Unsupervised Predict within EDSA's Data Science course. It required us to build and deploy a basic recommender engine based upon the Streamlit web application framework. In the end, I completed the following;

* Constructed several recommendation algorithm based on Content and Collaborative filtering
* Applied Model Versioning with COMET
* Carried out Several Data Processing, Exploratory Data Analysis, and Feature Engineering techniques
* Applied the 'Did you mean?' Algorithm (Levenshtein Distance)
* Deploying our best Model to a Streamlit Web App.

**Keywords:** Recommendation algorithm, Streamlit web deployment, COMET versioning, Exploratory Data Analysis (EDA)

---


### [EDSA/DataWare Solutions - Climate Change Belief Analysis 2022](https://github.com/2110ACDS-T12/classification-predict-streamlit-template/blob/master/Project%20File/5.0%20Advance_Classification_Notebook.ipynb)

![pic](https://camo.githubusercontent.com/c57ebd961169e55085c23e5cea0eddaef383d1aa5ea187f122d6c293552573f5/68747470733a2f2f696d67732e7365617263682e62726176652e636f6d2f6d2d5374454971416f723650656a7a684f31514971566677594650347a6e726645436f30504e7a4464634d2f72733a6669743a313230303a3738383a312f673a63652f6148523063484d364c793930595856692f62574675593239736247566e5a5335312f62576c6a6143356c5a48557663326c302f5a584d765a47566d595856736443396d2f6157786c6379397a64486c735a584d762f5a6d3931636c396a62327831625735662f5a6d5668644856795a53397764574a732f61574d765a6d5668644856795a5752662f615731685a32567a4c304e73615731682f6447557451326868626d646c4c555a6c2f59584a7a587a4175616e426e50326c302f623273395654567063326872515849)

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

### [EDSA Classification Hackathon: South African language Identification](https://github.com/Ezysticks/south_african_language_identification)

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

### [EDSA: Spain Electricity Shortfall Challenge](https://github.com/Ezysticks/load-shortfall-regression-predict-api/blob/master/Advanced-Regression-Starter-Data/Versions_Team_9_EDSA/3.1%20starter-notebook-checkpoint_.ipynb)

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
