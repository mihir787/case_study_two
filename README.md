# Case Study Two: Predicting Job Attrition

## Introduction
DDSAnalytics is an analytics company that specializes in talent management solutions for Fortune 1000 companies. Talent management is defined as the iterative process of developing and retaining employees. It may include workforce planning, employee training programs, identifying high-potential employees and reducing/preventing voluntary employee turnover (attrition). To gain a competitive edge over its competition, DDSAnalytics decided to leverage data science for talent management. The executive leadership identified predicting employee turnover as its first application of data science for talent management. Before the business green lights the project, they tasked the data science team to conduct an analysis of existing employee data. The scope of this report is to summarize those findings.

To conduct exploratory data analysis (EDA), the data science team was provided with CaseStudy2Data.zip file to determine factors that lead to attrition. From this data, the team was asked to identify (at least) the top three factors that contribute to turnover, to learn about any job role specific trends that may exist in the data set (e.g., "Data Scientists have the highest job satisfaction") and to provide any other interesting trends and observations. All the Experiments and analysis were conducted in R.

## Executive Summary
There are three reliable factors that impact attrition: Overtime, Age, and Monthly Income. Greater Overtime requirements, younger employees, and low monthly incomes contribute significantly to attrition. Other factors such as Environment Satisfaction, Number of Companies Worked For, and Job Satisfaction can also be more minor contributors to job attrition. As executives, there are a few next steps we recommend you take. First, if more data can be acquired then we can build a stronger model. Currently, we have a strong sense of the top three contributing factors but we would like to get a better sense of the top five to ten factors. Secondly, location based data would be very beneficial for providing more insight into geographical areas, which would allow for more targeted strategies. Lastly, if we can get more data then we can more confidently address some of the less influential factors and see how attrition is impacted. For example, factors such as Job Involvement and Years In Current Role can be easier fixes than increasing Monthly Income. If we can build a stronger model that can expose factors which can be "easier wins" then those can be prioritized first.

## Approach
The approach that was taken was to build models in both Python and R and decide on the most influential factors based off the strongest models in each respective language. The python code is included in a Jupyter Notebook but is included (as not executable) in the Rmd file.

## Group Members
Mihir Parikh, Aaron Tomkins, and Lokesh Maganti
