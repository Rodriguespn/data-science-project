### China Air Quality Dataset

---

[Dataset description](!https://en.wikipedia.org/wiki/Air_quality_index#China)

---

#### Data Profiling

Dataset statistics:
- Number of variables: 32
- Number of observations: 169273
- Missing cells: 202308
- Missing cells (%): 3.7%
- Duplicate rows: 0
- Duplicate rows (%): 0.0%

Variable types:

- DateTime: 1
- Numeric: 27
- Categorical: 3
- Unsupported: 1

Missing Values:

 - Field_1 - alguns missing values (10,1%)
 - CO_Mean - alguns missing values (4,6%)
 - CO_Min - alguns missing values (4,6%)
 - CO_Max - alguns missing values (4,6%)
 - CO_Std - alguns missing values (4,6%)
 - NO2_Mean - alguns missing values (4,6%)
 - NO2_Min - alguns missing values (4,6%)
 - NO2_Max - alguns missing values (4,5%)
 - NO2_Std - alguns missing values (4,6%)
 - O3_Mean - alguns missing values (4,6%)
 - O3_Min - alguns missing values (4,6%)
 - O3_Max - alguns missing values (4,6%)
 - O3_Std - alguns missing values (4,6%)
 - PM2.5_Mean - alguns missing values (4,5%)
 - PM2.5_Min - alguns missing values (4,5%)
 - PM2.5_Max - alguns missing values (4,5%)
 - PM2.5_Std - alguns missing values (4,5%)
 - PM10_Mean - alguns missing values (4,5%)
 - PM10_Min - alguns missing values (4,5%)
 - PM10_Max - alguns missing values (4,5%)
 - PM10_Std - alguns missing values (4,5%)
 - SO2_Mean - alguns missing values (4,6%)
 - SO2_Min - alguns missing values (4,6%)
 - SO2_Max - alguns missing values (4,6%)
 - SO2_Std - alguns missing values (4,6%)
 
Outliers:

- CO_Std - 

Correlations:

- 


---

Modeling:

**target = ALARM**

 - Classification, file = air_quality_tabular.csv
 - Non-Supervised, file = = air_quality_tabular.csv (remove the target column)

**target = AQI_BEIJING**

 - Forecasting, file = air_quality_timeseries.csv