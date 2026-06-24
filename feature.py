from import_lib import *
from data_cleaning import *

#label encodeing for target varible

df['HeartDisease'] = df['HeartDisease'].map({'Yes': 1 , 'No' : 0})

#Label encoding for binary values
df['Smoking'] = df['Smoking'].map({'Yes': 1 , 'No' : 0})
df['AlcoholDrinking'] = df['AlcoholDrinking'].map({'Yes': 1 , 'No' : 0})
df['Stroke'] = df['Stroke'].map({'Yes': 1 , 'No' : 0})
df['DiffWalking'] = df['DiffWalking'].map({'Yes': 1 , 'No' : 0})
df['Sex'] = df['Sex'].map({'Male': 1 , 'Female' : 0})
df['PhysicalActivity'] = df['PhysicalActivity'].map({'Yes': 1 , 'No' : 0})
df['Asthma'] = df['Asthma'].map({'Yes': 1 , 'No' : 0})
df['KidneyDisease'] = df['KidneyDisease'].map({'Yes': 1 , 'No' : 0})
df['SkinCancer'] = df['SkinCancer'].map({'Yes': 1 , 'No' : 0})

#one hot encoding for categorical feature
data_cat = pd.get_dummies(df[['AgeCategory','Race','GenHealth','Diabetic']],drop_first=True,dtype=int)
df = pd.concat([df, data_cat], axis = 1)
df.drop(['AgeCategory','Race','GenHealth','Diabetic'], axis = 1, inplace = True)

print(df.info())
print(df.head())

# Add new Feature

# add BMI_Category
def bmi_category(bmi):
    if bmi < 18.5:
        return 0 # underweight 
    elif bmi < 25:
        return 1  # normal
    elif bmi < 30:
        return 2 # overweight
    else:
        return 3 #obesity

df['BMI_Category'] = df['BMI'].apply(bmi_category)


# BMI_Risk flag 
df['BMI_Risk'] = (df['BMI'] > 30).astype(int)

#total_unhealthy_days

df['total_unhealthy_days'] = df['PhysicalHealth'] + df['MentalHealth']

#Chronic_count is a total of stroke,Asthma,KidnetDisease,SkinCancer
df['chronic_count'] = (
    df['Stroke'] +
    df['Asthma'] +
    df['KidneyDisease'] +
    df['SkinCancer']
)

df['risk_score'] = (
    df['Smoking'] +
    df['Diabetic_Yes'] +
    df['DiffWalking'] +
    df['Stroke']
)

# Age_index 
df['Age_Index'] = df[
    [col for col in df.columns if 'AgeCategory_' in col]
].idxmax(axis=1).str.extract('(\d+)').astype(int)