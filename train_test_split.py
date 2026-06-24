

from feature import *

num_col = [ 'BMI','PhysicalHealth','MentalHealth','total_unhealthy_days','SleepTime', 'risk_score','chronic_count','Age_Index',]

cat_col = ['Sex','Smoking', 'AlcoholDrinking', 'Stroke','DiffWalking','PhysicalActivity','Asthma', 'KidneyDisease',
       'SkinCancer','Race_Asian', 'Race_Black', 'Race_Hispanic',
       'Race_Other', 'Race_White', 'GenHealth_Fair', 'GenHealth_Good',
       'GenHealth_Poor', 'GenHealth_Very good', 'Diabetic_No, borderline diabetes', 'Diabetic_Yes',
       'Diabetic_Yes (during pregnancy)','BMI_Category','BMI_Risk']


df =  df[num_col + cat_col + ['HeartDisease']  ]

# Split data for Trainning and Testing

y = df['HeartDisease']
x = df.drop('HeartDisease',axis=1)
def train_test_split_and_features(x,y):  
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.67, shuffle = False, random_state = 0)
    print(x.head(5))
    print(x.columns)
    features = list(x.columns)
    return x_train, x_test, y_train, y_test,features

x_train, x_test, y_train, y_test,features = train_test_split_and_features(x,y)

