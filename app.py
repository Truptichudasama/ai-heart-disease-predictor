import os
import sys
import feature
# Ensure the script can find 'imports_lib.py' if it's in the root
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import your shared libraries (ensures Scikit-Learn components are loaded)
try:
    from import_lib import *
except ImportError:
    print("Warning: imports_lib.py not found. Ensure sklearn and pandas are installed.")

print("successfully import lib")


app = Flask(__name__,template_folder='.')

# --- CONFIGURATION ---
# Paths to your saved artifacts from the training phase
MODEL_PATH = "model_classifier.pkl"
# Load the model , once when the server starts
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("✅ Model  loaded successfully.")
else:
    print("❌ Error: .pkl files not found in the root directory.")

# --- ROUTES ---

@app.route('/')
def home():
    """Renders the HTML input form from the /templates folder."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    Sex = int(request.form['Sex'])
    BMI = float(request.form['BMI'])
    AgeCategory = int(request.form['AgeCategory'])
    Race = int(request.form['Race'])
    Smoking = int(request.form['Smoking'])
    AlcoholDrinking = int(request.form['AlcoholDrinking'])
    PhysicalActivity = int(request.form['PhysicalActivity'])
    Stroke = int(request.form['Stroke'])
    Asthma = int(request.form['Asthma'])
    KidneyDisease = int(request.form['KidneyDisease'])
    SkinCancer = int(request.form['SkinCancer'])
    DiffWalking = int(request.form['DiffWalking'])
    PhysicalHealth = int(request.form['PhysicalHealth'])
    MentalHealth = int(request.form['MentalHealth'])
    SleepTime = int(request.form['SleepTime'])
    Diabetic = int(request.form['Diabetic'])
    GenHealth = int(request.form['GenHealth'])

    print("Received Data:")
    print(request.form)
    #BMI_Category
    BMI_Category = feature.bmi_category(BMI)
    print("****************************")
    print("BMI_category_val:::",BMI_Category)
    print("BMI:::::::::::::::::::::::::::::::::::",BMI)
    BMI_Risk = 1 if BMI >30 else 0
    print("BMI_risk:::::",BMI_Risk)


    #total_unhealthy_days
    total_unhealthy_days = PhysicalHealth + MentalHealth
    print("total_unhealthy_days::",total_unhealthy_days)

    #Chronic_count is a total of stroke,Asthma,KidnetDisease,SkinCancer
    chronic_count = Stroke + Asthma + KidneyDisease + SkinCancer
    print("chronic_count::",chronic_count)

    risk_score = Smoking + DiffWalking + Stroke + (Diabetic if Diabetic == 2 else 0)
    print("risk_score:: ",risk_score)

    GenHealth_Poor      = int(GenHealth == 0)
    GenHealth_Fair      = int(GenHealth == 1)
    GenHealth_Good      = int(GenHealth == 2)
    GenHealth_VeryGood  = int(GenHealth == 3)
    GenHealth_Excellent = int(GenHealth == 4)

    print("&&&&&&&&&&&&&&&&&&&&&&GenHealth")
    print("GenHealth_Poor::",GenHealth_Poor, "GenHealth_Fair::",GenHealth_Fair,"GenHealth_Good::",GenHealth_Good,"GenHealth_VeryGood::",GenHealth_VeryGood,"GenHealth_Excellent::",GenHealth_Excellent)


    Diabetic_Yes = int(Diabetic == 2)
    Diabetic_Pregnancy  = int(Diabetic ==3)

    print("Diabetic_Yes::",Diabetic_Yes , "Diabetic_Yes (during pregnancy)" ,Diabetic_Pregnancy )


    Race_White = int(Race == 0)
    Race_Hispanic = int(Race == 1)
    Race_Black = int(Race == 2)
    Race_Asian = int(Race == 3)
    Race_Other = int(Race == 4)
    Race_American_Indian_Alaskan_Native = int(Race ==5)
    print("Race_White:",Race_White,"Race_Hispanic:",Race_Hispanic,"Race_Black:",Race_Black,"Race_Asian:",Race_Asian)
    print("Race_Other:",Race_Other,"Race_American_Indian_Alaskan_Native:",Race_American_Indian_Alaskan_Native)

    input_data = pd.DataFrame([{
        'risk_score': risk_score,
        'Age_Index': AgeCategory,
        'GenHealth_Poor': GenHealth_Poor,
        'Sex': Sex,
        'GenHealth_Fair': GenHealth_Fair,
        'GenHealth_Good': GenHealth_Good,
        'chronic_count': chronic_count,
        'Stroke': Stroke,
        'GenHealth_Very good': GenHealth_VeryGood,
        'KidneyDisease': KidneyDisease,
        'Diabetic_Yes': Diabetic_Yes,
        'PhysicalHealth': PhysicalHealth,
        'Race_White': Race_White,
        'AlcoholDrinking': AlcoholDrinking,
        'Race_Black': Race_Black,
        'DiffWalking': DiffWalking,
        'total_unhealthy_days': total_unhealthy_days,
        'Smoking': Smoking,
        'SkinCancer': SkinCancer,
        'BMI_Risk': BMI_Risk,
        'BMI_Category': BMI_Category,
        'BMI': BMI,
        'SleepTime': SleepTime,
        'Race_Asian': Race_Asian,
        'Race_Hispanic': Race_Hispanic,
        'Asthma': Asthma,
        'MentalHealth': MentalHealth,
        'PhysicalActivity': PhysicalActivity,
        'Race_Other': Race_Other,
        'Diabetic_Yes (during pregnancy)': Diabetic_Pregnancy
    }])

    print(model.feature_names_in_)
    print("input data********************************")
    print(input_data.columns.tolist())
    print("input data types:*****************************")
    print(input_data.dtypes)
    

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    print("prediction:::",prediction)
    print("probability::",probability)
    if prediction == 1:
        result = f"High Risk ({probability*100:.2f}%)"
    else:
        result = f"Low Risk ({(1-probability)*100:.2f}%)"

    return render_template(
        'result.html',
        prediction=result,
        probability=round(probability * 100, 2)
    )

   





if __name__ == '__main__':
    # Run the Flask server
    # Set debug=False in a real production environment
    app.run(debug=True, port=5000)