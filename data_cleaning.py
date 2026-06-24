import sys
import os

# Adds the root directory (HEART/) to the search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from import_lib import *

df = pd.read_csv('Heart_disease.csv')
print(f"Data Loaded: {df.shape}")
print(df.head())
print("\n")

print(df.info())

def df_main_stats_plain(df: pd.DataFrame) -> pd.DataFrame:
    stats = pd.DataFrame(index=df.columns)

    # Basic info
    stats["dtype"] = df.dtypes
    stats["type"] = df.dtypes.apply(
        lambda x: "numerical" if pd.api.types.is_numeric_dtype(x) else "categorical"
    )

    # Missing values
    stats["missing"] = df.isna().sum()
    stats["missing_%"] = (df.isna().mean() * 100).round(2)

    # Cardinality
    stats["unique"] = df.nunique()
    stats["top"] = df.mode().iloc[0]

    # Numerical stats (only for numerical columns)
    stats["mean"] = df.mean(numeric_only=True)
    stats["median"] = df.median(numeric_only=True)
    stats["std"] = df.std(numeric_only=True)

    # Sort by missing percentage
    stats = stats.sort_values("missing_%", ascending=False)

    return stats

df_main_stats_plain(df)

print("Missing value imputation")
#Missing value_imputation

# Overwrite the dataframe with everything except those 2 rows
df = df[df['SkinCancer'] != 'SkinCancer']

#Numeric feature

# Convert to numeric first
df['BMI'] = pd.to_numeric(df['BMI'], errors='coerce')
df['PhysicalHealth'] = pd.to_numeric(df['PhysicalHealth'], errors='coerce')
df['MentalHealth'] = pd.to_numeric(df['MentalHealth'], errors='coerce')
df['SleepTime'] = pd.to_numeric(df['SleepTime'], errors='coerce')

num_imputer = SimpleImputer(strategy='median')
df[['BMI', 'PhysicalHealth']] = num_imputer.fit_transform(df[['BMI', 'PhysicalHealth']])

# Categorical_feature
cat_imputer = SimpleImputer(strategy='most_frequent')
df[['Sex', 'Smoking', 'GenHealth']] = cat_imputer.fit_transform(df[['Sex', 'Smoking', 'GenHealth']])
df[['SkinCancer','Diabetic','Asthma','DiffWalking']]=cat_imputer.fit_transform(df[['SkinCancer','Diabetic','Asthma','DiffWalking']])

print(df.shape)
