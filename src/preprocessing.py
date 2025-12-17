from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def build_preprocessor(num_cols, cat_cols):
    transformers = []
    if num_cols:
        # Impute missing values with the mean, then scale
        num_pipeline = [
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]
        transformers.append(("num", Pipeline(num_pipeline), num_cols))
    if cat_cols:
        # Impute missing values with a constant 'missing', then one-hot encode
        cat_pipeline = [
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
        ]
        transformers.append(("cat", Pipeline(cat_pipeline), cat_cols))
    pre = ColumnTransformer(transformers)
    return pre
