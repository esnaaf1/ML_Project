from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, minmax_scale, MaxAbsScaler, Normalizer, QuantileTransformer, PowerTransformer

# Define a function that uses MinMaxScaler on the data

def prepData(data, scaler):
    
    # call pandas get dummies to convert to 1's and 0's in separate columns
    X = pd.get_dummies(copyData)
    
    # Split the data into test and train groups
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
    
    # Save data out for use in other Jupyter Notebooks
    X_train_csv = X_train.to_csv ('X_train.csv', index = None, header=True)
    X_test_csv = X_test.to_csv ('X_test.csv', index = None, header=True)
    y_train_csv = y_train.to_csv ('y_train.csv', index = None, header=True)
    y_test_csv = y_test.to_csv ('y_test.csv', index = None, header=True)
    

    # Apply a scalar, first train then scale test and train data
    # TO DO:  Insert function to select scalar
    X_scaler = scaler.fit(X_train)
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    # Label encode the dependent variable, then Hot encode it
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    y_train_categorical = to_categorical(label_encoder.transform(y_train))
    y_test_categorical = to_categorical(label_encoder.transform(y_test))

    return X, y, X_train_scaled, X_test_scaled, y_train_categorical, y_test_categorical
