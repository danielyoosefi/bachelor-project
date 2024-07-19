# This document focuses on the implementation of the decision tree model for the Heart Disease Dataset and the related formal methods/properties

# Before beginning implementation, the necessary libraries must be implemented and the dataset must be imported
# Import pandas for data manipulation, numpy for numerical operations and time for time manipulation
import pandas
import numpy

#------------------------------------------------------------------------------------------------------------------------------------------------------
# Import SKLearn Library for splitting the dataset, training a decision tree classifier, and calculating the accuracy
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle # For Temporal Consistency
from sklearn.model_selection import KFold # For Generalizability Verification

# Import the tree function for outputting the individual decision tree (Correctness)
from sklearn.tree import _tree

# Import Z3 Theorem Solver (Microsoft's Z3) (Correctness and Safety)
from z3 import Solver, If, sat, BoolVal, Real, And, Or, Not

# Import ART library for performing adversarial attacks (Robustness)
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import FastGradientMethod, BoundaryAttack

# Import psutil and time for scalability verification
import time
import psutil

# Import the heart disease dataset from local file
heart_disease_dataset = # Insert path to dataset on local machine

#------------------------------------------------------------------------------------------------------------------------------------------------------
# This section of the code focuses on loading the dataset and preprocessing the data
# Define names for the numerical and categorical features of the dataset (the columns of the dataset)
dataset_features = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','heart_disease']

# Read the dataset from the defined file path, while assigning the predetermined feature names and replacing missing values with a questionmark
data = pandas.read_csv(heart_disease_dataset,header=None,names=dataset_features,na_values="?")

# Remove rows of data within the dataset that have missing values (part of preprocessing)
data.dropna(inplace=True)

# Store all columns in the dataset that contain features (excluding the heart_disease column which indicates whether a patient has heart disease)
features = data.drop('heart_disease',axis=1)

# Store the heart_disease column, which indicates whether a patient does or does not have heart disease. (HDV = heart disease value)
hdv = data['heart_disease']

# Since the result data is a value between 0 and 1, the hdv is classified as >0 = 1 = heart disease present. Otherwise hdv = 0 = no heart disease prsent.
hdv = numpy.where(hdv>0,1,0)

# The dataset is split into training and testing dataset. Here 80% of the dataset is used for training and 20% is used for testing.
train_features,test_features,train_hdv,test_hdv = train_test_split(features,hdv,test_size=0.2,random_state=42)

# Reset index
test_features.reset_index(drop=True, inplace=True)

# Create decision tree model and train it on the training subsets (using the fit() function)
dectree_model = DecisionTreeClassifier(random_state=42)
dectree_model.fit(train_features,train_hdv)

# Now that the model is trained, it can make predictions using the test subset. The accuracy is then evaluated by comparing the predictions to test_hdv values.
predictions = dectree_model.predict(test_features)
accuracy = accuracy_score(test_hdv,predictions)

# Now that the training, predictions and evaluation aspects have concluded, the models accuracy is printed for the console to be recorded.
print(f"The Decision Tree Model has successfully been trained on the Heart Disease Dataset.")
print(f"The achieved accuracy is: {accuracy:.2f}")

#------------------------------------------------------------------------------------------------------------------------------------------------------
# This section focuses on implementing the code for the correctness formal method.
# This function outputs the rules used by the decision tree, which can be used to verify correctness. 
def print_dectree_rules(dectree,dectree_features):
    # Access the structure of the tree and get the name of each nodes features
    tree_ = dectree.tree_
    f_name = [dectree_features[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]
    # Function to recursively traverse the tree
    def traverse_tree(tree_depth,node):
        # Variable to indent the output to make the tree more readable
        output_indent = "  " * tree_depth
        # Check if the node is a leaf. If not, print the rule and continue recursion
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            # Get the threshold value and feature name
            temp_threshold = tree_.threshold[node]
            temp_name = f_name[node]
            # Output rules and recurse. Left first, then right.
            print(f"{output_indent}if {temp_name} <= {temp_threshold:.2f}")
            traverse_tree(tree_depth+1,tree_.children_left[node])
            print(f"{output_indent}else:  # if {temp_name} > {temp_threshold:.2f}")
            traverse_tree(tree_depth+1,tree_.children_left[node])
        # If node is a leaf, then print it
        else:
            print(f"{output_indent}return {tree_.value[node]}")
    # Begin traversing the tree from its root node (node 0, depth = 1)
    traverse_tree(1,0)


def verify_dectree_correctness(dectree,test_features,test_hdv):
    # Reset index
    test_features.reset_index(drop=True, inplace=True)
    # Ensure test_hdv is numpy array to avoid out of bounds exception
    test_hdv = numpy.asarray(test_hdv)
    # Access the structure of the tree and get the name of each nodes features
    tree_ = dectree.tree_
    f_name = [test_features.columns[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]
    # Function to recursively traverse the tree
    def traverse_tree(node,tree_depth,row):
        # Check if the node is a leaf. If not, print the rule and continue recursion
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            # Get the threshold value and feature name
            temp_threshold = tree_.threshold[node]
            temp_name = f_name[node]
            # Store left and right child
            temp_left = tree_.children_left[node]
            temp_right = tree_.children_right[node]
            # Create If statement for Z3
            return If(row[temp_name] <= temp_threshold,traverse_tree(temp_left,tree_depth+1,row),traverse_tree(temp_right,tree_depth+1,row))
        # If node is a leaf, then print it
        else:
            # Obtain value and check if there is only one class. If yes, return z3 bool, else comapre classes
            value = tree_.value[node]
            if value.shape[1] == 1:
                return BoolVal(value[0,0]>0)
            else:
                return BoolVal(value[0,1]>value[0,0])
    # Create instance of Z3 solver
    z3_solver = Solver()
    # Define counter for the number of correct predictions
    prediction_counter = 0
    # Traverse test sample
    for idx, row in test_features.iterrows():
        # Check for out of bounds exception
        if idx >= len(test_hdv):
            print(f"Index {idx} out of bounds with {len(test_hdv)}")
            break
        # Get decision rule for specific prediction
        temp_prediction = traverse_tree(0,0,row)
        # Add the expression to Z3 solver
        z3_solver.add(temp_prediction == BoolVal(test_hdv[idx] == 1))
        # Check if the logical expression is satisfiable and if yes, increment counter
        if z3_solver.check() == sat:
            prediction_counter +=1
        # Reset solver for next iteration
        z3_solver.reset()
    # Calculate accuracy
    return prediction_counter/len(test_hdv)

#------------------------------------------------------------------------------------------------------------------------------------------------------
# This section focuses on implementing the code for the safety formal method.
# This function verifies the safety of the model using Z3 by creating a constraint
def verify_model_safety(model_to_test,test_features,test_hdv):
    # Reset index
    test_features.reset_index(drop=True,inplace=True)
    # Create instance of Z3 solver
    z3_solver = Solver()
    # The implemented safety constraint is: High-risk patients should not be classified as low risk
    # The metrics for high risk include age > 60 and cholesterol > 240. Low risk patients have no heart disease.
    # First, the high-risk patients are filtered out based on age and cholesterol
    high_risk_patients=test_features[(test_features['age'] > 60) | (test_features['chol'] > 240)]
    # Now the model is used to make predictions for these patients
    low_risk_pred = model_to_test.predict(high_risk_patients)
    # Initialize a variable to count how often this constraint is violated
    violation_counter = 0
    # Iterate through high-risk features
    for idx, row in high_risk_patients.iterrows():
        # Create variable for Z3
        test_age = Real('age')
        test_cholesterol = Real('chol')
        test_prediction = Real('prediction')
        # Add constraints to Z3
        z3_solver.push()
        z3_solver.add(test_age == row['age'])
        z3_solver.add(test_cholesterol == row['chol'])
        z3_solver.add(test_prediction == low_risk_pred[idx])
        # Define the conditions for high-risk or low-risk patients
        high_risk = Or(test_age > 60, test_cholesterol > 240)
        low_risk = test_prediction == 0
        # Define that if a high risk is predicted as low risk, it is a violation
        z3_solver.add(And(high_risk, low_risk))
        # Check the constraints. If they are violated, increment counter.
        if z3_solver.check() == sat:
            violation_counter += 1
        # Restore solver
        z3_solver.pop()
        # Calculate total number of patients that are high-risk and calculate safety compliance
        num_high_risk = len(high_risk_patients)
        safety_compliance = 1-(violation_counter/num_high_risk)
        return safety_compliance

#------------------------------------------------------------------------------------------------------------------------------------------------------
# This section focuses on implementing the code for the robustness formal method.
# This function performs adversarial attacks on the model using the ART library to verify robustness.
def verify_model_robustness(model,test_features,test_labels):
    # Create new Classifier for adversarial attacks
    art_classifier = SklearnClassifier(model=model)
    # Create attack using the ART library for adversarial attacks depending on which model is being used
    if isinstance(model, (DecisionTreeClassifier,RandomForestClassifier,KNeighborsClassifier,SVC)):
        # Use a boundary attack as gradient based attacks do not work on tree-based models
        art_attack = BoundaryAttack(estimator=art_classifier,targeted=False,max_iter=100)
    elif isinstance(model, LogisticRegression):
        # Use fast gradient method to perform art attack on logreg model
        art_attack = FastGradientMethod(estimator=art_classifier,eps=0.2)
    # Generate samples for attack
    test_samples = art_attack.generate(x=test_features.to_numpy())
    # Convert samples back to pandas DataFrame
    test_samples_new = pandas.DataFrame(test_samples,columns=test_features.columns)
    # Make new predictions using the classifier
    art_predictions = art_classifier.predict(test_samples_new)
    # Calculate new accuracy of the model after the adversarial attack
    new_accuracy = accuracy_score(test_labels,numpy.argmax(art_predictions,axis=1))
    # Return new accuracy
    return new_accuracy

#------------------------------------------------------------------------------------------------------------------------------------------------------
# This section focuses on implementing the code for the fairness formal method.
# The following functions calculates the disparate impact ratio and equal opportunity difference for the model, two metrics that indicate fairness
def verify_model_fairness(model_to_test,test_features,test_hdv,demographic_features,temp=1):
    # Make predictions on the features using the model
    model_pred = model_to_test.predict(test_features)
    # Defined indices for each demographic group based on feature
    first_group = test_features[demographic_features] == 0
    second_group = test_features[demographic_features] == 1
    # Calculate proportion of positives outcomes for each group
    pos_first = sum(model_pred[first_group] == temp) / sum(first_group)
    pos_second = sum(model_pred[second_group] == temp) / sum(second_group)
    # Calculate true positive for each group
    true_pos_first = sum((model_pred[first_group] == temp) & (test_hdv[first_group] == temp))
    true_pos_second = sum((model_pred[second_group] == temp) & (test_hdv[second_group] == temp))
    # Calculate actual positives for each group
    actual_pos_first = sum(test_hdv[first_group] == temp)
    actual_pos_second = sum(test_hdv[second_group] == temp)
    # Calculate true positive rate for each group
    true_pos_rate_first = true_pos_first/actual_pos_first if actual_pos_first != 0 else 0
    true_pos_rate_second = true_pos_second/actual_pos_second if actual_pos_second != 0 else 0
    # Calculate disparate impact ratio (divide positive rate of second group by first group)
    disparate_impact_ratio = pos_second / pos_first
    # Calculate equal opportunity difference (subtract true pos rate of first group fro second)
    equal_opportunity_difference = true_pos_rate_second - true_pos_rate_first
    # Return both metrics
    return disparate_impact_ratio,equal_opportunity_difference

#------------------------------------------------------------------------------------------------------------------------------------------------------
# This section focuses on implementing the code for the temporal consistency formal method.
# The following function simulates collecting data over time by shuffling the data and executing the model multiple times to return an average accuracy
# The parameter n defines how many iterations of executing the model should be completed
def verify_model_temporal_consistency(model_to_test,model_features,model_hdv, n=5):
    # First, a list is defined to store the accuracies
    acc_list = []
    # Iterate for each run:
    for _ in range(n):
        # Shuffle the data for the model
        shuffled_features,shuffled_hdv = shuffle(model_features,model_hdv,random_state=42)
        # Split the data that was shuffled into sets
        train_features,test_features,train_hdv,test_hdv = train_test_split(shuffled_features,shuffled_hdv,test_size=0.2,random_state=42)
        # Train the model
        model_to_test.fit(train_features,train_hdv)
        # Generate predictions using the trained model
        model_pred = model_to_test.predict(test_features)
        # Compute accuracies and append them to the accuracy list
        model_acc = accuracy_score(test_hdv,model_pred)
        acc_list.append(model_acc)
    # Return accuracy list
    return acc_list

#------------------------------------------------------------------------------------------------------------------------------------------------------
# This section focuses on implementing the code for the generalizability formal method.
# This function performs k-cross fold validation, where the data is split into five folds. Four folds are used for training and one for testing.
# The new accuracy indicates generalizability.
def verify_model_generalizability(model_to_test,model_features,model_target):
    # Define the number of folds
    folds = 5
    # Do the k-fold operation with k splits and shuffle the data
    test_folds = KFold(folds,shuffle=True,random_state=42)
    # Define a list to store accuracies
    acc_list = []
    # Traverse each fold in for loop
    for train, test in test_folds.split(model_features):
        # Split data for this fold
        fold_train, fold_test = features.iloc[train], features.iloc[test]
        # Split target data for this fold
        fold_target_train, fold_target_test = model_target[train],model_target[test]
        # Train the model
        model_to_test.fit(fold_train,fold_target_train)
        # Make predictions using the model
        model_pred = model_to_test.predict(fold_test)
        # Comput accuracy and append to list
        model_acc = accuracy_score(fold_target_test,model_pred)
        acc_list.append(model_acc)
    # Calculate mean accuracy and return
    result_acc = numpy.mean(acc_list)
    return result_acc

#------------------------------------------------------------------------------------------------------------------------------------------------------
# This section focuses on implementing the code for the scalability formal method.
# This function splits uses the dataset at 50% and at 100%. The memory and time it takes for the model to complete is then returned.
def verify_model_scalability(model_to_test,model_features,model_hdv):
    # Function to measure time and memory 
    def measure_complexity(model_to_test,features,hdv):
        init_time = time.time()
        mem_proc = psutil.Process()
        init_mem = mem_proc.memory_info().rss
        model_to_test.fit(features,hdv)
        complete_time = time.time()
        complete_mem = mem_proc.memory_info().rss
        return complete_time - init_time, (complete_mem-init_mem) / (1024 ** 2)
    # Split dataset into 50% and 100%
    split_features_half, _, split_hdv_half, _ = train_test_split(model_features,model_hdv,test_size=0.5,random_state=42)
    # Complete verification for 50% of dataset
    time_half, mem_half = measure_complexity(model_to_test,split_features_half,split_hdv_half)
    # Complete verification for 100% of dataset
    time_full, mem_full = measure_complexity(model_to_test,model_features,model_hdv)
    # Return obtained values
    return time_half,time_full,mem_half,mem_full

#------------------------------------------------------------------------------------------------------------------------------------------------------
# This section implements the functionality for a user to call upon the functions for each integrated formal method
def menu_interface():
    while True:
        print("\nNow that the model has successfully compiled, please choose which functionality to execute. The following optiuons are available:")
        print("0. Exit")
        print("1. Correctness: Print decision tree rules")
        print("2. Correctness: Verify Model using Z3 and Print Accuracy")
        print("3. Check Safety Compliance using Z3")
        print("4. Check Model Robustness using Adversarial Attacks (ART)")
        print("5. Verify Fairness by calculating DIR and EOD Metrics")
        print("6. Verify Temporal Consistency and Print new Mean Accuracy")
        print("7. Verify Generalizability using K-Fold Validation")
        print("8. Verify Scalability and Output Memory and Time used")
        user_selection = input("Please enter the number that you have selected: ").strip()

        match user_selection:
            case '1':
                print("Decision Tree Rules:")
                print_dectree_rules(dectree_model,features.columns)
            case '2':
                correctness_accuracy = verify_dectree_correctness(dectree_model,test_features,test_hdv)
                print(f"New Correctness Verified Accuracy using Z3: {correctness_accuracy:.2f}")
            case '3':
                safety_compliance = verify_model_safety(dectree_model,test_features,test_hdv)
                print(f"Safety Compliance: {safety_compliance:.2f}")
            case '4':
                robustness_accuracy = verify_model_robustness(dectree_model,test_features,test_hdv)
                print(f"The Robustness Accuracy of the Model after Adversairal Attacks is: {robustness_accuracy:.2f}")
            case '5':
                # Select demographic feature to test fairness
                demographic_feature = 'sex'
                disp_imp_rat, equ_opp_dif = verify_model_fairness(dectree_model,test_features,test_hdv,demographic_feature)
                print(f"Disparate Impact Ratio for Model: {disp_imp_rat:.2f}")
                print(f"Equal Opportunity Difference for Model: {equ_opp_dif:.2f}")
            case '6':
                temporal_consistency_acc = verify_model_temporal_consistency(dectree_model,features,hdv)
                mean_acc = numpy.mean(temporal_consistency_acc)
                print(f"Average accuracy over iterations: {mean_acc:.2f}")
                print(f"Individal Accuracies over all runs: {temporal_consistency_acc}")
            case '7':
                generalizability_acc = verify_model_generalizability(dectree_model,features,hdv)
                print(f"Model Accuracy after Generalizability Verification: {generalizability_acc:.2f}")
            case '8':
                time_half,time_full,mem_half,mem_full = verify_model_scalability(dectree_model,features,hdv)
                print(f"Using 50% of Dataset; Time: {time_half:.2f}s, Memory: {mem_half:.2f}MB")
                print(f"Using 100% of Dataset; Time: {time_full:.2f}s, Memory: {mem_full:.2f}MB")
            case '0':
                print("Exiting.")
                break

# Run the interface for the user to either execute the function for a formal method or exit the program.
menu_interface()
