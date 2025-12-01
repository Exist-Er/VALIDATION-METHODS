import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split,
    LeaveOneOut,
    StratifiedKFold,
    KFold,
    RepeatedKFold
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

original_df = pd.read_csv("archive/Iris.csv")
df = original_df.copy()
species_names = original_df['Species'].unique()

models = {
    "knn": KNeighborsClassifier(),
    "logreg": LogisticRegression(max_iter=500),
    "dtree": DecisionTreeClassifier()
}

current_validation = "Holdout"
is_shuffled = False

def holdout_validation(shuffled_df):
    X = shuffled_df.drop(["Species", "Id"], axis=1)
    y = shuffled_df["Species"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y
    )
    return X_train, X_test, y_train, y_test

def loocv_validation(shuffled_df):
    X = shuffled_df.drop(["Species", "Id"], axis=1).values
    y = shuffled_df["Species"].values
    loo = LeaveOneOut()
    return X, y, loo

def stratified_cv_validation(shuffled_df, k=5):
    X = shuffled_df.drop(["Species", "Id"], axis=1).values
    y = shuffled_df["Species"].values
    skf = StratifiedKFold(n_splits=k, shuffle=True)
    return X, y, skf

def kfold_validation(shuffled_df, k=5):
    X = shuffled_df.drop(["Species", "Id"], axis=1).values
    y = shuffled_df["Species"].values
    kf = KFold(n_splits=k, shuffle=True)
    return X, y, kf

def repeated_kfold_validation(shuffled_df, k=5, r=3):
    X = shuffled_df.drop(["Species", "Id"], axis=1).values
    y = shuffled_df["Species"].values
    rkf = RepeatedKFold(n_splits=k, n_repeats=r)
    return X, y, rkf

def plot_confusion_matrices(matrices, model_full_names):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, (name, matrix) in enumerate(matrices.items()):
        ax = axes[i]
        im = ax.imshow(matrix, cmap='viridis', interpolation='nearest')
        ax.set_xticks(np.arange(len(species_names)))
        ax.set_yticks(np.arange(len(species_names)))
        ax.set_xticklabels(species_names)
        ax.set_yticklabels(species_names)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        for r in range(len(species_names)):
            for c in range(len(species_names)):
                ax.text(c, r, matrix.iloc[r, c],
                               ha="center", va="center", color="w")
        ax.set_title(model_full_names[i])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.tight_layout()
    plt.show()

def perform_all_models(shuffled_df, vmethod):
    print(f"\n--- {vmethod} Validation ---")

    def run_and_get_matrix(model, X_train, X_test, y_train, y_test):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        cm = confusion_matrix(y_test, preds, labels=species_names)
        acc = accuracy_score(y_test, preds)
        return cm, acc

    def run_cv_and_get_matrix(model, X, y, cv):
        preds = []
        actual = []
        for train_index, test_index in cv.split(X, y):
            model.fit(X[train_index], y[train_index])
            p = model.predict(X[test_index])
            preds.extend(p)
            actual.extend(y[test_index])
        cm = confusion_matrix(actual, preds, labels=species_names)
        acc = accuracy_score(actual, preds)
        return cm, acc

    if vmethod == "Holdout":
        X_train, X_test, y_train, y_test = holdout_validation(shuffled_df)
        eval_func = lambda model: run_and_get_matrix(model, X_train, X_test, y_train, y_test)
    elif vmethod == "LOOCV":
        X, y, cv = loocv_validation(shuffled_df)
        eval_func = lambda model: run_cv_and_get_matrix(model, X, y, cv)
    elif vmethod == "Stratified":
        X, y, cv = stratified_cv_validation(shuffled_df)
        eval_func = lambda model: run_cv_and_get_matrix(model, X, y, cv)
    elif vmethod == "KFold":
        X, y, cv = kfold_validation(shuffled_df)
        eval_func = lambda model: run_cv_and_get_matrix(model, X, y, cv)
    elif vmethod == "Repeated":
        X, y, cv = repeated_kfold_validation(shuffled_df)
        eval_func = lambda model: run_cv_and_get_matrix(model, X, y, cv)
    else:
        raise ValueError("Unknown validation method")

    matrices = {}
    accuracies = {}

    for name, model in models.items():
        cm, acc = eval_func(model)
        matrices[name] = pd.DataFrame(cm, index=species_names, columns=species_names)
        accuracies[name] = acc

    names_order = ["knn", "logreg", "dtree"]
    model_full_names = ["K Nearest Neighbour", "Logistic Regression", "Decision Tree"]
    
    plot_confusion_matrices(matrices, model_full_names)

    print("\nAccuracies:")
    for name in names_order:
        print(f"{name.upper():10} : {accuracies[name]:.4f}")

    return matrices, accuracies

def choose_validation():
    print("\nChoose validation method:")
    print("1. Holdout")
    print("2. LOOCV")
    print("3. Stratified K-Fold")
    print("4. K-Fold")
    print("5. Repeated K-Fold")
    ch = input("Your choice: ").strip()
    if ch == "1": return "Holdout"
    if ch == "2": return "LOOCV"
    if ch == "3": return "Stratified"
    if ch == "4": return "KFold"
    if ch == "5": return "Repeated"
    print("Invalid choice. Defaulting to Holdout.")
    return "Holdout"

last_matrices = {"knn": None, "logreg": None, "dtree": None}
last_accuracies = {"knn": None, "logreg": None, "dtree": None}

while True:
    print(f"\nValidation: {current_validation}, Shuffled: {{'Yes' if is_shuffled else 'No'}}")
    
    print("\nMenu:")
    print("1. Change validation method")
    print("2. Shuffle data")
    print("3. Reset data")
    print("4. Run evaluation")
    print("5. Quit")

    choice = input("Your choice: ").strip()

    if choice == "1":
        current_validation = choose_validation()
    elif choice == "2":
        df = df.sample(frac=1).reset_index(drop=True)
        is_shuffled = True
        print("Data shuffled.")
    elif choice == "3":
        df = original_df.copy()
        is_shuffled = False
        print("Data reset.")
    elif choice == "4":
        matrices, accuracies = perform_all_models(df, current_validation)
        last_matrices.update(matrices)
        last_accuracies.update(accuracies)
    elif choice == "5":
        print("Farewell, traveler.")
        break
    else:
        print("Invalid choice.")
