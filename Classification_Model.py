# Title: Classification Model using Streamlit

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.feature_selection import RFE, SelectKBest, chi2, SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
from statsmodels.stats.outliers_influence import variance_inflation_factor
from functools import reduce
import joblib

# Streamlit UI
def main():
    st.title("Classification Model using Streamlit")

    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("Dataset Preview")
        st.write(df.head())

        st.subheader("Data Info")
        st.text(str(df.dtypes))

        st.subheader("Descriptive Statistics")
        st.write(df.describe())

        st.subheader("Target Variable Distribution")
        if 'target' in df.columns:
            st.write(df['target'].value_counts(normalize=True))
        else:
            st.error("No column named 'target' found.")
            return

        st.subheader("Correlation Matrix")
        corr = df.corr(numeric_only=True)
        fig, ax = plt.subplots()
        sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, ax=ax)
        st.pyplot(fig)

        # Label Encoding for numeric and categorical columns
        d = defaultdict(preprocessing.LabelEncoder)
        for col in df.columns:
            if df[col].dtype == 'object' or str(df[col].dtype).startswith('category'):
                df[col] = d[col].fit_transform(df[col].fillna('NA'))
            elif np.issubdtype(df[col].dtype, np.number) and df[col].nunique() < 20 and col != 'target':
                # Label encode low-cardinality numeric columns (excluding target)
                df[col] = d[col].fit_transform(df[col].fillna(-9999))

        features = df.drop(columns=['target']).fillna(0)
        labels = df['target']

        # --- WOE and IV Calculation ---
        st.subheader("Information Value (IV) for Features")
        def calc_iv(df, feature, target):
            lst = []
            for val in df[feature].unique():
                total = len(df[df[feature] == val])
                event = len(df[(df[feature] == val) & (df[target] == 1)])
                non_event = len(df[(df[feature] == val) & (df[target] == 0)])
                event_rate = event / (df[target] == 1).sum() if (df[target] == 1).sum() > 0 else 0
                non_event_rate = non_event / (df[target] == 0).sum() if (df[target] == 0).sum() > 0 else 0
                woe = np.log((event_rate + 1e-8) / (non_event_rate + 1e-8))
                iv = (event_rate - non_event_rate) * woe
                lst.append({'Value': val, 'Event': event, 'NonEvent': non_event, 'WOE': woe, 'IV': iv})
            iv_df = pd.DataFrame(lst)
            return iv_df['IV'].sum()
        
        iv_list = []
        for col in features.columns:
            if col != 'target':
                iv = calc_iv(df, col, 'target')
                iv_list.append({'feature': col, 'IV': iv})
        iv_df = pd.DataFrame(iv_list)
        st.write(iv_df.sort_values('IV', ascending=False))
        # IV Bar Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=iv_df.sort_values('IV', ascending=False), x='feature', y='IV', ax=ax)
        ax.set_title('Information Value (IV) by Feature')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig)

        st.subheader("Running Random Forest for Feature Importance")
        clf_rf = RandomForestClassifier()
        clf_rf.fit(features, labels)
        vi_rf = pd.DataFrame(clf_rf.feature_importances_, columns=["RF"], index=features.columns).reset_index()
        st.write(vi_rf.sort_values("RF", ascending=False))

        st.subheader("Recursive Feature Elimination")
        model = LogisticRegression()
        rfe = RFE(model, n_features_to_select=20)
        fit_rfe = rfe.fit(features, labels)
        rfe_selected = pd.DataFrame(fit_rfe.support_, columns=["RFE"], index=features.columns).reset_index()
        st.write(rfe_selected[rfe_selected['RFE'] == True])

        st.subheader("ExtraTrees Feature Importance")
        model_et = ExtraTreesClassifier()
        model_et.fit(features, labels)
        vi_et = pd.DataFrame(model_et.feature_importances_, columns=["ExtraTrees"], index=features.columns).reset_index()
        st.write(vi_et.sort_values("ExtraTrees", ascending=False))

        st.subheader("Chi-Square Test")
        chi = SelectKBest(score_func=chi2, k='all')
        fit_chi = chi.fit(features.abs(), labels)
        chi_scores = pd.DataFrame(fit_chi.scores_, columns=["Chi_Square"], index=features.columns).reset_index()
        st.write(chi_scores.sort_values("Chi_Square", ascending=False))

        st.subheader("L1 Feature Selection")
        lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(features, labels)
        model_l1 = SelectFromModel(lsvc, prefit=True)
        l1 = pd.DataFrame(model_l1.get_support(), columns=["L1"], index=features.columns).reset_index()
        st.write(l1[l1['L1'] == True])

        # Combine all feature selection
        st.subheader("Vote Based Feature Selection")
        dfs = [iv_df, vi_rf.rename(columns={"index": "feature"}),
               vi_et.rename(columns={"index": "feature"}),
               chi_scores.rename(columns={"index": "feature"}),
               rfe_selected.rename(columns={"index": "feature"}),
               l1.rename(columns={"index": "feature"})]
        final_features = reduce(lambda left, right: pd.merge(left, right, on='feature'), dfs)
        final_features['final_score'] = (final_features[['IV', 'RF', 'ExtraTrees', 'Chi_Square']] > 0).astype(int).sum(axis=1) + \
                                        final_features[['RFE', 'L1']].astype(int).sum(axis=1)
        top_features = final_features.sort_values("final_score", ascending=False).head(10)
        st.write(top_features)

        selected_vars = top_features['feature'].tolist()
        features = features[selected_vars]

        # --- Feature Importance Cutoffs and Best Features ---
        st.subheader("Best Features by Each Importance Method")
        # Define cutoffs for each method (can be tuned)
        iv_cutoff = 0.02  # IV > 0.02 is considered useful
        rf_top_n = 5      # Top 5 by Random Forest
        et_top_n = 5      # Top 5 by ExtraTrees
        chi_top_n = 5     # Top 5 by Chi2
        rfe_selected = rfe_selected[rfe_selected['RFE'] == True]
        l1_selected = l1[l1['L1'] == True]

        # Get best features for each method
        best_iv = iv_df[iv_df['IV'] > iv_cutoff]['feature'].tolist()
        best_rf = vi_rf.sort_values('RF', ascending=False)['index'].head(rf_top_n).tolist()
        best_et = vi_et.sort_values('ExtraTrees', ascending=False)['index'].head(et_top_n).tolist()
        best_chi = chi_scores.sort_values('Chi_Square', ascending=False)['index'].head(chi_top_n).tolist()
        best_rfe = rfe_selected['index'].tolist()
        best_l1 = l1_selected['index'].tolist()

        st.markdown(f"""
        **Best Features by Method:**
        - **IV (>{iv_cutoff}):** {best_iv}
        - **Random Forest (Top {rf_top_n}):** {best_rf}
        - **ExtraTrees (Top {et_top_n}):** {best_et}
        - **Chi2 (Top {chi_top_n}):** {best_chi}
        - **RFE:** {best_rfe}
        - **L1:** {best_l1}
        """)

        # Visualize feature importances for each method
        st.subheader("Feature Importance Visualization by Method")
        fig, axes = plt.subplots(2, 3, figsize=(18, 8))
        sns.barplot(data=iv_df.sort_values('IV', ascending=False), x='feature', y='IV', ax=axes[0,0])
        axes[0,0].set_title('IV')
        axes[0,0].set_xticklabels(axes[0,0].get_xticklabels(), rotation=45, ha='right')
        sns.barplot(data=vi_rf.sort_values('RF', ascending=False), x='index', y='RF', ax=axes[0,1])
        axes[0,1].set_title('Random Forest')
        axes[0,1].set_xticklabels(axes[0,1].get_xticklabels(), rotation=45, ha='right')
        sns.barplot(data=vi_et.sort_values('ExtraTrees', ascending=False), x='index', y='ExtraTrees', ax=axes[0,2])
        axes[0,2].set_title('ExtraTrees')
        axes[0,2].set_xticklabels(axes[0,2].get_xticklabels(), rotation=45, ha='right')
        sns.barplot(data=chi_scores.sort_values('Chi_Square', ascending=False), x='index', y='Chi_Square', ax=axes[1,0])
        axes[1,0].set_title('Chi2')
        axes[1,0].set_xticklabels(axes[1,0].get_xticklabels(), rotation=45, ha='right')
        axes[1,1].bar(rfe_selected['index'], rfe_selected['RFE'])
        axes[1,1].set_title('RFE (1=Selected)')
        axes[1,1].set_xticklabels(rfe_selected['index'], rotation=45, ha='right')
        axes[1,2].bar(l1_selected['index'], l1_selected['L1'])
        axes[1,2].set_title('L1 (1=Selected)')
        axes[1,2].set_xticklabels(l1_selected['index'], rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

        # --- Final Feature Selection by Voting Cutoff (before Train-Test Split) ---
        st.subheader("Final Feature Selection by Voting Cutoff")
        vote_cutoff = st.slider("Select minimum vote count for feature selection", min_value=1, max_value=int(final_features['final_score'].max()), value=4)
        final_selected = final_features[final_features['final_score'] >= vote_cutoff]
        st.write(f"Features selected with vote cutoff >= {vote_cutoff}:")
        st.write(final_selected[['feature', 'IV', 'RF', 'ExtraTrees', 'Chi_Square', 'RFE', 'L1', 'final_score']])
        selected_vars = final_selected['feature'].tolist()
        features = df[selected_vars]  # Use original df to avoid chained selection issues

        # --- Horizontal Bar Chart for Features selected with vote cutoff >= 4 ---
        st.subheader("Feature Importances (Selected Features, Vote Cutoff ≥ 4)")
        if not final_selected.empty:
            chart_data = final_selected[['feature', 'final_score']].sort_values('final_score', ascending=True)
            fig, ax = plt.subplots(figsize=(8, max(4, len(chart_data)*0.5)))
            bars = ax.barh(chart_data['feature'], chart_data['final_score'], color=plt.cm.viridis(chart_data['final_score']/chart_data['final_score'].max()))
            ax.set_xlabel('Final Score (Votes)')
            ax.set_ylabel('Feature')
            ax.set_title('Selected Feature Importances (Vote Cutoff ≥ 4)')
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, f'{int(width)}', va='center', color='black', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info('No features selected with the current vote cutoff.')

        st.subheader("Train-Test Split and Model Training")
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=42)

        model_choice = st.selectbox("Choose Classifier", ["Random Forest", "Logistic Regression", "Naive Bayes", "Neural Network"])
        if model_choice == "Random Forest":
            clf = RandomForestClassifier()
        elif model_choice == "Logistic Regression":
            clf = LogisticRegression()
        elif model_choice == "Naive Bayes":
            clf = GaussianNB()
        else:
            clf = MLPClassifier()

        clf.fit(X_train, y_train)
        pred_train = clf.predict(X_train)
        pred_test = clf.predict(X_test)

        acc_train = accuracy_score(y_train, pred_train)
        acc_test = accuracy_score(y_test, pred_test)

        fpr_train, tpr_train, _ = roc_curve(y_train, clf.predict_proba(X_train)[:, 1])
        auc_train = auc(fpr_train, tpr_train)

        fpr_test, tpr_test, _ = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
        auc_test = auc(fpr_test, tpr_test)

        st.write(f"Train Accuracy: {acc_train:.3f}, AUC: {auc_train:.3f}")
        st.write(f"Test Accuracy: {acc_test:.3f}, AUC: {auc_test:.3f}")

        # ROC Curve (Test)
        st.subheader("ROC Curve (Test)")
        fig, ax = plt.subplots()
        ax.plot(fpr_test, tpr_test, label=f"Test AUC = {auc_test:.2f}")
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve - Test")
        ax.legend()
        st.pyplot(fig)

        # KS Statistic and Decile Analysis
        st.subheader("KS Statistic and Decile Analysis")
        # Get predicted probabilities for test set
        y_score = clf.predict_proba(X_test)[:, 1]
        # Calculate deciles
        deciles = pd.qcut(pd.Series(y_score).rank(method='first'), 10, labels=False) + 1
        test_df = pd.DataFrame({
            'y_true': y_test.reset_index(drop=True),
            'y_score': y_score,
            'decile': deciles
        })
        # Aggregate by decile
        decile_table = test_df.groupby('decile').agg(
            total=('y_true', 'count'),
            events=('y_true', 'sum'),
            non_events=('y_true', lambda x: (1-x).sum())
        ).reset_index()
        decile_table['event_rate'] = decile_table['events'] / decile_table['total']
        decile_table['cum_events'] = decile_table['events'].cumsum()
        decile_table['cum_non_events'] = decile_table['non_events'].cumsum()
        decile_table['cum_event_rate'] = decile_table['cum_events'] / decile_table['events'].sum()
        decile_table['cum_non_event_rate'] = decile_table['cum_non_events'] / decile_table['non_events'].sum()
        decile_table['ks'] = np.abs(decile_table['cum_event_rate'] - decile_table['cum_non_event_rate'])
        st.write(decile_table)
        st.write(f"Max KS Statistic: {decile_table['ks'].max():.3f}")
        # Plot KS curve
        fig, ax = plt.subplots()
        ax.plot(decile_table['decile'], decile_table['cum_event_rate'], label='Cumulative Event Rate')
        ax.plot(decile_table['decile'], decile_table['cum_non_event_rate'], label='Cumulative Non-Event Rate')
        ax.plot(decile_table['decile'], decile_table['ks'], label='KS Statistic')
        ax.set_xlabel('Decile')
        ax.set_ylabel('Cumulative Rate')
        ax.set_title('KS Statistic and Decile Analysis')
        ax.legend()
        st.pyplot(fig)

        # Show confusion matrix counts for each model
        st.subheader("Confusion Matrix (Test Set)")
        conf_matrix = pd.crosstab(y_test, pred_test, rownames=['ACTUAL'], colnames=['PRED'])
        st.write(conf_matrix)

        st.subheader("Save Trained Model")
        if st.button("Save Model"):
            joblib.dump([d, clf, selected_vars], 'final_model.model')
            st.success("Model saved as 'final_model.model'")

        # --- Feature Importance Bar Chart Based on Model ---
        st.subheader("Feature Importances from Trained Model")
        if model_choice == "Random Forest":
            importances = clf.feature_importances_
            feature_names = X_train.columns
        elif model_choice == "ExtraTrees":
            importances = clf.feature_importances_
            feature_names = X_train.columns
        elif model_choice == "Logistic Regression":
            importances = np.abs(clf.coef_[0])
            feature_names = X_train.columns
        elif model_choice == "Naive Bayes":
            importances = np.abs(clf.theta_[0])
            feature_names = X_train.columns
        elif model_choice == "Neural Network":
            importances = np.abs(clf.coefs_[0]).sum(axis=1)
            feature_names = X_train.columns
        else:
            importances = None
            feature_names = None
        if importances is not None:
            imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            imp_df = imp_df.sort_values('Importance', ascending=True)
            fig, ax = plt.subplots(figsize=(8, max(4, len(imp_df)*0.5)))
            bars = ax.barh(imp_df['Feature'], imp_df['Importance'], color=plt.cm.viridis(imp_df['Importance']/imp_df['Importance'].max()))
            ax.set_xlabel('Importance')
            ax.set_ylabel('Feature')
            ax.set_title(f'Feature Importances ({model_choice})')
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.01*imp_df['Importance'].max(), bar.get_y() + bar.get_height()/2, f'{width:.3f}', va='center', color='black', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info('Feature importances not available for this model.')

if __name__ == '__main__':
    main()
