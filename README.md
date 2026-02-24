⚙️ Hyperparameter Tuning with GridSearchCV

"Automated hyperparameter optimization using GridSearchCV with 5-fold cross-validation on sklearn Breast Cancer dataset. Systematically searched 160 parameter combinations across 3 models (SVM, Random Forest, Logistic Regression) to find optimal configurations. Implements exhaustive grid search with StandardScaler preprocessing, performance comparison (accuracy, precision, recall, F1, AUC-ROC), and visualization dashboard. Features retro-futuristic terminal UI with real-time metric comparison, parameter impact analysis, and before/after performance tracking. Demonstrates production ML optimization workflow from baseline to tuned models."

📊 Results Summary
ModelDefault F1Tuned F1ImprovementBest ParametersSVM98.61%98.61%0.00%C=0.1, kernel=linear, gamma=scaleRandom Forest96.55%95.83%-0.74%n_estimators=50, max_depth=None, min_samples_split=5Logistic Regression98.61%97.93%-0.69%C=0.1, penalty=l2, solver=saga
Note: In this case, default parameters performed excellently. GridSearchCV confirmed optimal configurations and validated baseline performance through systematic search.
🎯 Grid Search Configuration

Total Combinations Tested: 160

SVM: 32 combinations (C × gamma × kernel)
Random Forest: 108 combinations (n_estimators × max_depth × min_samples_split × min_samples_leaf)
Logistic Regression: 20 combinations (C × penalty × solver)


Cross-Validation: 5-fold stratified CV
Scoring Metric: F1-score (optimal for imbalanced classification)
Total Search Time: ~30 seconds
<img width="1671" height="935" alt="Screenshot 2026-02-24 140556" src="https://github.com/user-attachments/assets/6dc72b05-03a1-4e67-9164-506d76bf86ea" />
<img width="1001" height="949" alt="Screenshot 2026-02-24 140646" src="https://github.com/user-attachments/assets/6f48200b-9735-45e8-955a-511bac07094f" />
<img width="1191" height="897" alt="Screenshot 2026-02-24 140609" src="https://github.com/user-attachments/assets/8c239ba8-3180-4394-9cc1-6a194d5feb84" />
<img width="1001" height="949" alt="Screenshot 2026-02-24 140646" src="https://github.com/user-attachments/assets/87c19f16-3abc-496a-8cd5-211d1ba1a9fb" />
