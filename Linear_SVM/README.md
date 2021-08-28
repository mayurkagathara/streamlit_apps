# Linear SVM

Linear SVM or support vector machine is used in classification and regression both. We will implement support vector classifier.
We will be using the squared hinge loss and L2 regularization.

## Formulation

- Objective Function (LOSS) L = $`\LARGE \underset{w,b}{min} \;\; \frac{1}{n}\sum_{i=0}^{n} [max(0, 1-z)]^2 + \lambda ||w||^2 \\ \text{where } z = y_i(w^T \cdot x + b)`$
  
- Derrivatives
  - w.r.t. w  
  $\LARGE \frac{\partial L}{\partial w} = \frac{1}{n}\sum_{i=0}^{n} (-2)\cdot max(0, 1-z) \cdot y_ixi + 2\cdot \lambda\cdot w$
  - w.r.t. b  
  $\LARGE \frac{\partial L}{\partial b} = \frac{1}{n}\sum_{i=0}^{n} (-2)\cdot max(0, 1-z) \cdot y_i$
  
- Update  
  $\large w_{new} = w_{old} + \eta \; \frac{\partial L}{\partial w} \\ b_{new} = b_{old} + \eta \; \frac{\partial L}{\partial b}$
  
- Python file simulation: [Linear_SVM.py](https://github.com/mayurkagathara/streamlit_apps/blob/main/Linear_SVM/Linear_SVM.py)
- Jupyter notebook: [linear_svm.ipynb](https://github.com/mayurkagathara/streamlit_apps/blob/main/Linear_SVM/notebooks/linear_svm.ipynb)
- simulation video: [Linear SVM simulation](https://www.youtube.com/watch?v=0OLhfoOXou8)

[![Linear SVM simulation](../static/SVM_simulation.gif)](https://www.youtube.com/watch?v=0OLhfoOXou8)
