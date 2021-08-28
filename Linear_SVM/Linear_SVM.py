import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.datasets import make_classification

def get_loss_gradient(x, y_actual, y_pred, w, b, C):
  """
  calculate gradient with respect to w and intercept b.
  return loss, gradient value wrt w and b
  """
  Lw = -x*y_actual[:,None]
  Lb = -y_actual

  z = y_actual*y_pred
  Lw[z >= 1] = 0
  Lb[z >= 1] = 0
  
  dL_by_dw = w + C*np.sum(Lw, axis=0)
  dL_by_db = b + C*np.sum(Lb, axis=0)

  total_loss_2 = 1 - z
  total_loss_2[z >= 1] = 0
  total_loss = 0.5*(w.T.dot(w)) + C*np.sum(total_loss_2)
  
  return total_loss, dL_by_dw, dL_by_db

def solve_linear_SVM(x, y, C=1, eta=0.01, epochs=10000, w_start=1, b_start=0):
  '''
  takes input x, output y and learning rate eta as input.
  gradient descent update is stopped when update is less than tolerance.
  gives best m and c.
  '''

  # Intialize slope and intercept
  if x.ndim == 1:
    x = x.reshape(x.shape[0],1)
    w = np.array([w_start])
  else:
    w = np.zeros(x.shape[-1]) + w_start
  b = b_start
  errors = [np.inf]
  w_array = [w]
  b_array = [b]
  continue_loop = True
  epoch = 0

  for _ in range(epochs):
    y_pred = w.dot(x.T) + b   #positive value for category 1 and negative for -1
    L, dw, db = get_loss_gradient(x, y, y_pred, w, b, C)

    w = w - (eta*dw)
    b = b - (eta*db)

    w_array.append(w)
    b_array.append(b)
    errors.append(L)

    epoch += 1
    # continue_loop = (errors[epoch-1] - errors[epoch]) > tolerance

  metadata = np.array([errors[1:], w_array[:-1], b_array[:-1]])
  return metadata.T, w, b

def plot_line_wb(weight, bias, x_vals, epoch, loss, color='b'):
  # global fig, ax1, ax2, ax3
  """Plot a line from weight and bias"""
  # axes = plt.gca()
  # x_vals = np.array(ax1.get_xlim())

  ########### ax2 drawing ################
  y_vals = bias + weight * x_vals

  title = f'epoch = {epoch}'
  ax2.set_title(title)

  data = f'w(slope) = {round(weight,4)}\n b(bias) = {round(bias,4)}'
  text = ax2.text(5,5,data,color='b')

  line, = ax2.plot(x_vals, y_vals,'--', color=color)
  # print(line)

  ########### ax3 drawing ################
  ax3.plot(epoch,loss,'-ro')
  ax3.set_title(f'loss: {round(loss,5)}')

  ########## Figure drawing ##############
  fig.canvas.draw()
  fig.canvas.flush_events()
  plt.pause(0.00000001)
  line.remove()
  text.remove()

def simulate_linreg(x,y,metadata_lmc):
  loss = list(metadata_lmc[:,0])
  weight_array = list(map(lambda x: -x[0]/x[1], metadata_lmc[:,1]))
  bias_array = list(map(lambda x: -x[1]/x[0][1], metadata_lmc[:,1:]))
  epochs = iter(list(range(len(loss))))
  loss_iter = iter(loss)

  ax1.set_ylim(ymin=np.min(x)-2, ymax=np.max(x)+2)
  sns.pointplot(x=x.T[0,:], y=x.T[1,:], hue=y, join=False, ax=ax1)
  
  x_vals = np.array(ax1.get_xlim())
  # print(ax1.get_xlim())
  for w,b in zip(weight_array,bias_array):
    plot_line_wb(w,b, x_vals, next(epochs), next(loss_iter))
    # time.sleep(0.0001)

def get_figure():
  global fig, ax1, ax2, ax3
  fig = plt.figure(figsize=(12,8))
  ax1 = fig.add_subplot(121)
  ax2 = fig.add_subplot(121)
  ax3 = fig.add_subplot(122)
  ax3.set_ylim(ymin=0, ymax=5)
  ax3.set_xlim(xmin=0, xmax=300)

if __name__=='__main__':
  # x = np.array([[1, 12], [1.5, 11],
  #               [2, 13], [2.5, 12],
  #               [3, 15], [3.5, 14],
  #               [4, 13], [4.5, 2],
  #               [3, 6], [1.5, 14],
  #               [4, 7], [2.5, 3.5],
  #               [5, 9], [3.5, 6],
  #               [6, 7], [4.5, 3]]) 
  # y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])

  x,y = make_classification(n_samples=40, n_features=2, n_informative=2, n_redundant=0, n_classes=2, n_clusters_per_class=2, class_sep=3, random_state=1995)
  
  print(x.shape, y.shape)
  assert x.shape[0]==y.shape[0]

  metadata_lmc, coef_, intercept_ = solve_linear_SVM(x, y, C=0.1, w_start=1, b_start=0, eta=0.001 ,epochs=2000)
  print(f'coef_ = {coef_} \nintercept_term={intercept_}')
  slope = -(coef_[0]/coef_[1])  # slope = -w1/w2
  intercept = -(intercept_/coef_[1]) # b = -w0/w2
  print(f'slope = {slope} \nintercept={intercept}')

  print(x.shape, y.shape)
  assert x.shape[0]==y.shape[0]

  if x.shape[1] == 2:
    get_figure()
    print(f'total iterations = {metadata_lmc.shape[0]}')
    simulate_linreg(x,y,metadata_lmc)
    input('Enjoyed the show?')