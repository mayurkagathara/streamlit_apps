import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.datasets import make_classification

def get_loss_gradient(x, y_actual, y_pred, w, b, alpha):
  """
  calculate gradient with respect to w and intercept b.
  return loss, gradient value wrt w and b
  """
  N = len(y)
  z = y*y_pred

  dL_by_dw = -2/N*(np.maximum(0, 1-z).dot(
        y[:, None]*x)) + 2 * alpha * w

  lb = (-2/N)*(y_actual*(1-z))
  lb[z >=1 ] = 0
  dL_by_db = np.sum(lb)

  total_loss = (1/N) * (np.sum( (np.maximum(0, 1-z)**2) ))+ alpha * np.linalg.norm(w)**2
  return total_loss, dL_by_dw, dL_by_db

def solve_linear_SVM(x, y, alpha=0.3, eta=0.01, epochs=10000, w_start=1, b_start=0, tolerance=1e-5):
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

  while epoch<epochs and continue_loop:
    y_pred = w.dot(x.T) + b   #positive value for category 1 and negative for -1
    L, dw, db = get_loss_gradient(x, y, y_pred, w, b, alpha)

    w = w - (eta*dw)
    b = b - (eta*db)

    w_array.append(w)
    b_array.append(b)
    errors.append(L)

    epoch += 1
    if errors[epoch-1] > errors[epoch]:
      continue_loop = (errors[epoch-1] - errors[epoch]) > tolerance

  metadata = np.array([errors[1:], w_array[:-1], b_array[:-1]])
  return metadata.T, w, b

def plot_line_wb(weight, bias, margin, x_vals, epoch, loss, color='b'):
  # global fig, ax1, ax2, ax3
  """Plot a line from weight and bias"""
  # axes = plt.gca()
  # x_vals = np.array(ax1.get_xlim())

  ########### ax2 drawing ################
  y_vals = bias + weight * x_vals
  y_vals_margin_negative = y_vals - 1
  y_vals_margin_positive = y_vals + 1

  title = f'epoch = {epoch}'
  ax2.set_title(title)

  data = f'w(slope) = {round(weight,4)}\n b(bias) = {round(bias,4)}\n margin(2/w) = {round(margin,4)}'
  text = ax2.text(5,5,data,color='b')

  line1, = ax2.plot(x_vals, y_vals,'--', color=color)
  line2, = ax2.plot(x_vals, y_vals_margin_negative,'--', color='g')
  line3, = ax2.plot(x_vals, y_vals_margin_positive,'--', color='g')

  # print(line)

  ########### ax3 drawing ################
  ax3.plot(epoch,loss,'-ro')
  ax3.set_title(f'loss: {round(loss,5)}')

  ########## Figure drawing ##############
  fig.canvas.draw()
  fig.canvas.flush_events()
  plt.pause(0.00000001)
  line1.remove()
  line2.remove()
  line3.remove()
  text.remove()

def simulate_linreg(x,y,metadata_lmc):
  loss = list(metadata_lmc[:,0])
  weight_array = list(map(lambda x: -x[0]/x[1], metadata_lmc[:,1]))
  bias_array = list(map(lambda x: -x[1]/x[0][1], metadata_lmc[:,1:]))
  margin_array = iter(list(map(lambda w: 2/np.linalg.norm(w), metadata_lmc[:,1])))
  epochs = iter(list(range(len(loss))))
  loss_iter = iter(loss)
  
  ax3.set_ylim(ymin=0, ymax=max(loss))
  ax3.set_xlim(xmin=0, xmax=len(loss))

  ax1.set_ylim(ymin=np.min(x)-2, ymax=np.max(x)+2)
  sns.pointplot(x=x.T[0,:], y=x.T[1,:], hue=y, join=False, ax=ax1)
  
  x_vals = np.array(ax1.get_xlim())
  # print(ax1.get_xlim())
  for w,b in zip(weight_array,bias_array):
    loss_i = next(loss_iter)
    epochs_i = next(epochs)+1
    margin = next(margin_array)
    plot_line_wb(w, b, margin, x_vals, epochs_i, loss_i)
    # time.sleep(0.0001)

def get_figure():
  global fig, ax1, ax2, ax3
  fig = plt.figure(figsize=(12,8))
  ax1 = fig.add_subplot(121)
  ax2 = fig.add_subplot(121)
  ax3 = fig.add_subplot(122)
  # ax3.set_ylim(ymin=0, ymax=5)
  # ax3.set_xlim(xmin=0, xmax=25)

if __name__=='__main__':
  # x = np.array([[1, 12], [1.5, 11],
  #               [2, 13], [2.5, 12],
  #               [3, 15], [3.5, 14],
  #               [4, 13], [4.5, 2],
  #               [3, 6], [1.5, 14],
  #               [4, 7], [2.5, 3.5],
  #               [5, 9], [3.5, 6],
  #               [6, 7], [4.5, 3]]) 
  # y = np.array([1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1])

  x,y = make_classification(n_samples=40, n_features=2, n_informative=2, n_redundant=0, n_classes=2, n_clusters_per_class=2, class_sep=3, random_state=1995)
  y[y==0] = -1
  print(x.shape, y.shape)
  assert x.shape[0]==y.shape[0]

  metadata_lmc, coef_, intercept_ = solve_linear_SVM(x, y, alpha=8, w_start=0.1, b_start=0.1, eta=0.01 ,epochs=10000, tolerance=1e-4)
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