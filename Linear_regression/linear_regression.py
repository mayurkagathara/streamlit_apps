import numpy as np
import matplotlib.pyplot as plt
import time

def get_loss_gradient(x, y_actual, y_pred):
  """
  Take data point(or array), actual, and predicted output values as input.
  calculate gradient with respect to slope m and intercept c.
  return gradient value wrt m and c
  """
  if y_actual.shape != y_pred.shape:
    raise ValueError(f'y_actual{y_actual.shape} and y_pred{y_pred.shape} shape not matching')

  n = len(y_actual)
  error = y_actual-y_pred
  # print(error)
  total_error = np.sum(error**2)
  # print(error.dot(x))

  dL_by_dm = (-2/n) * error.dot(x)
  dL_by_dc = (-2/n) * np.sum(error)
  
  return total_error, dL_by_dm, dL_by_dc

def solve_linear_regression(x, y, eta=0.01, tolerance=1e-6, w_start=0, b_start=0):
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

  while continue_loop:
    y_pred = (x.dot(w.T)) + b

    L, dw, db = get_loss_gradient(x, y, y_pred)

    w = w - (eta*dw)
    b = b - (eta*db)

    w_array.append(w)
    b_array.append(b)
    errors.append(L)

    epoch += 1
    continue_loop = (errors[epoch-1] - errors[epoch]) > tolerance

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
  plt.pause(0.00001)
  line.remove()
  text.remove()

def simulate_linreg(x,y,metadata_lmc):
  loss, weights_array, bias_array = metadata_lmc.T
  epochs = iter(list(range(len(loss))))
  loss_iter = iter(loss)
  ax1.plot(x,y,'o')
  x_vals = np.array(ax1.get_xlim())
  # print(ax1.get_xlim())
  for w,b in zip(weights_array,bias_array):
    plot_line_wb(w[0],b, x_vals, next(epochs), next(loss_iter))
    time.sleep(0.01)

def get_figure():
  global fig, ax1, ax2, ax3
  fig = plt.figure(figsize=(12,8))
  ax1 = fig.add_subplot(121)
  ax2 = fig.add_subplot(121)
  ax3 = fig.add_subplot(122)
  ax3.set_ylim(ymin=0, ymax=1000)
  ax3.set_xlim(xmin=0, xmax=100)

if __name__=='__main__':
  # x = np.array([[1, 2],
  #             [2, 3],
  #             [3, 4],
  #             [4, 5],
  #             [5, 6],
  #             [6, 7],
  #             [7, 8],
  #             [8, 9]]) # 3D Data

  x = np.array([2, 3, 4, 5, 6, 7, 8, 9]).reshape(8,1)   #2d example
  y = np.array([4, 6, 9, 9, 13, 13, 17, 19])
  assert x.shape[0]==y.shape[0]

  metadata_lmc, slope, intercept = solve_linear_regression(x, y, w_start=0, b_start=-2, eta=1e-3 ,tolerance=1e-2)
  
  print(f'slope = {slope} \nintercept={intercept}')
  
  if x.shape[1] == 1:
    get_figure()
    print(f'total iterations = {metadata_lmc.shape[0]}')
    simulate_linreg(x,y,metadata_lmc)
    input('Enjoyed the show?')
