import torch
import numpy as np
from scipy.integrate import solve_ivp
from numpy.linalg import inv


def get_time_grid(t,T,size):
    if t == T:
        return np.array([T])
    #  Ensures that T is included in the grid   
    grid = np.arange(t, T+0.00001, np.divide(T-t, np.divide(T-t, size)))
    return grid


class LQR:
    # 1)
    def __init__(self, H, M, C, D, R, t = 0, T = 1, n=100):          
      self.H = H
      self.M = M
      self.C = C
      self.D = D
      self.R = R
      self.t = t
      self.T = T
      self.number_of_time_steps = n
      self.tss = np.divide(T - t, n)
      self.D_inv = inv(self.D)


    # 2)
    def ricatti_ode(self,t, Q, H, M, C, D):
      # using Q' = S'(T-t)
      Q = Q.reshape(2, 2)
      dq = -2* H.T @ Q + Q @ M @ np.linalg.inv(D) @ M.T @ Q - C
      return dq.flatten()


    def solve_ricatti_ode(self, time_grid):
      if len(time_grid) == 1:
        return [self.R]
      res = solve_ivp(self.ricatti_ode, (time_grid[0], time_grid[-1]), self.R.flatten() , t_eval=time_grid, args=(self.H, self.M, self.C, self.D))
      return res.y.T.reshape(-1, 2, 2)[::-1]


    #3)
    def control_problem_value(self, time_tensor, space_tensor, sigma):
      v = []
      for t, x in zip(time_tensor, space_tensor):
        time_grid = get_time_grid(t,self.T, self.tss)
        integral = 0
        S = self.solve_ricatti_ode(time_grid) 
        for s_t in S:
          integral += np.trace(sigma @ sigma.transpose() * s_t) * self.tss
        values = x @ S[0] @ torch.t(x) + integral
        v.append(values.flatten()[0])
      return torch.tensor(v)
    

    #4）
    def markov_control(self, time_tensor, space_tensor):
      a = np.zeros(shape=(1, 2))
      for t, x in zip(time_tensor, space_tensor):
        time_grid = get_time_grid(t,self.T, self.tss)
        S = self.solve_ricatti_ode(time_grid)
        control = (-self.D_inv @ self.M.T @ S[0] @ x.numpy().T).T
        a = np.vstack((a, control))
      return torch.tensor(a[1:])