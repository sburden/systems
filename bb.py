
import sys

from systems import Hybrid

import numpy as np
import pylab as plt

import matplotlib 
from matplotlib import rc
#
# Say, "the default sans-serif font is Helvetica"
rc('font',**{'sans-serif':'Helvetica','family':'sans-serif','size':12})
rc('text',usetex=False)
# limit number of decimals printed in arrays
np.set_printoptions(precision=4)

class BouncingBall(Hybrid):

  def __init__(self):
    """
    bouncing ball hybrid system

    parameters:
      m - scalar - mass of ball
      g - scalar - gravitational constant
      c - scalar - coefficient of restitution
    """
    Hybrid.__init__(self)

  def F(self, (k,t), (j,x), **p):
    if j == 1:
      h,dh = x
      dx = np.array([dh, -p['g']])

    return dx  

  def G(self, (k,t), (j,x), **p):
    if j == 1:
      g = x[0]

    else:
      raise RuntimeError,'G -- unknown discrete mode'

    return g

  def R(self, (k,t), (j,x), **p):
    if j == 1:
      h,dh = x

      k_ = k+1
      t_ = t
      j_ = j
      x_ = np.array([h+0.1, -p['c']*dh])

    return (k_,t_),(j_,x_)

  def DxF(self, (k,t), (j,x), **p):
    if j == 1:
      DxF = np.array([[0.,1.], [0.,0.]])
    return DxF

  def DxG(self, (k,t), (j,x), **p):
    if j == 1:
      DxG = np.array([1.,0.])
    return DxG

  def DxR(self, (k,t), (j,x), **p):
    if j == 1:
      DxR = np.array([[1.,0.], [0.,-p['c']]])
    return DxR

  def O(self, (k,t), (j,x), **p):
    if j == 1:
      # height, velocity
      h,dh = x
      # kinetic energy, potential energy
      KE = 0.5 * p['m'] * dh**2
      PE = p['m'] * p['g'] * h
      #
      o = dict(h=h,dh=dh,KE=KE,PE=PE)
      if 'costate' in p:
        o['p'] = p['costate']
      if 'variation' in p:
        o['P'] = p['variation']

    return o

  def O_(self, (k,t), (j,x), **p):
    if j == 1:
      # height, velocity
      h,dh = x
      # kinetic energy, potential energy
      KE = 0.5 * p['m'] * dh**2
      PE = p['m'] * p['g'] * h
      #
      o = [h,dh,KE,PE]


    return o

if __name__ == "__main__":

  K = 2
  T = .75
  T = .5
  T = .4

  j = 1
  x = np.array([1., 0.])
  m = 1.
  g = 10.
  c = 1.
  c = 0.66
  #c = 0
  debug = True
  debug = False
  Zeno = True
  Zeno = False

  p = dict(m=m, g=g, c=c, debug=debug, Zeno=Zeno)

  hs = BouncingBall()

  dt = 1e-6
  rx = 1e-12

  T = 2.5*dt
  x = np.asarray([2*dt,-1])

  trjs = hs.sim((K,T), (j,x), dt, rx, **p);
  trjs = hs.cosim(trjs, np.identity(len(x)), **p)
  trjs = hs.varsim(trjs, np.identity(len(x)), **p)

  import util

  flow_T = lambda x,j=j : hs.sim((K,T), (j,x), dt, rx, **p)[-1]['x'][-1]

  (k,t),(j,o) = hs.obs(trjs, insert_nans=False, **p)
  (kd,td),(jd,od) = hs.obs(trjs, only_disc=True, **p)

  # check stepsize for derivative
  plt.figure(1); plt.clf()
  d = np.logspace(-10,0,11)
  D = np.asarray([util.D(flow_T,x,d=_).flatten() for _ in d])
  plt.loglog(d,np.abs(D))

  # choose auspicious stepsize
  d = dt
  dx = np.random.randn(2); dx /= np.sqrt((dx**2).sum())

  dx = np.array([1,0])
  dx = np.array([0,1])
  print dx
  print (flow_T(x) - flow_T(x+dx*d))/d
  print np.dot(dx,trjs[0]['p'][0].T)

  #print 'dx =',dx 
  Dflow = util.D(flow_T,x,d=d)
  p0 = o['p'][0]
  PT = o['P'][-1]
  print 'Dflow =\n',Dflow
  print 'p[0] =\n',p0
  print 'P[T] =\n',PT
  print 'diff = %0.2e'%np.abs(Dflow - p0).max()
  print 'diff = %0.2e'%np.abs(Dflow - PT).max()
  sys.exit(0)


  # check analytical derivatives
  #print util.D(lambda x : hs.F((0,0.),(1,x),**p), dx, d=d)
  #print hs.DxF((0,0.),(1,x),**p)
  #print util.D(lambda x : hs.G((0,0.),(1,x),**p), dx, d=d)
  #print hs.DxG((0,0.),(1,x),**p)
  #print util.D(lambda x : hs.R((0,0.),(1,x),**p)[1][1], dx, d=d)
  #print hs.DxR((0,0.),(1,x),**p)
  assert np.allclose(util.D(lambda x : hs.F((0,0.),(1,x),**p), dx, d=d) , hs.DxF((0,0.),(1,x),**p))
  assert np.allclose(util.D(lambda x : hs.G((0,0.),(1,x),**p), dx, d=d) , hs.DxG((0,0.),(1,x),**p))
  assert np.allclose(util.D(lambda x : hs.R((0,0.),(1,x),**p)[1][1], dx, d=d) , hs.DxR((0,0.),(1,x),**p))

  #import sys
  #sys.exit(0)

  lw = 2
  mew = lw
  ms = 10
  lt = '-'
  fs = (8,8)

  plt.figure(1,figsize=fs); plt.clf()

  # continuous transitions
  ax = plt.subplot(1,1,1); plt.grid('on')
  ax.plot(o['dh'],o['h'],'k.-',lw=lw,ms=ms)
  # discrete transitions
  ax.plot(od['dh'][::2], od['h'][::2], 'o',ms=ms,mew=mew,mec='g',mfc='None')
  ax.plot(od['dh'][1::2],od['h'][1::2],'o',ms=ms,mew=mew,mec='r',mfc='None')
  # dashed line at discrete transitions
  ax.plot(np.vstack(( od['dh'][1:-1:2],od['dh'][2::2])), 
          np.vstack(( od['h'][1:-1:2], od['h'][2::2])), 'k--',lw=lw)
  ax.set_xlabel(r'$\dot{h}$')
  ax.set_ylabel(r'$h$')

  plt.figure(2,figsize=fs); plt.clf()

  ax = plt.subplot(3,1,1); ax.grid('on')
  # continuous transitions
  ax.plot(t,o['h'],'k'+lt,lw=lw,ms=ms)
  ax.plot(t,o['p'][:,0,0],'b'+lt,lw=lw,ms=ms)
  # discrete transitions -- height is continuous, so these don't appear
  #ax.plot(np.vstack(( td[1:-1:2], td[2::2])), 
  #         np.vstack(( hd[1:-1:2], hd[2::2])), 'k--',lw=lw)
  ax.set_ylabel(r'$h$')
  ax.set_xticklabels([])

  ax = plt.subplot(3,1,2); ax.grid('on')
  # continuous transitions
  ax.plot(t,o['dh'],'k'+lt,lw=lw,ms=ms)
  ax.plot(t,o['p'][:,1,1],'b'+lt,lw=lw,ms=ms)
  # dashed line at discrete transitions
  ax.plot(np.vstack(( td[1:-1:2], td[2::2])), 
           np.vstack((od['dh'][1:-1:2],od['dh'][2::2])), 'k--',lw=lw)
  ax.set_ylabel(r'$\dot{h}$')
  ax.set_xticklabels([])

  ax = plt.subplot(3,1,3); ax.grid('on')
  # continuous transitions
  ax.plot(t,o['KE'],'b'+lt,lw=lw,ms=ms)
  ax.plot(t,o['PE'],'g'+lt,lw=lw,ms=ms)
  # dashed line at discrete transitions -- KE is discontinuous when c != 1
  ax.plot(np.vstack(( td[1:-1:2], td[2::2])), 
           np.vstack((od['KE'][1:-1:2],od['KE'][2::2])), 'b--',lw=lw)
  ax.legend(('KE','PE'))
  ax.set_ylabel(r'energy')
  ax.set_xlabel('time (sec)')

