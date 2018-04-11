
import sys

from systems import Mechanical

import util

import numpy as np
import pylab as plt

import matplotlib 
from matplotlib import animation, rc
#
# Say, "the default sans-serif font is Helvetica"
rc('font',**{'sans-serif':'Helvetica','family':'sans-serif','size':12})
rc('text',usetex=False)

class PendulumWall(Mechanical):

  def __init__(self):
    """
    pendulum with wall hybrid system

    pw = PendulumWall()

    parameters:
      m - scalar - mass at end of rod
      l - scalar - length of rod
      g - scalar - gravitational constant
      c - scalar - coefficient of restitution
    """
    Mechanical.__init__(self)

  def M(self, (k,t), (J,q,dq), **p):
    d = len(q)
    return p['m'] * np.identity(d)

  def c(self, (k,t), (J,q,dq), **p):
    d = len(q)
    return np.zeros((d,d))

  def f(self, (k,t), (J,q,dq), **p):
    return np.asarray([0.,-p['m']*p['g']])

  def a(self, (k,t), (J,q), **p):
    x,y = q
    if J[0]:
      #return np.asarray([-y])
      return np.asarray([x+0.8*p['l']])#+.1*np.sin(2*y)])
    else:
      return np.asarray([])

  def gamma(self, (k,t), (J,q,dq), **params):
    return p['c']

  def b(self, (k,t), (J,q), **p):
    x,y = q
    return np.asarray([p['l'] - np.sqrt( np.sum( q**2 ) )])

  def O(self, (k,t), (j,x), **params):
    x,y,dx,dy = x
    o = dict(x=x,y=y,dx=dx,dy=dy)
    if 'costate' in params:
      o['p'] = params['costate']
    return o


if __name__ == "__main__":

  #print 

  K = 5
  #T = 2e-2
  T = 10.
  j = np.asarray([0])
  #j = np.asarray([])
  m = 1.
  l = 1.
  g = 10.
  c = 1.
  #c = 2.
  #c = 0.66
  c = 0
  q,dq = [l,0.],[0.,0.]
  q,dq = [l,0.],[0.,-1.]
  #q,dq = [l/np.sqrt(2),l/np.sqrt(2)],[0.,0.]
  x = np.hstack((q,dq))
  #x = np.asarray([-0.77358726, -0.63369775, -2.34328972,  2.86054208])
  debug = True
  debug = False
  Zeno = True
  Zeno = False

  p = dict(m=m, l=l, g=g, c=c, debug=debug, Zeno=Zeno)

  hs = PendulumWall()

  dt = 1e-2
  rx = 1e-12

  trjs = hs.sim((K,T), (j,x), dt, rx, **p)
  trjs = hs.cosim(trjs, np.identity(len(x)), **p)

  #sys.exit(0)

  import util

  flow_T = lambda x,j=j : hs.sim((K,T), (j,x), dt, rx, **p)[-1]['x'][-1]

  (k,t),(j,o) = hs.obs(trjs, insert_nans=True, **p)
  (kd,td),(jd,od) = hs.obs(trjs, only_disc=True, **p)

  #sys.exit(0)

  # check stepsize for derivative
  #plt.figure(1); plt.clf()
  #d = np.logspace(-10,0,11)
  #D = np.asarray([util.D(flow_T,x,d=_).flatten() for _ in d])
  #plt.loglog(d,np.abs(D))

  # choose auspicious stepsize
  #d = dt 
  ##d = 1e-3
  #p0 = o['p'][0]
  #Dflow = util.D(flow_T,x,d=d)
  #print 'p[0] =\n',p0
  #print 'Dflow =\n',Dflow
  #print 'diff = %0.2e'%np.abs(Dflow - p0).max()
  #sys.exit(0)

  lw = 2
  mew = lw
  ms = 10
  lt = '-'
  fs = (6,6)

  if 0:

    fig = plt.figure(1,figsize=fs); plt.clf()
    ax = plt.subplot(1,1,1); plt.grid('on'); plt.axis('equal')

    rod, = ax.plot([],[],'k.-',lw=2.,ms=10.)
    mass, = ax.plot([],[],'bo',lw=2.,ms=20.)

    def init():
      i = 0
      xlim = (-1.1*l,+1.1*l)
      ylim = (-1.1*l,+1.1*l)
      # fill guard
      d = l/100.
      x = np.arange(xlim[0],xlim[1],d)
      y = np.arange(ylim[0],ylim[1],d)
      X,Y = np.meshgrid(x, y)
      Z = [hs.a((0,0.),([1],q),**p) for q in zip(X.flatten(),Y.flatten())]
      Z = 1*(np.asarray(Z).reshape((y.size,x.size)) >= 0)
      ax.contourf(X,Y,Z,levels=[-1,0],colors=[.25*np.ones(3),np.ones(3)])
      ax.contour(X,Y,Z,levels=[0],colors=[0.*np.ones(3)],linewidths=4)
      #
      ax.set_xlim(xlim)
      ax.set_ylim(ylim)
      ax.set_xlabel(r'$x$')
      ax.set_ylabel(r'$y$')

    def animate(_t):
      i = (t >= _t).nonzero()[0][0]
      rod.set_data([0.,o['x'][i]],[0.,o['y'][i]])
      mass.set_data([o['x'][i]],[o['y'][i]])
      return (rod,mass,)

    fps = 30
    dt = 1./fps
    msec = int(1000*dt)
    ani = animation.FuncAnimation(fig, animate, np.arange(0.,t[-1],dt), init_func=init, 
                                  interval=msec, blit=True, repeat=False)

    plt.show()

  #sys.exit(0)

  plt.figure(1,figsize=fs); plt.clf()

  # continuous transitions
  ax = plt.subplot(1,1,1); plt.grid('on'); plt.axis('equal')
  ax.plot(o['x'],o['y'],'k.-',lw=lw,ms=ms)
  # discrete transitions
  ax.plot(od['x'][::2], od['y'][::2], 'o',ms=ms,mew=mew,mec='g',mfc='None')
  ax.plot(od['x'][1::2],od['y'][1::2],'o',ms=ms,mew=mew,mec='r',mfc='None')
  # dashed line at discrete transitions
  #ax.plot(np.vstack((dhd[1:-1:2],dhd[2::2])), 
  #         np.vstack(( hd[1:-1:2], hd[2::2])), 'k--',lw=lw)
  ax.set_xlabel(r'$x$')
  ax.set_ylabel(r'$y$')
  ax.set_xlim((-1.1*l,+1.1*l))
  ax.set_ylim((-1.1*l,+1.1*l))

  plt.figure(2,figsize=fs); plt.clf()

  ax = plt.subplot(2,1,1); ax.grid('on')
  # continuous transitions
  ax.plot(t,o['x'],'r'+lt,lw=lw,ms=ms)
  ax.plot(t,o['y'],'b'+lt,lw=lw,ms=ms)
  # discrete transitions -- height is continuous, so these don't appear
  #ax.plot(np.vstack(( td[1:-1:2], td[2::2])), 
  #         np.vstack(( hd[1:-1:2], hd[2::2])), 'k--',lw=lw)
  ax.set_ylabel(r'$x, y$')
  ax.set_xticklabels([])

  ax = plt.subplot(2,1,2); ax.grid('on')
  # continuous transitions
  ax.plot(t,o['dx'],'r'+lt,lw=lw,ms=ms)
  ax.plot(t,o['dy'],'b'+lt,lw=lw,ms=ms)
  # dashed line at discrete transitions
  #ax.plot(np.vstack(( td[1:-1:2], td[2::2])), 
  #         np.vstack((dhd[1:-1:2],dhd[2::2])), 'k--',lw=lw)
  ax.set_ylabel(r'$\dot{x}, \dot{y}$')
  #ax.set_xticklabels([])

  ax.set_xlabel('time (sec)')

