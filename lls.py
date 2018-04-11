import sys

from systems import Hybrid

import numpy as np
import pylab as plt

import matplotlib 
from matplotlib import animation, rc
#
# Say, "the default sans-serif font is Helvetica"
rc('font',**{'sans-serif':'Helvetica','family':'sans-serif','size':12})
rc('text',usetex=False)
# limit number of decimals printed in arrays
np.set_printoptions(precision=2)
	
class LLS(Hybrid):
  def __init__(self):
    """
    lateral leg-spring hybrid system

    parameters:
      m - scalar - body mass 
      I - scalar - body moment-of-inertia
      eta0 - scalar - leg rest length
      k - scalar - leg stiffness
      d - scalar - hip offset
      beta - scalar - leg angle
    """
    Hybrid.__init__(self)

  def leg(self, (k,t), (j,x), **p):
    """
    leg length
    """
    # unpack state
    x,y,theta,dx,dy,dtheta,fx,fy = x
    # foot, COM, hip
    f = np.array([fx,fy])
    c = np.array([x,y])
    h = c + p['d']*np.array([np.sin(theta),np.cos(theta)])
    # leg length
    eta = np.linalg.norm(h - f)
    # leg extension
    return eta 

  def foot(self, (k,t), (j,x), **p):
    """
    foot in body frame
    """
    # unpack state
    x,y,theta,dx,dy,dtheta,fx,fy = x
    # foot, COM, hip
    f = np.array([fx,fy])
    c = np.array([x,y])
    h = c + p['d']*np.array([np.sin(theta),np.cos(theta)])
    # foot in body frame
    fh = f-h
    fb = [fh[0]*np.cos(-theta)  + fh[1]*np.sin(-theta),
          fh[0]*-np.sin(-theta) + fh[1]*np.cos(-theta)]
    # foot distance from body axis
    return fb
      
  def dVdeta(self, eta, **p):
    """
    spring force
    """
    # linear spring
    return p['k']*(eta - p['eta0'])

  def F(self, (k,t), (j,x), **p):
    """
    .dyn  evaluates system dynamics
    """
    # perturbation
    acc = p['accel']((k,t), (j,x))
    # leg length
    eta = self.leg((k,t), (j,x), **p)
    # spring force
    dV = self.dVdeta(eta,**p)
    # unpack state
    x,y,theta,dx,dy,dtheta,fx,fy = x
    # Cartesian dynamics
    dx = [dx, dy, dtheta,
          -dV*(x + p['d']*np.sin(theta) - fx)/(p['m']*eta) + acc[0],
          -dV*(y + p['d']*np.cos(theta) - fy)/(p['m']*eta) + acc[1],
          -dV*p['d']*((x - fx)*np.cos(theta) 
               - (y - fy)*np.sin(theta))/(p['I']*eta) + acc[2],
          0.,0.]
    # return vector field
    return np.asarray(dx)

  def G(self, (k,t), (j,x), **p):
    sgn = (+1 if j == 'L' else -1)
    # leg extension, foot distance from body axis
    return np.asarray([p['eta0'] - self.leg((k,t), (j,x), **p), 
                       -sgn*self.foot((k,t), (j,x), **p)[0]])

  #def trans(self, t, x, q, e):
  def R(self, (k,t), (j,x), **p):
    # leg length
    eta = self.leg((k,t), (j,x), **p)
    # foot in body frame
    fb = self.foot((k,t), (j,x), **p)
    # unpack state
    x,y,theta,dx,dy,dtheta,fx,fy = x
    # foot, com, hip
    f = np.array([fx,fy])
    c = np.array([x,y])
    h = c + p['d']*np.array([np.sin(theta),np.cos(theta)])
    # leg length
    eta = np.linalg.norm(h - f)
    # left foot is right of body axis OR right foot is left of body axis
    if ( ( fb[0] > 0 ) and ( j == 'L' ) ) or ( ( fb[0] < 0 ) and ( j == 'R' ) ):
      # terminate simulation
      j_ = None
    else:	
      # switch stance foot
      j_ = ('R' if j == 'L' else 'L')
      # leg reset; q even for left stance, odd for right
      sgn = (+1 if j_ == 'L' else -1)
      # vector from hip to foot
      eta = p['eta0']*np.array([np.sin(theta - p['beta']*sgn),
                                np.cos(theta - p['beta']*sgn)])
      # new foot position
      fx,fy = h + eta
      # com vel
      dc = np.array([dx,dy])
      # hip vel
      dh = dc + p['d']*dtheta*np.array([np.cos(theta),-np.sin(theta)])
      # hip vel in body frame 
      dhb=[dh[0]*np.cos(-theta)  + dh[1]*np.sin(-theta),
           dh[0]*-np.sin(-theta) + dh[1]*np.cos(-theta)]
      ## leg will instantaneously extend 
      #if np.dot(fb, dhb) < 0:
      #  # terminate simulation
      #  j_ = None

    k_ = k+1
    t_ = t
    x_ = np.array([x,y,theta,dx,dy,dtheta,fx,fy])

    return (k_,t_),(j_,x_)

  def O(self, (k,t), (j,x), **p):
    # perturbation
    acc = p['accel']((k,t), (j,x))
    # leg length
    eta = self.leg((k,t), (j,x), **p)
    # unpack state
    x,y,theta,dx,dy,dtheta,fx,fy = x
    # foot, COM, hip
    f = np.array([fx,fy])
    c = np.array([x,y])
    h = c + p['d']*np.array([np.sin(theta),np.cos(theta)])
    hx,hy = h
    # energies
    PE = (k/2.)*(eta - p['eta0'])**2
    KE = p['m']*(dx**2 + dy**2)/2. + p['I']*(dtheta**2)/2.
    # translationally-invariant states
    v = np.abs(dx + 1.j*dy)
    delta = np.angle(np.exp(-1.j*theta)*(dy + 1.j*dx))
    omega = dtheta
    # observations
    o = dict(x=x,y=y,theta=theta,dx=dx,dy=dy,dtheta=dtheta,
             fx=fx,fy=fy,hx=hx,hy=hy,PE=PE,KE=KE,E=PE+KE,v=v,delta=delta,omega=omega,acc=acc)
    if 'costate' in p:
      o['p'] = p['costate']
    if 'variation' in p:
      o['P'] = p['variation']

    return o

  def phi(self,tf,x0,q0,t0=0.,debug=False):
    """
    t,x,q = phi(t,x0,q0)  hybrid flow

    Inputs:
      tf - scalar - final time
      x0 - initial state
      q0 - initial params
      (optional)
      t0 - scalar - initial time

    Outputs:
      T - times
      X - states
      Q - params

    """
    if tf == t0:
      T = [np.array([t0])]
      X = [np.array(x0)]
      Q = [np.array(q0)]
      
    else:
      self(t0,tf,x0,q0,np.inf,clean=True)

      T = np.hstack(self.t)
      X = np.vstack(self.x)
      Q = np.vstack([np.ones((x.shape[0],1))*q for x,q in zip(self.x,self.q)])

    return T,X,Q

  def step(self, z0, q0, steps=2):
    """
    .step  LLS stride from touchdown in body-centric coords

    Inputs:
      z - 1 x 3 - (v,delta,omega)
      q - 1 x 9 - (q,m,I,eta0,k,d,beta,.,.)
      (optional)
      steps - int - number of LLS steps to take

    Outputs:
      z - 1 x 3 - (v,delta,omega)
    """
    # instantiate extrinsic coords
    x0,q0 = self.extrinsic(z0, q0)
    # simulate for specified number of steps
    lls = self; lls.__init__(dt=self.dt)
    t,x,q = lls(0, 1e99, x0, q0, steps)
    # extract intrinsic coords
    z,_ = self.intrinsic(x[-1][-1], q[-1])
    return z

  def omap(self, z, args):
    """
    .omap  LLS orbit map in (v,delta) coordinates

    INPUTS
      z - 1 x 3 - (v,delta,omega)
        v - speed
        delta - heading
        omega - angular velocity
      args - (q)
        q - 1 x 9 - (q,m,I,eta0,k,d,beta,.,.)

    OUTPUTS
      z - 1 x 3 - (v,delta,omega)
    """
    if np.isnan(z).any():
        return np.nan*z
    q, = args

    return self.step(z, q)

  def anim(self, o=None, dt=1e-3, fign=1):
    """
    .anim  animate trajectory

    INPUTS:
      o - Obs - trajectory to animate

    OUTPUTS:
    """
    if o is None:
      o = self.obs().resample(dt)

    t = np.hstack(o.t)
    x = np.vstack(o.x)
    y = np.vstack(o.y)
    fx = np.vstack(o.fx)
    fy = np.vstack(o.fy)
    v = np.vstack(o.v)
    delta = np.vstack(o.delta)
    theta = np.vstack(o.theta)
    dtheta = np.vstack(o.dtheta)
    PE = np.vstack(o.PE)
    KE = np.vstack(o.KE)
    E = np.vstack(o.E)

    te = np.hstack(o.t[::2])
    xe = np.vstack(o.x[::2])
    ye = np.vstack(o.y[::2])
    thetae = np.vstack(o.theta[::2])
 
    z = np.array([v[-1],delta[-1],theta[-1],dtheta[-1]])

    def zigzag(a=.2,b=.6,c=.2,p=4,N=100):
      x = np.linspace(0.,a+b+c,N); y = 0.*x
      mb = np.round(N*a/(a+b+c)); Mb = np.round(N*(a+b)/(a+b+c))
      y[mb:Mb] = np.mod(np.linspace(0.,p-.01,Mb-mb),1.)-0.5
      return np.vstack((x,y))

    def Ellipse((x,y), (rx, ry), N=20, t=0, **kwargs):
      theta = 2*np.pi/(N-1)*np.arange(N)
      xs = x + rx*np.cos(theta)*np.cos(-t) - ry*np.sin(theta)*np.sin(-t)
      ys = y + rx*np.cos(theta)*np.sin(-t) + ry*np.sin(theta)*np.cos(-t)
      return xs, ys

    r = 1.01

    mx,Mx,dx = (x.min(),x.max(),x.max()-x.min())
    my,My,dy = (y.min(),y.max(),y.max()-y.min())
    dd = 5*r

    fig = plt.figure(fign,figsize=(5*(Mx-mx+2*dd)/(My-my+2*dd),5))
    plt.clf()
    ax = fig.add_subplot(111,aspect='equal')
    ax.set_xticks([])
    ax.set_yticks([])

    Lcom, = ax.plot(x[0], y[0], 'b.', ms=10.)
    Ecom, = ax.plot(*Ellipse((x[0],y[0]), (r, 0.5*r), t=theta[0]))
    Ecom.set_linewidth(4.0)
    Lft,  = ax.plot([x[0],fx[0]],[y[0],fy[0]],'g.-',lw=4.)

    ax.set_xlim((mx-dd,Mx+dd))
    ax.set_ylim((my-dd,My+dd))

    for k in range(x.size):

        Lcom.set_xdata(x[k])
        Lcom.set_ydata(y[k])
        Lft.set_xdata([x[k],fx[k]])
        Lft.set_ydata([y[k],fy[k]])
        Ex,Ey = Ellipse((x[k],y[k]), (0.5*r, r), t=theta[k])
        Ecom.set_xdata(Ex)
        Ecom.set_ydata(Ey)

        fig.canvas.draw()

  def plot(self,o=None,dt=1e-3,fign=-1,clf=True,axs0={},ls='-',ms='.',
                alpha=1.,lw=2.,fill=True,legend=True,color='k',
           plots=['2d','v','E'],label=None,cvt={'t':1000,'acc':1./981}):
    """
    .plot  plot trajectory

    INPUTS:
      o - Obs - trajectory to plot

    OUTPUTS:
    """
    if o is None:
      o = self.obs().resample(dt)

    t      = np.hstack(o.t) * cvt['t']
    x      = np.vstack(o.x)
    y      = np.vstack(o.y)
    theta  = np.vstack(o.theta)
    dx     = np.vstack(o.dx)
    dy     = np.vstack(o.dy)
    dtheta = np.vstack(o.dtheta)
    v      = np.vstack(o.v)
    delta  = np.vstack(o.delta)
    fx     = np.vstack(o.fx)
    fy     = np.vstack(o.fy)
    PE     = np.vstack(o.PE)
    KE     = np.vstack(o.KE)
    E      = np.vstack(o.E)
    acc    = np.vstack(o.acc) * cvt['acc'] 

    qe      = np.vstack(o.q[::2])
    te      = np.hstack(o.t[::2]) * 1000
    xe      = np.vstack(o.x[::2])
    ye      = np.vstack(o.y[::2])
    thetae  = np.vstack(o.theta[::2])
    ve      = np.vstack(o.v)
    deltae  = np.vstack(o.delta[::2])
    thetae  = np.vstack(o.theta[::2])
    dthetae = np.vstack(o.dtheta[::2])
    fxe    = np.vstack(o.fx[::2])
    fye    = np.vstack(o.fy[::2])

    def do_fill(te,qe,ylim):
      for k in range(len(te)-1):
        if qe[k,0] == 0:
          color = np.array([1.,0.,0.])
        if qe[k,0] == 1:
          color = np.array([0.,0.,1.])
        ax.fill([te[k],te[k],te[k+1],te[k+1]],
                [ylim[1],ylim[0],ylim[0],ylim[1]],
                fc=color,alpha=.35,ec='none',zorder=-1)
 
    fig = plt.figure(fign)
    if clf:
      plt.clf()
    axs = {}

    Np = len(plots)
    pN = 1

    if '2d' in plots:
      if '2d' in axs0.keys():
        ax = axs0['2d']
      else:
        ax = plt.subplot(Np,1,pN,aspect='equal'); pN+=1; ax.grid('on')
      #xlim = np.array([te[0],te[-1]]) + np.array([-.02,.02])*(te[-1]-te[0])
      #ylim = np.array([-.1,1.1])*E.mean()
      if fill:
        #ax.plot(xe,ye,'y+',mew=2.,ms=8)
        ax.plot(fxe[qe==0.],fye[qe==0.],'ro',mew=0.,ms=10)
        ax.plot(fxe[qe==1.],fye[qe==1.],'bo',mew=0.,ms=10)
      ax.plot(x ,y ,color=color,ls=ls,lw=lw,alpha=alpha,label=label)
      #ax.set_xlim(xlim); ax.set_ylim(ylim)
      ax.set_xlabel('x (cm)'); ax.set_ylabel('y (cm)')
      axs['2d'] = ax

    if 'y' in plots:
      if 'y' in axs0.keys():
        ax = axs0['y']
      else:
        ax = plt.subplot(Np,1,pN); pN+=1; ax.grid('on')
      #xlim = np.array([te[0],te[-1]]) + np.array([-.02,.02])*(te[-1]-te[0])
      #ylim = np.array([-.1,1.1])*E.mean()
      ax.plot(t,y ,color=color,ls=ls,lw=lw,alpha=alpha,label=label)
      #ax.set_xlim(xlim); ax.set_ylim(ylim)
      ax.set_ylabel('y (cm)')
      axs['y'] = ax

    if 'v' in plots:
      ax = plt.subplot(Np,1,pN); pN+=1; ax.grid('on')
      xlim = np.array([te[0],te[-1]]) + np.array([-.02,.02])*(te[-1]-te[0])
      ylim = np.array([-.1,1.1])*v.max()
      #ax.plot(np.vstack([te,te]),(np.ones((te.size,1))*ylim).T,'k:',lw=1)
      ax.plot(t,v,color=color,ls=ls,  lw=lw,label='$v$',alpha=alpha)
      ax.set_xlim(xlim); ax.set_ylim(ylim)
      if legend:
        ax.legend(loc=7,ncol=3)
      if fill:
        do_fill(te,qe,ylim)
      ax.set_ylabel('v (cm/sec)')
      axs['v'] = ax

    if 'acc' in plots:
      if 'acc' in axs0.keys():
        ax = axs0['acc']
      else:
        ax = plt.subplot(Np,1,pN); pN+=1; ax.grid('on')
      xlim = np.array([te[0],te[-1]]) + np.array([-.02,.02])*(te[-1]-te[0])
      ylim = np.array([min(0.,1.2*acc.min()),1.2*acc.max()])
      ax.plot(t,acc,color=color,ls=ls,  lw=lw,label='$a$',alpha=alpha)
      #ax.set_xlim(xlim); #ax.set_ylim(ylim)
      if legend:
        ax.legend(loc=7,ncol=3)
      #if fill:
      #  do_fill(te,qe,ylim)
      #ax.set_ylabel('roach perturbation (cm / s$^{-2}$)')
      ax.set_ylabel('cart acceleration (g)')
      axs['acc'] = ax

    if 'E' in plots:
      ax = plt.subplot(Np,1,pN); pN+=1; ax.grid('on')
      xlim = np.array([te[0],te[-1]]) + np.array([-.02,.02])*(te[-1]-te[0])
      ylim = np.array([-.1,1.1])*E.mean()
      ax.plot(t,E ,color=color,ls=ls,lw=lw,label='$E$',alpha=alpha)
      ax.plot(t,KE,'b',ls=ls,  lw=lw,label='$KE$',alpha=alpha)
      ax.plot(t,PE,'g',ls=ls,  lw=lw,label='$PE$',alpha=alpha)
      ax.set_xlim(xlim); ax.set_ylim(ylim)
      if legend:
        ax.legend(loc=7,ncol=3)
      for k in range(len(te)-1):
        if qe[k,0] == 0 and fill:
          ax.fill([te[k],te[k],te[k+1],te[k+1]],
                  [ylim[1],ylim[0],ylim[0],ylim[1]],
                  fc=np.array([1.,1.,1.])*0.75,ec='none',zorder=-1)
      ax.set_ylabel('E (g m**2 / s**2)')
      axs['E'] = ax

    ax.set_xlabel('time (msec)');

    return axs

  def extrinsic(self, z, x=0., y=0., theta=np.pi/2., **p):
    """
    .extrinsic  extrinsic LLS state from intrinsic (i.e. Poincare map) state

    INPUTS:
      z - 1 x 3 - (v,delta,omega) - TD state

    OUTPUTS:
      x - 1 x 8 - (x,y,theta,dx,dy,dtheta,fx,fy)
    """
    v,delta,omega = z
    # extrinsic state variables
    dx = v*np.sin(theta + delta)
    dy = v*np.cos(theta + delta)
    dtheta = omega
    # COM, hip
    c = np.array([x,y])
    h = c + d*np.array([np.sin(theta),np.cos(theta)])
    # foot
    sgn = (+1 if j == 'L' else -1)
    f = h + p['eta0']*np.array([np.sin(theta - p['beta']*sgn),
                                np.cos(theta - p['beta']*sgn)])
    fx,fy = f
    # pack params, state
    x = np.array([x,y,theta,dx,dy,dtheta,fx,fy])
    return x

  def intrinsic(self, x, q):
    """
    .intrinsic  Poincare map state from from full state

    Inputs:
      x - 1 x 6 - (x,y,theta,dx,dy,dtheta,fx,fy)

    Outputs:
      z - 1 x 3 - (v,delta,omega) - TD state
    """
    x,y,theta,dx,dy,dtheta,fx,fy = x
    v = np.abs(dx + 1.j*dy)
    delta = np.angle(np.exp(-1.j*theta)*(dy + 1.j*dx))
    omega = dtheta
    z = np.array([v,delta,omega])
    return z

if __name__ == "__main__":

  import sys
  args = sys.argv

  j = 'L'
  sgn = (+1 if j == 'L' else -1)

  kg2g = 1000
  m2cm = 100

  m = 0.0029 * kg2g
  I = 2.5e-7 * kg2g * m2cm**2
  eta0 = 0.017 * m2cm
  k = 1.53 * kg2g
  d = -0.002 * m2cm
  beta = np.pi/4.

  accel = lambda (k,t), (j,x) : np.zeros_like(x).T[:3]

  debug = True
  debug = False
  Zeno = True
  Zeno = False

  hs = LLS()
  p = dict(m=m, I=I, eta0=eta0, k=k, d=d, beta=beta, accel=accel, debug=debug, Zeno=Zeno)

  v = 0.51 * m2cm
  delta = -0.03 * sgn
  omega = 0.1 * sgn
  v,delta,omega = .5 * m2cm,0.,0.
  fx=0.
  fy=0.

  z = [v,delta,omega]

  X,Y,theta = 0.,0.,np.pi/2.
  #x=np.random.randn(); y=np.random.randn(); theta=2*np.pi*np.random.rand()
  x = hs.extrinsic(z, x=X, y=Y, theta=theta, **p)

  fps = 500
  dt  = 1./fps
  dt  = 1e-4
  rx = 1e-12

  K   = 2 # number of hybrid transitions in simulation
  T   = np.inf

  trjs = hs.sim((K,T), (j,x), dt, rx, **p);
  trjs = hs.cosim(trjs, np.identity(len(x)), **p)
  trjs = hs.varsim(trjs, np.identity(len(x)), **p)

  import util

  flow_T = lambda x,j=j : hs.sim((K,T), (j,x), dt, rx, **p)[-1]['x'][-1]

  #(k,t),(j,o) = hs.obs(trjs, insert_nans=True, **p)
  (k,t),(j,o) = hs.obs(trjs, insert_nans=False, **p)
  (kd,td),(jd,od) = hs.obs(trjs, only_disc=True, **p)

  #sys.exit(0)
  p0 = o['p'][0]
  PT = o['P'][-1]

  (k,t),(j,o) = hs.obs(trjs, insert_nans=False, **p)
  (kd,td),(jd,od) = hs.obs(trjs, only_disc=True, **p)

  x,y,theta = o['x'],o['y'],o['theta']
  fx,fy,hx,hy = o['fx'],o['fy'],o['hx'],o['hy']

  def zigzag(z0,z1,a=.2,b=.6,c=.2,h=1.,p=4,N=100):
    x = np.linspace(0.,a+b+c,N); y = 0.*x
    mb = int(N*a/(a+b+c)); Mb = int(N*(a+b)/(a+b+c))
    y[mb:Mb] = np.mod(np.linspace(0.,p-.01,Mb-mb),1.)-0.5
    z = ((np.abs(z1-z0)*x + 1.j*h*y)) * np.exp(1.j*np.angle(z1-z0)) + z0
    return z 

  def Ellipse((x,y), (rx, ry), N=20, t=0, **kwargs):
    theta = 2*np.pi/(N-1)*np.arange(N)
    xs = x + rx*np.cos(theta)*np.cos(-t) - ry*np.sin(theta)*np.sin(-t)
    ys = y + rx*np.cos(theta)*np.sin(-t) + ry*np.sin(theta)*np.cos(-t)
    return xs, ys

  r = 1.01

  mx,Mx,dx = (x.min(),x.max(),x.max()-x.min())
  my,My,dy = (y.min(),y.max(),y.max()-y.min())
  dd = 5*r

  lw = 2
  mew = lw
  ms = 10
  lt = '-'
  fs = (6,6)

  fig = plt.figure(1,figsize=(5*(Mx-mx+2*dd)/(My-my+2*dd),5))
  plt.clf()
  ax = fig.add_subplot(111,aspect='equal')

  Lcom, = ax.plot([],[], 'b.', ms=10.)
  Ecom, = ax.plot([],[],'b',lw=2*lw)
  Lft,  = ax.plot([],[],'g-',lw=lw)

  def init():
    #
    #ax.set_xticks([])
    #ax.set_yticks([])
    #
    ax.set_xlim((mx-dd,Mx+dd))
    ax.set_ylim((my-dd,My+dd))
    #
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

  def animate(_t):
    i = (t >= _t).nonzero()[0][0]
    #
    Lcom.set_xdata(x[i])
    Lcom.set_ydata(y[i])
    Lz = zigzag(hx[i] + 1.j*hy[i], fx[i] + 1.j*fy[i],h=0.5)
    Lft.set_xdata(Lz.real)
    Lft.set_ydata(Lz.imag)
    Ex,Ey = Ellipse((x[i],y[i]), (0.5*r, r), t=theta[i])
    Ecom.set_xdata(Ex)
    Ecom.set_ydata(Ey)
    return (Lcom,Lft,Ecom,)

  #fps = 30
  dt = 1./fps
  msec = int(1000*dt)
  ani = animation.FuncAnimation(fig, animate, np.arange(0.,t[-1],dt), init_func=init, 
                                interval=msec, blit=True, repeat=False)

  plt.show()
  sys.exit(0)


  #if 'plot' in args or 'anim' in args:
  #  o = lls.obs().resample(dt)
  #  if 'anim' in args:
  #    lls.anim(o=o)
  #  if 'plot' in args:
  #    lls.plot(o=o)

  #v,delta,omega = z
  #op.pars(lls=lls,
  #        x=X,y=Y,theta=theta,
  #        v=v,delta=delta,omega=omega,
  #        x0=x0,q0=q0,
  #        T=np.diff([tt[0] for tt in t[::2]]).mean())

  #x = np.array([  9.89e-01,  -1.82e-02,   1.57e+00,   4.88e+01,  -2.66e+00, -6.16e-01,   1.00e+00,   1.20e+00]) 
  #T = 0.03

  #x = np.array([  2.22363379,  -0.15055052,   1.5392972 ,  49.54367402, -6.74504873,  -1.56679983,   1.        ,   1.2       ])
  #T = 20*1e-6

  trjs = hs.sim((K,T), (j,x), dt, rx, **p);
  trjs = hs.cosim(trjs, np.identity(len(x)), **p)
  trjs = hs.varsim(trjs, np.identity(len(x)), **p)

  import util

  flow_T = lambda x,j=j : hs.sim((K,T), (j,x), dt, rx, **p)[-1]['x'][-1]

  #(k,t),(j,o) = hs.obs(trjs, insert_nans=True, **p)
  (k,t),(j,o) = hs.obs(trjs, insert_nans=False, **p)
  (kd,td),(jd,od) = hs.obs(trjs, only_disc=True, **p)

  #sys.exit(0)
  p0 = o['p'][0]
  PT = o['P'][-1]

  # check stepsize for derivative
  #plt.figure(1); plt.clf()
  #d = np.logspace(-10,0,11)
  #d = np.logspace(-5,-3,11)
  ##D = np.asarray([util.D(flow_T,x,d=_).flatten() for _ in d])
  #D = np.asarray([(util.D(flow_T,x,d=_)-p0).flatten() for _ in d])
  #plt.loglog(d,np.abs(D))
  #plt.loglog(d,np.abs(D).max(axis=1),lw=10)

  ## choose auspicious stepsize
  ##d = dt 
  ##d = 5e-5
  #d = d[np.argmin(D.max(axis=1))]
  #Dflow = util.D(flow_T,x,d=d)
  #print 'Dflow =\n',Dflow
  #print 'Dflow - p[0] = %0.2e'%np.abs(Dflow - p0).max()
  #print 'Dflow - P[T] = %0.2e'%np.abs(Dflow - PT).max()
  #
  #print 'p[0] =\n',p0
  #print 'P[T] =\n',PT
  print 'p[0] - P[T] = %0.2e'%np.abs(p0 - PT).max()
  #sys.exit(0)
