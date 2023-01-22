import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('observed.txt') #obserced mu
sim = np.loadtxt('simulated.txt') # simulated mu - unoscillated event rate prediction

def P(E, theta, dm_sq, L):
    '''
    Probability that \nu_mu will be observed as \nu_mu
    '''
    return 1 - np.sin(2*theta)**2 * np.sin((1.267 * dm_sq * L)/E)**2

def oscillated_sim(sim, theta, dm):
    '''
    ''
    Impose probability of neutrino to be tau on simulated data.
    Parameters
    ----------
    sim : array
        simulated data
    theta : scalar
        theta parameter in the probability
    dm : int
        dm**2 parameter in the probability

    Returns array of predicted counts according to probability
    '''
    L = 295
    lamdas = []
    E = np.arange(0.025,10.025,0.05) #taking the midpoint energy of each bin
    for i,En in zip(sim,E):
        newcount = P(En,theta,dm,L) * i
        lamdas.append(newcount)
    return np.array(lamdas)


def NLL(u , sim = sim, data = data):
    '''
    Calculates the Negative log Likehood for a set of input parameters u = [theta, dm**2],
    sim = predicted data using those parameters
    data = actual observed data

    '''
    li = oscillated_sim(sim, *u)
    mi = data
    if len(li) == len(mi):
        suma = 0
        for i,m in zip(li,mi):
            if m == 0:
                suma += i #set this like this to avoid log(0)
            else:
                suma += i - m + m * (np.log(m) - np.log(i))
        return suma
    else:
        print('lambda_i and m_i not same size')
        

def P_2(f, xarray):
    '''
    Takes in array of 3 xvalues:xstart, xmidpoint, xstop
    plots a parabola and gives the minimum of parapola, x_min
    '''
    x0 = xarray[0]
    x1 = xarray[1]
    x2 = xarray[2]
    y0 = f(x0)
    y1 = f(x1)
    y2 = f(x2)
    num = (x2*x2 - x1*x1)*y0 + (x0*x0 - x2*x2)*y1 + (x1*x1 - x0*x0)*y2
    denom = (x2-x1)*y0 + (x0-x2)*y1 + (x1-x0)*y2
    return 1/2 * num/denom

def secand(f, x_1, x_2, tol = 0.001):
    f1 = f(x_1)
    if f1 == 0.0: return x_1
    f2 = f(x_2)
    if f2 == 0.0: return x_2
    iteration = 0
    x_3 = 0
    while abs(f(x_2)) > tol:
        # print(f2-f1)
        x_3 = x_2 - f2 * (x_2 - x_1)/(f2 - f1)
        x_1 = x_2
        x_2 = x_3
        f1 = f(x_1)
        f2 = f(x_2)
        iteration += 1
        # print(f1,f2)
        # print(f2)
        # print(f'iteration = {iteration}')
    return x_3, iteration


def parabmin(f, a, b, threshold):
    '''
    Parabolic optimiser.
    Gived minimum value of f and takes input of x=a, x=b, and the threshold where
   abs( x_min_new - x_min_old )< threshold --> converging condition
    '''
    x = [a, (a+b)/2, b]
    y = [f(x[0]), f(x[1]), f(x[2])]
    x_min = P_2(f,x)
    
    y_min = f(x_min)
    # print('first approx=', x_min)
    # if max(y) > y_min:
    # print(y)
    ylowind = y.index(min(y))
    dx = abs(x[ylowind] - x_min) 
    # print('first dx ', dx)
    count = 0
    while dx > threshold:
        count += 1
        yhighind = y.index(max(y))
        x[yhighind] = x_min
        y = [f(x[0]), f(x[1]), f(x[2])]
    
        x_min = P_2(f,x)
        y_min = f(x_min)
        ylowind = y.index(min(y))
        dx = abs(x[ylowind] - x_min)
        # print('iteration =', count) # UNHASH THIS TO PRINT EACH ITERATION NUMBER
        # print(x_min)
    y_last = [f(x[0]), f(x[1]), f(x[2])]

    return x_min,count,dx, x, y_last



def bisection(f,a,b,tol = 0.01):
    if np.sign(f(a)) == np.sign(f(b)):
        raise Exception('a,b do not enclose the root')
    m = (a+b)/2
    count = 0
    while abs(f(m)) > tol:
        if np.sign(f(a)) != np.sign(f(m)):
            b = m
            m = (a+b)/2
        elif np.sign(f(b)) != np.sign(f(m)):
            a = m
            m = (a+b)/2
        count += 1
        # print(a,b,count)
    return m, count


def minimize_theta(theta, dm, sim, data, extra = 0, parabminimiserthreshold = 1e-7):
    ''' 
    these are here to find the index of the lowest value to be used as a guess
    '''
    thetas = np.linspace(0, 3/2 *np.pi/4, 1000)
    dms = np.array([dm] * len(thetas))

    u = np.stack((thetas,dms))
    nlltheta = [NLL(u[:,i], sim, data) for i in range(len(thetas))]  
    theta_guess = thetas[nlltheta.index(min(nlltheta))]
    def NLL_theta(theta, dm_val = dm): #defining NLL just for theta varying to use in minimiser
        u = [theta,dm_val]
        nllvalue = NLL(u,sim,data)
        return nllvalue
    # print(theta_guess)
    theta_min ,count,dx,x_latest,y_latest = parabmin(NLL_theta, theta_guess-np.pi/1000, theta_guess +np.pi/1000, parabminimiserthreshold) # 
    
    #find error
    def nllroot(theta):
        b = NLL_theta(theta_min) + 0.5
        return NLL_theta(theta) - b
    
    leftroot, count = secand(nllroot,theta_min,theta_min + 0.001)
    std = abs(leftroot - theta_min)
    
    if extra == 0:
        # plt.clf()
        # plt.plot(thetas, nlltheta,'-', label = r'fixed $dm $ =  %s ' % dms[0])
        # plt.axvline(x = theta_min, linestyle = '-.',label = r'$thetamin$ from parabolic search',  )
        # plt.title(r'NLL($\theta$)')
        # plt.ylabel(r'NLL($\theta$)')
        # plt.xlabel(r'\theta')
        # plt.legend()
        # plt.show()
        # plt.xlim(0.002,0.0028)
        return [theta_min,dm], std
    if extra == 1:
        print(r'Value of theta_min from parabolic optimisation = %a, \
          Number of iterations = %a  \
              last change in x = %a  \
                  Last 3 x values in search = %a \
                      last 3 y values in search = %a \
                          theta_guess from min = %a'
              % (theta_min, count, dx, x_latest,y_latest, theta_guess)) 
        return theta_min,count,dx,x_latest,y_latest
    

def minimize_dm(theta,dm,sim,data,extra = 0, parabminimiserthreshold = 1e-7):
    dms = np.linspace(0,0.02,1000)
    thetas = np.array([theta] * len(dms))
    u = np.stack((thetas,dms))
    nlldm = [NLL(u[:,i], sim, data) for i in range(len(dms))]
    # print(type(nlldm))
    dm_guess = dms[nlldm.index(min(nlldm))]
    
    def NLL_dm(dm, theta_val = theta): #defining NLL just for dm varying to use in minimiser
        u = [theta,dm]
        nllvalue = NLL(u,sim,data)
        return nllvalue
    
    dm_min ,count,dx,x_latest,y_latest = parabmin(NLL_dm, dm_guess-0.0025/5, dm_guess+0.0025/4, parabminimiserthreshold)
    
    def nllroot(dm):
        b = NLL_dm(dm_min) + 0.5
        return NLL_dm(dm) - b
    
    leftroot, count = secand(nllroot, dm_min, dm_min + 0.001)
    std = abs(leftroot - dm_min)
    
    if extra == 0:
        # plt.clf()
        # plt.plot(dms, nlldm,'-', label = r'fixed $\theta $ =  %s ' % thetas[0])
        # plt.axvline(x = dm_min, linestyle = '-.',label = r'$dm_{min}$ from parabolic search',  )
        # plt.title(r'NLL($dm$)')
        # plt.ylabel(r'NLL($dm$)')
        # plt.xlabel(r'dm')
        # plt.legend()
        # plt.show()
        # plt.xlim(0.002,0.0028)
        return [theta, dm_min], std
       
    
    if extra == 1:
        print(r'Value of dm_min from parabolic optimisation = %a, \
          Number of iterations = %a  \
              last change in x = %a  \
                  Last 3 x values in search = %a \
                      last 3 y values in search = %a \
                          dm_guess from min = %a'
              % (dm_min, count, dx, x_latest,y_latest, dm_guess)) 
        return dm_min, count, dx, x_latest, y_latest
     
     
def univariate(theta, dm, sim, data, first = 0, threshold = 1e-4):
    count = 0
    errors = []
    if first == 0:    
        u0 = [theta,dm]
        u1, er1 = minimize_theta(*u0,sim,data)
        # print(u0,u1)
        # print( abs(u0[0] - u1[0]), abs(u0[1] - u1[1]) )
        errors.append(er1)
        # print(u1)
        while abs(u0[0] - u1[0]) > threshold or abs(u0[1] - u1[1]) > threshold:
            u0, er0 = minimize_dm(*u1,sim,data)
            u1, er1 = minimize_theta(*u0,sim,data)
            count += 1
            errors.append(er0)
            errors.append(er1)
            print(f'Univariate iteration number = {count}')
            # print(u1)
            if count == 50:
                break
        return u1,count, errors
    if first == 1:
        u0 = [theta,dm]
        u1, er1 = minimize_dm(*u0,sim,data)
        errors.append(er1)
        # print(u0,u1)
        # print( abs(u0[0] - u1[0]), abs(u0[1] - u1[1]) )
        while abs(u0[0] - u1[0]) > threshold or abs(u0[1] - u1[1]) > threshold:
            u0,er0 = minimize_theta(*u1,sim,data)
            u1,er1 = minimize_dm(*u0,sim,data)
            count += 1
            errors.append(er0)
            errors.append(er1)
            print(f'Univariate iteration number = {count}')
            if count == 50:
                break
        return u1,count, errors # as of now, errors are returned as a list of errors for each iteration: think about how you're gonna use them to find total error

def nll(theta,dm, sim = sim, data = data):
    '''
    Calculates the Negative log Likehood for a set of input parameters u = [theta, dm**2],
    sim = predicted data using those parameters
    data = actual observed data

    '''
    li = oscillated_sim(sim, theta,dm)
    mi = data
    if len(li) == len(mi):
        suma = 0
        for i,m in zip(li,mi):
            if m == 0:
                suma += i #set this like this to avoid log(0)
            else:
                suma += i - m + m * (np.log(m) - np.log(i))
        return suma
    else:
        print('lambda_i and m_i not same size')

def f_x(f, x, y, h, variable = 0): # variable = 1 gives f_y
    '''
    Parameters
    ----------
    f : function we want to find 1st derivative of (2d function)
    x : value of x at the ppoint where we want to find 1st derivative
    y : value of y at the point where we want to find 1st derivative
    h : difference (method) value
    
    variable : TYPE, optional
        DESCRIPTION. if x = 0, differentiaten 1st variable => x
                     if x = 1, differentiate 2nd variable => y

    Returns: value of f 1st derivative at point x,y
    '''
    if variable == 0:
        numerator = f(x+h, y) - f(x-h, y)
        # print(f(x+h,y))
        # print(f(x-h,y))
        # print(numerator)
        return numerator/(2*h)
    if variable == 1:
        numerator = f(x, y+h) - f(x, y-h)
        return numerator/(2*h)


    
def f_xx(f, x, y, h, variable = 0): #variable = 1 gives f_yy
    if variable == 0:
        numerator = f(x + h,y) - 2*f(x,y) + f(x-h,y)
        return numerator/(h**2)
    if variable == 1:
        numerator = f(x,y+h) - 2* f(x,y) + f(x,y-h)
        return numerator/(h**2)

def f_xy(f, x, y, h, k):
    numerator = f(x+h, y + k) - f(x+h,y) - f(x, y+k) + 2*f(x,y) - f(x - h, y) - f(x,y - k) + f(x-h, y-k)
    return numerator/(2*h*k)



def H(f, x, y, h):
    hessian = np.array([[f_xx(f,x,y,h,variable = 0), f_xy(f,x,y,h,h)], 
                       [f_xy(f,x,y,h,h), f_xx(f,x,y,h,variable = 1)]])
    return hessian

def newton(f,x,y,h,threshold = 1e-3):
    xn = np.array([x,y])
    hessian = H(f,xn[0],xn[1],h)
    print(hessian)
    delf = np.array([f_x(f,xn[0],xn[1],h), f_x(f,xn[0],xn[1],h,variable=1)])
    xnext = xn - np.dot(np.linalg.inv(hessian),  delf)
    dx = np.abs(xn - xnext)
    count = 0

    while dx[0] > threshold and dx[1] > threshold:
        xn = xnext
        hessian = H(f,xn[0],xn[1],h)
        delf = np.array([f_x(f,xn[0],xn[1],h), f_x(f,xn[0],xn[1],h,variable=1)])
        xnext = xn - np.dot(np.linalg.inv(hessian),  delf)
        dx = np.abs(xn - xnext)
        count += 1
        # print(dx)
    return xn,count

def acceptancefunction(dE, T):
    if dE <= 0.0:
        return 1
    elif np.random.uniform(0,1,1) < np.exp(-1.0 * dE/T):
        return 1
    return 0

def ProposalFunction(u,sigma_theta,sigma_dm):
    x = np.random.normal(0.0, sigma_theta) + u[0]
    y = np.random.normal(0.0, sigma_dm) + u[1]
    return np.array([x,y])

def runchain(nll,theta0, dm0, sim, data):
    T = 3
    temps = np.arange(T,0,-0.25)
    iterations = 50000
    
    values = [theta0,dm0]
    
    listofvalues = []
    
    for T in temps:
        zeros = np.zeros((iterations+1,2))
        zeros[0,:] = values 
        print('Currently runnin on temp =', T)
        for j in range(iterations):
            unow = zeros[j,:]
            # print(unow)
            unext = ProposalFunction(unow, 0.1, 0.5e-3)
            nllnow = nll(unow[0],unow[1])
            # print(unext)
            # print(type(unext))
            nllnext = nll(unext[0],unext[1])
            dE = nllnext - nllnow
            acceptstep = acceptancefunction(dE,T)
            if acceptstep == 1:
                zeros[j+1,:] = unext
            if acceptstep == 0:
                zeros[j+1,:] = unow
        values = zeros[-1,:]
        listofvalues.append(zeros)
    return listofvalues

def nllerror(theta_min,dm_min, thetafactor = 0.02):
    def nllroot_theta(theta):
        b = NLL_theta(theta_min) + 0.5 
        return NLL_theta(theta) - b
        
    def NLL_theta(theta, dm = dm_min): #defining NLL just for theta varying to use in minimiser
        # dm = 2.4e-3
        u = [theta,dm]
        nllvalue = NLL(u,sim,data)
        return nllvalue
    
    leftroot_theta, leftcount = bisection(nllroot_theta, 0.710, theta_min, tol = 1e-9)
    # print('leftroot from bisection =', leftroot, leftcount)
    leftdiff_theta = abs(leftroot_theta - theta_min)
    # print('Difference of NLL(thetaroot) - NLL(theta_min) =', NLL_theta(leftroot) - NLL_theta(theta_min))
    
    rightroot_theta, rightcount = bisection(nllroot_theta, theta_min, theta_min + thetafactor, tol = 1e-9)
    # print('rightroot from bisection =', rightroot, rightcount)
    rightdiff_theta = abs(rightroot_theta - theta_min)
    # print('Difference of NLL(thetaroot) - NLL(theta_min) =', NLL_theta(rightroot) - NLL_theta(theta_min))
    
    # std_theta = (leftdiff + rightdiff)/2
    # print(f'standard div = {std_theta}')
    
    def NLL_dm(dm, theta_val = theta_min): #defining NLL just for dm varying to use in minimiser
        u = [theta_val,dm]
        nllvalue = NLL(u,sim,data)
        return nllvalue
            
    def nllroot_dm(dm):
        b = NLL_dm(dm_min) + 0.5
        return NLL_dm(dm) - b
        
    leftroot_dm, leftcount = bisection(nllroot_dm, 0.0022, dm_min, tol = 1e-9)
    # print('leftroot from bisection =', leftroot, leftcount)
    leftdiff_dm = abs(leftroot_dm - dm_min)
    # print('Difference of NLL(dmroot) - NLL(dm_min) =', NLL_dm(leftroot) - NLL_dm(dm_min))
    
    rightroot_dm, rightcount = bisection(nllroot_dm, dm_min,0.0025, tol = 1e-9)
    # print('rightroot from bisection =', rightroot, rightcount)
    rightdiff_dm = abs(rightroot_dm - dm_min)
    # print('Difference of NLL(dmroot) - NLL(dm_min) =', NLL_dm(rightroot) - NLL_dm(dm_min))
    # std_dm = (leftdiff + rightdiff)/2
    # print(f'standard div dm = {std_dm}')
    return [rightdiff_theta,leftdiff_theta,rightdiff_dm,leftdiff_dm]
#%% Part 5 functions
def oscillated_sim3(sim, theta, dm, a):
    '''
    ''
    Impose probability of neutrino to be tau on simulated data.
    Parameters
    ----------
    sim : array
        simulated data
    theta : scalar
        theta parameter in the probability
    dm : int
        dm**2 parameter in the probability

    Returns array of predicted counts according to probability
    '''
    L = 295
    lamdas = []
    E = np.arange(0.025,10.025,0.05) #taking the midpoint energy of each bin
    for i,En in zip(sim,E):
        newcount = P(En,theta,dm,L) * i
        newcount = newcount * a * En
        lamdas.append(newcount)
    return np.array(lamdas)

def nll3(theta, dm, a, sim = sim, data = data):
    '''
    Calculates the Negative log Likehood for a set of input parameters u = [theta, dm**2],
    sim = predicted data using those parameters
    data = actual observed data

    '''
    li = oscillated_sim3(sim, theta,dm,a)
    mi = data
    if len(li) == len(mi):
        suma = 0
        for i,m in zip(li,mi):
            if m == 0:
                suma += i #set this like this to avoid log(0)
            else:
                suma += i - m + m * (np.log(m) - np.log(i))
        return suma
    else:
        print('lambda_i and m_i not same size')
 
def f_x3(f, x, y, z, h, variable = 0): # variable = 1 gives f_y
    '''
    Parameters
    ----------
    f : function we want to find 1st derivative of (2d function)
    x : value of x at the ppoint where we want to find 1st derivative
    y : value of y at the point where we want to find 1st derivative
    h : difference (method) value
    
    variable : TYPE, optional
        DESCRIPTION. if x = 0, differentiaten 1st variable => x
                     if x = 1, differentiate 2nd variable => y

    Returns: value of f 1st derivative at point x,y
    '''
    if variable == 0:
        numerator = f(x+h, y, z) - f(x-h, y, z)
        return numerator/(2*h)
    if variable == 1:
        numerator = f(x, y+h, z) - f(x, y-h, z)
        return numerator/(2*h)
    if variable == 2:
        numerator = f(x, y, z+h) - f(x, y, z-h)
        return numerator/(2*h)


    
def f_xx3(f, x, y, z, h, variable = 0): #variable = 1 gives f_yy
    if variable == 0:
        numerator = f(x + h,y,z) - 2*f(x,y,z) + f(x-h,y,z)
        return numerator/(h**2)
    if variable == 1:
        numerator = f(x,y+h,z) - 2* f(x,y,z) + f(x,y-h,z)
        return numerator/(h**2)
    if variable == 2:
        numerator = f(x,y,z+h) - 2*f(x,y,z) + f(x,y,z-h)
        return numerator/(h**2)

def f_xy3(f, x, y, z, h, k):
    numerator = f(x+h, y + k,z) - f(x+h,y,z) - f(x, y+k,z) + 2*f(x,y,z) - f(x - h, y,z) - f(x,y - k,z) + f(x-h, y-k,z)
    return numerator/(2*h*k)

def f_xz3(f,x,y,z,h,k):
    numerator = f(x+h,y,z+k) - f(x+h,y,z) - f(x,y,z+k) + 2*f(x,y,z) - f(x-h,y,z) - f(x,y,z-k) + f(x-h,y,z-k)
    return numerator/(2*h*k)

def f_yz3(f,x,y,z,h,k):
    numerator = f(x,y+h,z+k) - f(x,y+h,z) - f(x,y,z+k) + 2*f(x,y,z) - f(x,y-h,z) - f(x,y,z-k) + f(x,y-h,z-k)
    return numerator/(2*h*k)
        
        
def H3(f, x, y, z, h):
    hessian = np.array([[f_xx3(f,z,y,z,h,variable = 0), f_xy3(f, x, y, z, h, h), f_xz3(f, x, y, z, h, h)],
                        [f_xy3(f, x, y, z, h, h), f_xx3(f,z,y,z,h,variable = 1), f_yz3(f, x, y, z, h, h)],
                        [f_xz3(f, x, y, z, h, h),  f_yz3(f, x, y, z, h, h), f_xx3(f,z,y,z,h,variable = 2)]])
    return hessian

def newton3(f,x,y,z,h,threshold = 1e-4):
    xn = np.array([x,y,z])
    hessian = H3(f,xn[0],xn[1],xn[2],h)
    # print(hessian)
    delf = np.array([f_x3(f,xn[0],xn[1],xn[2],h), f_x3(f,xn[0],xn[1],xn[2],h,variable=1), f_x3(f,xn[0],xn[1],xn[2],h,variable=2)])
    xnext = xn - np.dot(np.linalg.inv(hessian),  delf)
    dx = np.abs(xn - xnext)
    # print(dx)
    count = 0

    while dx[0] > threshold and dx[1] > threshold and dx[2] > threshold:
        xn = xnext
        hessian = H3(f,xn[0],xn[1],xn[2],h)
        delf = np.array([f_x3(f,xn[0],xn[1],xn[2],h), f_x3(f,xn[0],xn[1],xn[2],h,variable=1), f_x3(f,xn[0],xn[1],xn[2],h,variable=2)])
        xnext = xn - np.dot(np.linalg.inv(hessian),  delf)
        dx = np.abs(xn - xnext)
        count += 1
        # print(dx)
    return xn,count


# def acceptancefunction3(dE, T):
#     if dE <= 0.0:
#         return 1
#     elif np.random.uniform(0,1,1) < np.exp(-1.0 * dE/T):
#         return 1
#     return 0

def ProposalFunction3(u,sigma_theta,sigma_dm, sigma_a):
    x = np.random.normal(0.0, sigma_theta) + u[0]
    y = np.random.normal(0.0, sigma_dm) + u[1]
    z = np.random.normal(0.0, sigma_a) + u[2]
    return np.array([x,y,z])

def runchain3(nll,theta0, dm0, a0, sim, data):
    T = 3
    temps = np.arange(T, 0.0, -0.25) #this should not be 0
    iterations = int(50000) #iterations per temperature
    
    values = [theta0, dm0, a0]
    
    listofvalues = []
    
    for T in temps:
        zeros = np.zeros((iterations+1,len(values)))
        zeros[0,:] = values 
        print('Currently runnin on temp =', T)
        for j in range(iterations):
            unow = zeros[j,:]
            # print(unow)
            unext = ProposalFunction3(unow, 0.1, 0.5e-3, 0.1) #change sigmas here
            nllnow = nll3(unow[0], unow[1], unow[2])
            # print(unext)
            # print(type(unext))
            nllnext = nll3(unext[0], unext[1], unext[2])
            dE = nllnext - nllnow
            acceptstep = acceptancefunction(dE,T)
            if acceptstep == 1:
                zeros[j+1,:] = unext
            if acceptstep == 0:
                zeros[j+1,:] = unow
        values = zeros[-1,:]
        print(values)
        listofvalues.append(zeros)
    return listofvalues


'univariate in 3D'
def minimize_theta3(theta, dm, a0, sim, data, extra = 0, parabminimiserthreshold = 1e-7):
    ''' 
    these are here to find the index of the lowest value to be used as a guess
    '''
    thetas = np.linspace(0, 2*np.pi/4, 1000)
    dms = np.array([dm] * len(thetas))
    a = np.array([a0] * len(thetas))
    
    nlltheta = [nll3(thetas[i], dms[i], a[i]) for i in range(len(thetas))]  

    theta_guess = thetas[nlltheta.index(min(nlltheta))]
    def nll_theta3(theta, dm_val = dm, a_val=a0): #defining NLL just for theta varying to use in minimiser
        nllvalue = nll3(theta, dm_val, a_val)
        return nllvalue
    # print(theta_guess)
    theta_min ,count,dx,x_latest,y_latest = parabmin(nll_theta3, theta_guess-np.pi/1000, theta_guess +np.pi/1000, parabminimiserthreshold) # 
    
    if extra == 0:
        # plt.clf()
        # plt.plot(thetas, nlltheta,'-', label = r'fixed $dm $ =  %s ' % dms[0])
        # plt.axvline(x = theta_min, linestyle = '-.',label = r'$thetamin$ from parabolic search',  )
        # plt.title(r'NLL($\theta$)')
        # plt.ylabel(r'NLL($\theta$)')
        # plt.xlabel(r'\theta')
        # plt.legend()
        # plt.show()
        # plt.xlim(0.002,0.0028)
        return [theta_min, dm, a0]
    if extra == 1:
        print(r'Value of theta_min from parabolic optimisation = %a, \
          Number of iterations = %a  \
              last change in x = %a  \
                  Last 3 x values in search = %a \
                      last 3 y values in search = %a \
                          theta_guess from min = %a'
              % (theta_min, count, dx, x_latest,y_latest, theta_guess)) 
        return theta_min,count,dx,x_latest,y_latest
    
def minimize_dm3(theta, dm, a0, sim, data, extra = 0, parabminimiserthreshold = 1e-7):
    ''' 
    these are here to find the index of the lowest value to be used as a guess
    '''
    dms = np.linspace(0,0.02,1000)
    thetas = np.array([theta] * len(dms))
    a = np.array([a0] * len(dms))
    
    nlldm = [nll3(thetas[i], dms[i], a[i]) for i in range(len(dms))]  

    dm_guess = dms[nlldm.index(min(nlldm))]
    def nll_dm3(dm, theta_val = theta, a_val = a0): #defining NLL just for theta varying to use in minimiser
        return nll3(theta_val, dm, a_val)

    dm_min ,count,dx,x_latest,y_latest = parabmin(nll_dm3, dm_guess-0.0025/5, dm_guess+0.0025/4, parabminimiserthreshold) # 
    
    if extra == 0:
        # plt.clf()
        # plt.plot(thetas, nlltheta,'-', label = r'fixed $dm $ =  %s ' % dms[0])
        # plt.axvline(x = theta_min, linestyle = '-.',label = r'$thetamin$ from parabolic search',  )
        # plt.title(r'NLL($\theta$)')
        # plt.ylabel(r'NLL($\theta$)')
        # plt.xlabel(r'\theta')
        # plt.legend()
        # plt.show()
        # plt.xlim(0.002,0.0028)
        return [theta, dm_min, a0]
    if extra == 1:
        print(r'Value of theta_min from parabolic optimisation = %a, \
          Number of iterations = %a  \
              last change in x = %a  \
                  Last 3 x values in search = %a \
                      last 3 y values in search = %a \
                          theta_guess from min = %a'
              % (dm_min, count, dx, x_latest,y_latest, dm_guess)) 
        return dm_min,count,dx,x_latest,y_latest

def minimize_a3(theta, dm, a0, sim, data, extra = 0, parabminimiserthreshold = 1e-7):
    ''' 
    these are here to find the index of the lowest value to be used as a guess
    '''
    a = np.linspace(0.1, 2, 1000)
    dms =  np.array([dm] * len(a))
    thetas = np.array([theta] * len(a))

    nlla = [nll3(thetas[i], dms[i], a[i]) for i in range(len(a))]  

    a_guess = a[nlla.index(min(nlla))]
    def nll_a3(a,theta_val = theta, dm_val = dm): #defining NLL just for theta varying to use in minimiser
        return nll3(theta_val, dm_val, a)

    a_min ,count,dx,x_latest,y_latest = parabmin(nll_a3, a_guess-0.0025/5, a_guess+0.0025/4, parabminimiserthreshold) # 
    
    if extra == 0:
        # plt.clf()
        # plt.plot(thetas, nlltheta,'-', label = r'fixed $dm $ =  %s ' % dms[0])
        # plt.axvline(x = theta_min, linestyle = '-.',label = r'$thetamin$ from parabolic search',  )
        # plt.title(r'NLL($\theta$)')
        # plt.ylabel(r'NLL($\theta$)')
        # plt.xlabel(r'\theta')
        # plt.legend()
        # plt.show()
        # plt.xlim(0.002,0.0028)
        return [theta, dm, a_min]
    if extra == 1:
        print(r'Value of theta_min from parabolic optimisation = %a, \
          Number of iterations = %a  \
              last change in x = %a  \
                  Last 3 x values in search = %a \
                      last 3 y values in search = %a \
                          theta_guess from min = %a'
              % (a_min, count, dx, x_latest,y_latest, a_guess)) 
        return a_min,count,dx,x_latest,y_latest     
     
def univariate3(theta, dm, a, sim, data, first = 0, threshold = 1e-4):
    count = 0 #clockwise theta -> dm -> a minimisation
    if first == 0:    
        u0 = [theta,dm,a]
        u1 = minimize_theta3(*u0,sim,data)
      
        
        while abs(u0[0] - u1[0]) > threshold or abs(u0[1] - u1[1]) > threshold or abs(u0[2] - u1[2]) > threshold:
            print(f'Number of clockwise minimisations of all parameters = {count}')
            u0 = u1
            u1 = minimize_dm3(*u1,sim,data)
            u1 = minimize_a3(*u0,sim,data)
            u1 = minimize_theta3(*u1,sim,data)
            count += 1
          
            if count == 50:
                break
        return u1,count
    if first == 1: #anticlockwise
        u0 = [theta,dm,a]
        u1 = minimize_a3(*u0,sim,data)
        print(u0,u1)
        while abs(u0[0] - u1[0]) > threshold or abs(u0[1] - u1[1]) > threshold or abs(u0[2] - u1[2]) > threshold:
            print(f'Number of anticlockwise minimisations of all parameters = {count}')
            u0 = u1
            u1 = minimize_dm3(*u1,sim,data)
            u1 = minimize_theta3(*u1,sim,data)
            u1 = minimize_a3(*u1,sim,data)
            count += 1
    
            if count == 50:
                break
        return u1,count
    
    
def nllerror3(theta_min, dm_min, a_min, thetafactor = 0.04):
    def nllroot_theta(theta):
        b = NLL_theta(theta_min) + 0.5 
        return NLL_theta(theta) - b
        
    def NLL_theta(theta, dm = dm_min, a = a_min): #defining NLL just for theta varying to use in minimiser
        # dm = 2.4e-3
        u = [theta,dm,a_min]
        nllvalue = nll3(*u,sim,data)
        return nllvalue
    
    leftroot_theta, leftcount = bisection(nllroot_theta, 0.710, theta_min, tol = 1e-9)
    leftdiff_theta = abs(leftroot_theta - theta_min)
    rightroot_theta, rightcount = bisection(nllroot_theta, theta_min, theta_min + thetafactor, tol = 1e-9)
    rightdiff_theta = abs(rightroot_theta - theta_min)
    
    def NLL_dm(dm, theta_val = theta_min, a_val = a_min): #defining NLL just for dm varying to use in minimiser
        u = [theta_val,dm,a_val]
        nllvalue = nll3(*u,sim,data)
        return nllvalue
            
    def nllroot_dm(dm):
        b = NLL_dm(dm_min) + 0.5
        return NLL_dm(dm) - b
        
    leftroot_dm, leftcount = bisection(nllroot_dm, 0.0022, dm_min, tol = 1e-9)
    leftdiff_dm = abs(leftroot_dm - dm_min)    
    rightroot_dm, rightcount = bisection(nllroot_dm, dm_min,0.003, tol = 1e-9)
    rightdiff_dm = abs(rightroot_dm - dm_min)
    
   
    def NLL_a(a, dm_val = dm_min , theta_val = theta_min): #defining NLL just for dm varying to use in minimiser
        u = [theta_val,dm_val,a]
        nllvalue = nll3(*u,sim,data)
        return nllvalue
            
    def nllroot_a(a):
        b = NLL_a(a_min) + 0.5
        return NLL_a(a) - b
        
    leftroot_a, leftcount = bisection(nllroot_a, 0.7, a_min, tol = 1e-9)
    leftdiff_a = abs(leftroot_a - a_min)    
    rightroot_a, rightcount = bisection(nllroot_a, a_min, 1.4, tol = 1e-9)
    rightdiff_a = abs(rightroot_a - a_min)


    return [rightdiff_theta,leftdiff_theta,rightdiff_dm,leftdiff_dm,rightdiff_a,leftdiff_a]