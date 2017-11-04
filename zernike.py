""" defines the zernike polyonimals as used for all AO algorithm


it follows the normalization convention of
https://en.wikipedia.org/wiki/Zernike_polynomials

i.e. the first modes are

z_nm(0,0) = 1
z_nm(1,-1) = 2r cos(phi)
z_nm(1,1) = 2r sin(phi)
z_nm(2,0) = sqrt(3)(2 r^2 - 1)
...

with the orthogonality as

\int z_nm z_n'm' = pi delta_nn' delta_mm'



"""


import numpy as np
from scipy.special import binom

def norm_zernike(n, m):
    """the norm of the zernike mode n,m in born/wolf convetion


    i.e. sqrt( \int | z_nm |^2 )
    """
    return np.sqrt((1.+(m==0))*np.pi/(2.*n+2))


def norm_zernike_noll(i):
    """the norm of the zernike mode of noll index i in born/wolf convetion

    i.e. sqrt( \int | z_i |^2 )

    """
    n,m = noll_to_nm(i)
    return norm_zernike(n,m)

    # eps = 1.+1*(m==0)
    # return 2.*(n+1.)/eps/np.pi



def zernike(n,m, rho, theta, normed = True):
    """returns the zernike polyonimal by classical n,m enumeration

    if normed=True, then they form an orthonormal system

        \int z_nm z_n'm' = delta_nn' delta_mm'

        and the first modes are

        z_nm(0,0)  = 1/sqrt(pi)*
        z_nm(1,-1) = 1/sqrt(pi)* 2r cos(phi)
        z_nm(1,1)  = 1/sqrt(pi)* 2r sin(phi)
        z_nm(2,0)  = 1/sqrt(pi)* sqrt(3)(2 r^2 - 1)
        ...
        z_nm(4,0)  = 1/sqrt(pi)* sqrt(5)(6 r^4 - 6 r^2 +1)
        ...

    if normed =False, then they follow the Born/Wolf convention
        (i.e. min/max is always -1/1)

        \int z_nm z_n'm' = (1.+(m==0))/(2*n+2) delta_nn' delta_mm'

        z_nm(0,0)  = 1
        z_nm(1,-1) = r cos(phi)
        z_nm(1,1)  =  r sin(phi)
        z_nm(2,0)  = (2 r^2 - 1)
        ...
        z_nm(4,0)  = (6 r^4 - 6 r^2 +1)


    """
    if abs(m)>n:
        raise ValueError(" |m| <= n ! ( %s <= %s)"%(m,n))

    if (n-m)%2==1:
        return 0*rho+0*theta
    
    radial = 0
    m0 = abs(m)
    for k in range((n-m0)//2+1):
        radial += (-1.)**k*binom(n-k,k)*binom(n-2*k,(n-m0)//2-k)*rho**(n-2*k)

    radial *= (rho<=1.)

    if normed:
        prefac = 1./norm_zernike(n,m)
    else:
        prefac = 1.
    if m>=0:
        return prefac*radial*np.cos(m0*theta)
    else:
        return prefac*radial*np.sin(m0*theta)
            
def noll_to_nm(j):
    n = int(np.sqrt(2*j-1) + 0.5) - 1
    s = n%2
    m_even = 2 * int((2*j + 1 - n*(n+1)) / 4)
    m_odd = 2 * int((2*(j+1) - n*(n+1)) / 4) - 1
    m  = (m_odd*s + m_even*(1-s))*(1 - 2*(j%2))
    return n,m

def zernike_noll(i,rho,theta, normed = True):
    """returns the zernike polyonimal by noll index

    if normed=True, then  they are normalized s.t. they are orthogonal and of equal normal

        \int z_i z_j = pi delta_ij

    if normed =False, then they follow the Born/Wolf convention

        \int z_i z_j = (1.+(m==0))/(2*n+2) delta_ij

        with m,n = noll_to_nm(i)

    """
    n,m = noll_to_nm(i)
    return zernike(n,m,rho,theta, normed=normed)


def _dot_coeff_noll(i):
    n,m = noll_to_nm(i)
    eps = 1.+1*(m==0)
    return 2.*(n+1.)/eps/np.pi



def coeff_noll(out,i,rho,theta):
    pass


if __name__ == '__main__':

    
    # for i in range(1,12):
    #     print noll_to_nm(i)

    x = np.linspace(-1,1,512)
    X,Y = np.meshgrid(x,x)

    rho = np.hypot(X,Y)
    theta = np.arctan2(Y,X)

    Nz = 10
    zerns = [zernike_noll(i+1,rho,theta) for i in range(Nz)]

    dx = x[1]-x[0]

    cs = np.random.uniform(-1,1,10)
    y = np.sum([c*z for c,z in zip(cs,zerns)],axis=0)


    cs2 = [dx**2*np.sum(y*z)/np.pi for z in zerns]