import fenics as fe
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import ufl as ufl

#densidad
rho=1
#elasticidad
c=fe.Expression((('x[0]<50?100:200','x[0]<50?60:120','x[0]<50?33:66'),
                ('x[0]<50?60:120','x[0]<50?75:150','x[0]<50?22:44'),
                ('x[0]<50?33:66','x[0]<50?22:44','x[0]<50?15:30')),degree=0)

#dominio
lx=100
ly=100

#numero elementos
nx=50
ny=50

#Mallado
mesh = fe.RectangleMesh(fe.Point(0,0), fe.Point(lx, ly), nx, ny)

#Espacios de funciones

V=fe.VectorFunctionSpace(mesh,'P',1)
u_tr=fe.TrialFunction(V)
v=fe.TestFunction(V)
W=fe.TensorFunctionSpace(mesh,'P',1)
H=fe.FunctionSpace(mesh,'P',1)

#funciones y clases
def deformacionAvector(ten):
    return fe.as_vector([ten[0,0],ten[1,1],2*ten[1,0]])

def esfuerzoAtensor(vec):
    return fe.as_tensor(
        [[vec[0],vec[2]],
        [vec[2],vec[1]]]
    )


def fondo(x, on_boundary):
    return on_boundary

def area_margen(x,on_boundary):
    return((x[0]-0.5*lx)**2+(x[1]-0.5*ly)**2<0.1*lx)

def epsilon(u):
    eps=0.5*(fe.grad(u)+fe.grad(u).T)
    return eps

def sigma(u):
    epsi=epsilon(u)
    epsi=deformacionAvector(epsi)
    sigm=fe.dot(c,epsi)
    sigm=esfuerzoAtensor(sigm)
    print(ufl.shape(sigm))
    return sigm

#tiempo
t_0=0.0
t_1=25
t_pasos=250

t=np.linspace(t_0,t_1,t_pasos)
dt=t_1/t_pasos





#condiciones de frontera
u_D=fe.Expression(("(t<1*pi)?(sin(t/1)*sin(t/1)*sin(t/1)*2):0",
                   "(t<1*pi)?(sin(t/1)*sin(t/1)*sin(t/1)*2):0"),t=0.0,degree=0)

bc1=fe.DirichletBC(V,fe.Constant((0.0,0.0)),fondo)
bc2=fe.DirichletBC(V,u_D,area_margen)
bc=[bc1,bc2]

#Inicialización
u=fe.Function(V)
s=fe.Function(V)
u_bar=fe.Function(V)
du=fe.Function(V)
ddu=fe.Function(V)
ddu_old=fe.Function(V)
s=fe.Function(W)

#Archivo
file = fe.XDMFFile('ondaA/solution1.xdmf')

#Forma Débil
#F=fe.inner(sigma(u_tr),epsilon(v))*fe.dx + 4*rho/(dt*dt)*fe.dot(u_tr-u_bar,v)*fe.dx
#a, L = fe.lhs(F), fe.rhs(F)
F=fe.inner(sigma(u_tr),epsilon(v))*fe.dx + 4*rho/(dt*dt)*fe.dot(u_tr-u_bar,v)*fe.dx
a, L = fe.lhs(F), fe.rhs(F)


#Integracion
tiempo=0
n=0
for ti in t:
    tiempo +=dt
    u_D.t=ti
    #k=u+dt*du+0.25*dt*dt*ddu
    u_bar.assign(u+dt*du+0.25*dt*dt*ddu)

    fe.solve(a==L,u,bc)

    ddu_old.assign(ddu)
    ddu.assign(4/(dt*dt)*(u-u_bar))
    du.assign(du+0.5*dt*(ddu+ddu_old))
    #s=fe.project(sigma(u),W)
    #[s11,s12,s21,s22]=s.split(True)
    #s_1=fe.project(s11,H)
    #if(n%5==0):
        #s_1.rename("s_1","s_1")
    #s.rename("s","s")
    file.write(u, ti)
        #h=fe.project(fe.inner(u,u),H)
        #fe.plot(s11)
        #plt.savefig(str(n)+".png")
    n=n+1