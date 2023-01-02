import fenics as fe
import matplotlib.pyplot as plt
import numpy as np
import time
import os
    

#densidad
rho=1
#elasticidad
c1111='x[0]<50?100:200'
c=fe.Expression(((c1111,'x[0]<50?60:120'),
                ('x[0]<50?60:120','x[0]<50?75:150')),degree=0)
#dominio
lx=100
ly=100

#numero elementos
nx=200
ny=200

#funciones y clases

def fondo(x, on_boundary):
    return (on_boundary and (fe.near(x[0],lx) or fe.near(x[1],0) or fe.near(x[1],ly)))

def area_margen(x,on_boundary):
    return((x[0])**2+(x[1]-0.5*ly)**2<0.1*lx)

def epsilon(u):
    return 0.5*(fe.grad(u)+fe.grad(u).T)

def sigma(u):
    return fe.dot(c,epsilon(u))

#tiempo
t_0=0.0
t_1=25
t_pasos=100

t=np.linspace(t_0,t_1,t_pasos)
dt=t_1/t_pasos

#Mallado
mesh = fe.RectangleMesh(fe.Point(0,0), fe.Point(lx, ly), nx, ny)

#Espacios de funciones

V=fe.VectorFunctionSpace(mesh,'P',1)
u_tr=fe.TrialFunction(V)
v=fe.TestFunction(V)
W=fe.TensorFunctionSpace(mesh,'P',1)
H=fe.FunctionSpace(mesh,'P',1)

#condiciones de frontera
u_D=fe.Expression(("(t<1*pi)?(sin(t/1)*sin(t/1)*sin(t/1)*2):0",
                   "0"),t=0.0,degree=0)

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
file = fe.XDMFFile('onda/solution43.xdmf')

#Forma Débil
F=fe.inner(sigma(u_tr),epsilon(v))*fe.dx + 4*rho/(dt*dt)*fe.dot(u_tr-u_bar,v)*fe.dx
a, L = fe.lhs(F), fe.rhs(F)
#Integracion
tiempo=0
n=0
for ti in t:
    if tiempo>(np.pi):
        bc=bc1
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