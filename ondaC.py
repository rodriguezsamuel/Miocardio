import fenics as fe
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import ufl as ufl

#densidad
rho=1
#dominio
lx=100
ly=100
#elasticidad
c1111='sin(pi*x[0]/(2*'+lx+'))'
c1112='x[0]<200?90:180'
c1122='x[0]<200?80:160'
c1212='x[0]<200?70:140'
c1222='x[0]<200?60:120'
c2222='x[0]<200?50:200'
c = np.zeros((2, 2, 2, 2))
c =fe.Expression([[[[c1111,c1112],[c1112,c1122]],
                [[c1112,c1212],[c1212,c1222]]],
                
                [[[c1112,c1212],[c1212,c1222]],
                [[c1122,c1222],[c1122,c2222]]]],degree=0)



#numero elementos
nx=500
ny=500

#Mallado
mesh = fe.RectangleMesh(fe.Point(0,0), fe.Point(lx, ly), nx, ny)

#Espacios de funciones

V=fe.VectorFunctionSpace(mesh,'P',1)
u_tr=fe.TrialFunction(V)
v=fe.TestFunction(V)
W=fe.TensorFunctionSpace(mesh,'P',1)
H=fe.FunctionSpace(mesh,'P',1)

#funciones y clases

def fondo(x, on_boundary):
    return on_boundary

def area_margen(x,on_boundary):
    return((x[0])**2+(x[1]-0.5*ly)**2<0.2*lx)

def epsilon(u):
    return 0.5*(fe.grad(u)+fe.grad(u).T)

def sigma(u):
    i,j,k,l=ufl.indices(4)
    eps=epsilon(u)
    #si=ufl.operators.contraction(c[i,j,k,l],(i,j),eps[i,j],(i,j))
    #sdp=c[i,j,k,l]*eps[i,j]
    return fe.as_tensor(c[i,j,k,l]*eps[i,j],(k,l))
    return si
    #return sdp

#tiempo
t_0=0.0
t_1=5
t_pasos=100

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
fileU = fe.XDMFFile('ondaB/solution10U.xdmf')
fileE = fe.XDMFFile('ondaB/solution10E.xdmf')
fileS = fe.XDMFFile('ondaB/solution10S.xdmf')

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
    s=fe.project(sigma(u),W)
    esp=fe.project(epsilon(u),W)
    esp.rename("esp","esp")
    s.rename("s","s")
    #[s11,s12,s21,s22]=s.split(True)
    #s_1=fe.project(s11,H)
    #if(n%5==0):
        #s_1.rename("s_1","s_1")
    
    fileE.write(esp, ti)
    fileS.write(s, ti)
    fileU.write(u, ti)
        #h=fe.project(fe.inner(u,u),H)
        #fe.plot(s11)
        #plt.savefig(str(n)+".png")
    n=n+1
    print(n)
fileU.close()
fileE.close()
fileS.close()