#############################
#### Nikolas Hadjiantoni ####
## University of Birmigham ##
#############################

import meep as mp
import meep.adjoint as mpa
import numpy as np
from autograd import numpy as npa
from autograd import tensor_jacobian_product, grad
import nlopt
import math
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from scipy import special, signal
from matplotlib.backends.backend_pdf import PdfPages
import sys
from structural import sens_filter

## Define parameters
pi = np.pi
r = 1065 ## Measuring radius

mp.verbosity(0)
Si = mp.Medium(epsilon=2.72,  D_conductivity=2*pi*0.42*0.06)#3.44**0.5)
Air = mp.Medium(index=1.0)
metal = mp.metal

## Import lens radius, predifined lens height
design_region_width = int(str(sys.argv[1])[0])
design_region_height = 1.5

pml_size = 3.0
resolution = 40
design_region_resolution = int(resolution)

## Sx: domain size in x
## Sy: domain size in y
Sx = pml_size + design_region_width  #1 into pml size for support, this will be added to geometry later 
Sy = 2*pml_size + design_region_height + 6
cell_size = mp.Vector3(Sx, 0, Sy)

## Define frequencies, c=1
nf=3
frequencies = np.array([1/0.95, 1/1.0, 1/1.05])
fcen = frequencies[1] #central frequency

minimum_length = 2.2/20 # minimum length scale (microns)
eta_i = 0.50  # blueprint (or intermediate) design field thresholding point (between 0 and 1)
eta_e = 0.55  # erosion design field thresholding point (between 0.5 and 1)
eta_d = 1 - eta_e  # dilation design field thresholding point (between 0 and 1)
filter_radius = mpa.get_conic_radius_from_eta_e(minimum_length, eta_e)

pml_layers = [mp.PML(pml_size)] #set up PML

## Define excitation beam
width = 0.2
fwidth = width * fcen #width of beam in frequency
src = mp.GaussianSource(frequency=fcen,fwidth=fwidth, is_integrated=True)#frequency=1/1.666 
source = [mp.Source(src=src,
                     center=mp.Vector3(1/(2*np.pi*fcen),0,0),size=mp.Vector3(0,0,0),
                     component=mp.Hp)] 

##Define Optimisation Region
Nx = int(design_region_resolution * design_region_width)
Ny = int(design_region_resolution * design_region_height)


design_variables = mp.MaterialGrid(mp.Vector3(Nx, 0, Ny), Air, Si, grid_type="U_MEAN")
design_region = mpa.DesignRegion(
    design_variables,
    volume=mp.Volume(
        center=mp.Vector3(design_region_width/2,0,design_region_height/2+1/(2*fcen)+0.05),#adding 0.05 to account for half the PEC
        size=mp.Vector3(design_region_width,0, design_region_height),
    ),
)


def mapping(x, eta, beta):
    ## Filter to ensure minimum feature size is adhired to
    ## Filter to ensure discrete material distribuiton

    filtered_field = mpa.conic_filter(
         x.flatten(),#sen_filter.flatten(),
         filter_radius,
         design_region_width-1/resolution,
         design_region_height-1/resolution,
         design_region_resolution,
    )
    
    projected_field = mpa.tanh_projection( filtered_field.flatten(), beta, eta)

    return projected_field.flatten()#projected_field

## Define geometry and simulation

geometry = [mp.Block(center=design_region.center, size=design_region.size, material=design_variables),
            mp.Block(mp.Vector3(1/(2*fcen), 0, 1/(2*fcen)),center=mp.Vector3((Sx-pml_size)-1/(4*fcen), 0,1/(4*fcen)),material=Si), #Support
            mp.Block(mp.Vector3(Sx-pml_size, 0, 0.1),center=mp.Vector3((Sx-pml_size)/2, 0,0),material=metal), #PEC plate
            mp.Block(mp.Vector3(0.1, 0, 0.2),center=mp.Vector3(1/(2*np.pi*fcen), 0,0),material=Air),

]

kpoint = mp.Vector3()

sim = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=source, 
    default_material=Air,m=1,
    dimensions=mp.CYLINDRICAL,
    resolution=resolution)

## Define monitors

NearRegions = [
    mp.Near2FarRegion(
        center=mp.Vector3(design_region_width/2, 0,1/(fcen) + design_region_height/2),#
        size=mp.Vector3(design_region_width, 0, 0),
        weight=+1,
    )
]

points = np.arange(0,1/fcen,1/resolution)
points = points[1:]
print(points)
ob_list = []
for p in points:
    far_x = [mp.Vector3(p, 0, r)]
    FarFields = mpa.Near2FarFields(sim, NearRegions, far_x)
    ob_list.append(FarFields)

angles = np.linspace(0, 80, 10)

## Cost here is the negative of all the E-fields measured
def J1(*args):
    
    #cost function
    
    arr =args[:]
    # cost = [(npa.sum(i[0,:,1]-npa.conj(i[0,:,1])))**2 for i in arr]
    cost = [npa.sum(npa.abs(i[0,:,1]))**2 for i in arr]
    return -npa.abs(npa.sum(cost))


opt = mpa.OptimizationProblem(
    simulation=sim,
    objective_functions=[J1],
    objective_arguments=ob_list,
    design_regions=[design_region],
    frequencies=frequencies,
    decay_by=1e-5,
)

opt.plot2D(True)

evaluation_history = []
eval_f1 = []
eval_f2 = []
eval_f2_unweight = []
beta_hist = []
cur_iter = [0]
cur_iter_b = [0]

## Adjoint funcrion
def f(x, grad):
    t = x[0]  # "dummy" parameter
    v = x[1:]  # design parameters
    if grad.size > 0:
        grad[0] = 1
        grad[1:] = 0
    return t

def c(result, x, gradient, eta, beta, weight):
    print(
        "Current iteration: {};Current iter of beta: {}; current eta: {}, current beta: {}".format(
            cur_iter[0], cur_iter_b[-1], eta, beta
        )
    )

    t = x[0]  # dummy parameter
    v = x[1:] # design parameters

    f1, dJ_du0 = opt([mapping(v, eta, beta)])
    f2, dc_dv = sens_filter(mapping(v, eta, beta), Nx, Ny, minimum_length)#[mapping(v, eta, beta)], Nx, Ny)
    dJ_du0 = np.sum(dJ_du0,axis=1) 
    dj_w = 1 - weight
    dJ_du =  dj_w*dJ_du0 + dc_dv*weight 
    f0 = dj_w*f1+f2*weight

    # Backprop the gradients through our mapping function
    my_grad = np.zeros(dJ_du.shape)
    my_grad[:] = tensor_jacobian_product(mapping, 0)(v, eta, beta, dJ_du[:])
    
    # Assign gradients
    if gradient.size > 0:
        gradient[:, 0] = -1  # gradient w.r.t. "t"
        gradient[:, 1:] = my_grad.T  # gradient w.r.t. each frequency objective

    result[:] = np.real(f0) - t

    # store results
    evaluation_history.append(np.real(f0))
    eval_f1.append(np.real(f1))
    eval_f2.append(np.real(f2*weight))
    eval_f2_unweight.append(np.real(f2))
    
    if len(evaluation_history) <= 100:
        print('Evaluation', evaluation_history)
    else:
        print('Evaluation', evaluation_history[-100:])
        
    cur_iter[0] = cur_iter[0] + 1

algorithm = nlopt.LD_MMA#GD_STOGO#
n = Nx * Ny  # number of parameters

# Commented out restrictins for empty sim

# Initial guess
x = np.ones((n,)) * 0.5

# lower and upper bounds
lb = np.zeros((Nx * Ny,))
ub = np.ones((Nx * Ny,))

# insert dummy parameter bounds and variable
x = np.insert(x, 0, 0)  # our initial guess for the worst error
lb = np.insert(lb, 0, -np.inf)
ub = np.insert(ub, 0, 0)
cur_beta = 30
beta_scale = 1.05
weight = 1.5e-4
num_betas = 130#250
update_factor = 80#100
ftol = 1e-4

## Save data to one PDF
geom_plots = PdfPages('geom_plots-'+str(design_region_width)+'lam.pdf')

for iters in range(num_betas):
    solver = nlopt.opt(algorithm, n + 1)
    solver.set_lower_bounds(lb)
    solver.set_upper_bounds(ub)
    solver.set_min_objective(f)
    solver.set_maxeval(update_factor)
    solver.set_ftol_rel(ftol)
    solver.add_inequality_mconstraint(
        lambda r, x, g: c(r, x, g, eta_i, cur_beta, weight), np.array([1e-3] * nf)
    )
    x[:] = solver.optimize(x)

    # scale parameters for next optimisation
    cur_beta = beta_scale *cur_beta
    weight =  weight*1.01 
    cur_iter_b.append(cur_iter_b[-1]+1)

    print('weight', weight)
    
    # Save design to PDF
    levels = np.linspace(0,1,10)
    plt.figure(figsize=(10, 10))
    x1 = np.linspace(0, design_region_width, Nx)
    y1 = np.linspace(0, design_region_height, Ny)
    plt.contourf(y1,x1, mapping(x[1:], eta_i, cur_beta).reshape((Nx,Ny)),cmap='Greys',levels=levels)
    plt.gca().set_aspect('equal')
    geom_plots.savefig()
    plt.close()

geom_plots.close()
## Save geometry
np.savetxt('x_'+str(design_region_width)+'_lam.csv', mapping(x[1:], eta_i, cur_beta), delimiter=",")

analytics = True

if analytics:

    ## Changing sources is necessary as a bug
    ## leaves the adjoint source as default 
    ## after optimisation finishes.
    opt.sim.change_sources(source)

    ## One m=1 and one m=-1 simulation

    opt.sim = mp.Simulation(
        cell_size=mp.Vector3(Sx,0,90),
        boundary_layers=pml_layers,
        geometry=geometry,m=1,
        sources=source,dimensions=mp.CYLINDRICAL,
        default_material=Air,
        resolution=resolution)

    opt.sim1 = mp.Simulation(
        cell_size=mp.Vector3(Sx,0, 90),
        boundary_layers=pml_layers,
        geometry=geometry,
        sources=source, 
        default_material=Air,m=-1,
        dimensions=mp.CYLINDRICAL,
        resolution=resolution)

    nearfield_box = opt.sim.add_near2far(fcen, 0, 1,
                                    mp.Near2FarRegion(center=mp.Vector3(design_region_width/2, 0,design_region_height+1/(fcen)),
                                                    size=mp.Vector3(design_region_width,0,0),
                                                    weight=1),
    )
    nearfield_box1 = opt.sim1.add_near2far(fcen, 0, 1,
                                    mp.Near2FarRegion(center=mp.Vector3(design_region_width/2, 0,design_region_height+1/(fcen)),
                                                    size=mp.Vector3(design_region_width,0,0),
                                                    weight=1),
    )


    #Monitors
    vol_dft_1 = mp.Volume(center=mp.Vector3(Sx/2-pml_size, 0,24), size=mp.Vector3(Sx-pml_size, 0,44))
    dft_fields_1 = opt.sim.add_dft_fields([mp.Er, mp.Ep, mp.Ez, mp.Hr, mp.Hp],
                                    frequencies[0],0,1,
                                    where=vol_dft_1,
                                    yee_grid=True)

    vol_dft_1_1 = mp.Volume(center=mp.Vector3(Sx/2-pml_size, 0,24), size=mp.Vector3(Sx-pml_size, 0,44))
    dft_fields_1_1 = opt.sim1.add_dft_fields([mp.Er, mp.Ep, mp.Ez, mp.Hr, mp.Hp],
                                    frequencies[0],0,1,
                                    where=vol_dft_1_1,
                                    yee_grid=True)

    vol_dft_2 = mp.Volume(center=mp.Vector3(0, 0,40), size=mp.Vector3(design_region_width, 0,0))
    dft_fields_2 = opt.sim.add_dft_fields([mp.Er, mp.Ep,mp.Ez, mp.Hr, mp.Hp],
                                    frequencies[0],0,1,
                                    where=vol_dft_2,
                                    yee_grid=True)
    opt.sim.run(until=100)
    opt.sim1.run(until=100)

    pp = PdfPages('testfft'+str(design_region_width)+'lam.pdf')
    eval_f0 = [i for i in evaluation_history]

    plt.figure()
    plt.plot(eval_f0,"s", label='Total')
    plt.plot(eval_f1,"o-", label='Adjoint')
    plt.plot(eval_f2,"o-", label = 'SIMP')
    plt.grid(True)
    plt.xlabel("Iteration")
    plt.ylabel("FOM")
    plt.legend()
    pp.savefig()
    plt.close()


    plt.figure()
    plt.plot(eval_f2_unweight,"o-", label='Unweighted SIMP')
    plt.grid(True)
    plt.xlabel("Iteration")
    plt.ylabel("FOM")
    plt.legend()
    pp.savefig()
    plt.close()

    (Er,Ez,Ep,Hr,Hp) = [opt.sim.get_dft_array(dft_obj=dft_fields_1,component=c,num_freq=0) for c in [mp.Er, mp.Ez, mp.Ep, mp.Hr, mp.Hp]]
    (Er_1,Ez_1,Ep_1,Hr_1,Hp_1) = [opt.sim1.get_dft_array(dft_obj=dft_fields_1_1,component=c,num_freq=0) for c in [mp.Er, mp.Ez, mp.Ep, mp.Hr, mp.Hp]]
    (Ez_foc, Er_foc, Ep_foc) = [opt.sim.get_dft_array(dft_obj=dft_fields_2,component=c,num_freq=0) for c in [mp.Ez, mp.Er, mp.Ep]]

    plt.figure()
    new_title = '|Ez| at y='+str(r)
    plt.title(new_title)
    min_len = min([len(Ez_foc),len(Er_foc),len(Ep_foc)])
    dist = np.linspace(0, design_region_width, min_len)
    plt.plot(dist[0:], np.abs(Ez_foc[0:min_len])**2+np.abs(Er_foc[0:min_len])**2+np.abs(Ep_foc[0:min_len])**2)
    pp.savefig()
    plt.close()

    log_foc = 20*np.log10(np.abs(Ez_foc[0:min_len])**2+np.abs(Er_foc[0:min_len])**2+np.abs(Ep_foc[0:min_len])**2)
    log_foc_max = np.max(log_foc)
    log_foc = log_foc - log_foc_max 

    plt.figure()
    new_title = '20 Log |E| at y='+str(r)
    plt.title(new_title)
    plt.plot(dist[0:], log_foc)
    pp.savefig()
    plt.close()


    fig=plt.figure()
    ax = fig.add_subplot(111)
    plt.title('E')

    len_x = min([len(Ez[:,0]),len(Ep[:,0]),len(Er[:,0])])
    len_y = min([len(Ez[0,:]), len(Ep[0,:]), len(Er[0,:])]) 
    x = np.linspace(0, 44,len_x )
    y = np.linspace(0,Sx-pml_size, len_y)

    fields_sum =np.abs(Ez[:len_x,:len_y]-Ez_1[:len_x,:len_y])+np.abs(Ep[:len_x,:len_y]-Ep_1[:len_x,:len_y])+np.abs(Er[:len_x,:len_y]-Er_1[:len_x,:len_y])
    fields_sum =Ep[:len_x,:len_y]-Ep_1[:len_x,:len_y]
    print(-np.max(np.abs(fields_sum)),np.max(np.abs(fields_sum)))
    print(np.mean(np.abs(fields_sum)))
    levels = np.linspace(-1,1,50)
    plt.contourf(y,x, fields_sum ,cmap='seismic', levels=levels)
    plt.gca().set_aspect('equal')
    pp.savefig()
    plt.close()

    fig=plt.figure()
    ax = fig.add_subplot(111)

    plt.title('20*log(E)')
    logE = 20*np.log10(np.abs(fields_sum))
    max_loge = np.max(logE)
    logE = logE-max_loge
    levels = np.linspace(-90,0,100)
    for i in range(len(logE[0,:])):
        for j in range(len(logE[:,0])):
            if logE[j,i]<-60:
                logE[j,i] = -60

    plt.pcolor(y,x, logE,cmap='inferno')
    plt.colorbar(ax=ax)
    plt.gca().set_aspect('equal')
    pp.savefig()
    plt.close()

    npts = 360  # number of points in [0,2*pi) range of angles
    angles = pi/npts*np.arange(npts)
    E = np.zeros((npts,3),dtype=np.complex128)
    H = np.zeros((npts,3),dtype=np.complex128)

    for n in range(npts):
        ff = opt.sim.get_farfield(nearfield_box,
                            mp.Vector3(r*math.cos(angles[n]),0,
                                        r*math.sin(angles[n])))
        ff1 = opt.sim1.get_farfield(nearfield_box1,
                            mp.Vector3(r*math.cos(angles[n]),0,
                                        r*math.sin(angles[n])))
        #E[n,:] = [ff[j]*np.exp(1j*0)-ff1[j]*np.exp(-1j*0) for j in range(3)]
        E[n, :] = [np.conj(ff[j]) for j in range(3)]
        H[n, :] = [ff[j + 3] for j in range(3)]

    Pr = np.real(E[:, 1] * H[:, 2] - E[:, 2] * H[:, 1])
    Pz = np.real(E[:, 0] * H[:, 1] - E[:, 1] * H[:, 0])
    Prz = np.sqrt(np.square(Pr) + np.square(Pz))

    dtheta = 0.5 * pi / (npts - 1)
    dphi = 2 * pi
    flux_tot = np.sum(Prz * np.sin(angles)) * dtheta * dphi
    print('Dir', np.max(4*pi*Prz/(flux_tot)))

    np.savetxt('efarfield90'+str(design_region_width)+'.csv', E, delimiter=",")

    log = 4*pi*np.log(Prz/(flux_tot))

    ax = plt.subplot(111, projection='polar')
    plt.title('20*log_10(E) at phi=0')
    ax.plot(angles,log,'b-')
    ax.set_rmax(1)
    ax.grid(True)
    ax.set_rlabel_position(22)
    ax.set_xticks(np.linspace(0, 2*np.pi, 32, endpoint=False))
    pp.savefig()
    plt.close()

    E1 = np.zeros((npts,3),dtype=np.complex128)

    for n in range(npts):
        ff = opt.sim.get_farfield(nearfield_box,
                            mp.Vector3(r*math.cos(angles[n]),pi/2,
                                        r*math.sin(angles[n])))
        ff1 = opt.sim1.get_farfield(nearfield_box1,
                            mp.Vector3(r*math.cos(angles[n]),pi/2,
                                        r*math.sin(angles[n])))
        E1[n,:] = [ff[j]*np.exp(1j*pi/2)-ff1[j]*np.exp(-1j*pi/2) for j in range(3)]

    np.savetxt('efarfield0'+str(design_region_width)+'.csv', E1, delimiter=",")

    log1 = 20*np.log10((np.abs(E1[:,0])**2+np.abs(E1[:,1])**2+np.abs(E1[:,2])**2)**0.5)
    log_max1 = np.max(log1)
    log1 = log1-log_max1

    ax = plt.subplot(111, projection='polar')
    plt.title('20*log_10(E) at phi=pi/2')
    ax.plot(angles,log1,'b-')
    ax.set_rmax(1)
    ax.grid(True)
    ax.set_rlabel_position(22)
    ax.set_xticks(np.linspace(0, 2*np.pi, 32, endpoint=False))
    pp.savefig()
    plt.close()

    pp.close()
