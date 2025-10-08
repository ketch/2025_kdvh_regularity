#!/usr/bin/env python
# encoding: utf-8
r"""
Solitary wave formation in Lagrangian gas dynamics
===========================================================

Solve the one-dimensional p-system with a skew-symmetric source term:

.. math::
    V_t - u_x & = u \\
    u_t + p(V,s(x))_x & = (p(V)-pstar).

Here `V` is the specific volume, `p` is the pressure, `s(x)` is the entropy,
and u is the velocity.
We take the equation of state :math:`p = p_* e^{S(x)} (V_*/V)^\gamma`.

Here we just take s(x)=0.
This system was suggested by Roberto Natalini.
It is similar to a system studied by Majda, Rosales, & Schonbeck.

"""
import numpy as np

gamma = 1.4
Vstar=1.
pstar=1.
 
def qinit(state,amp=1.0,xupper=600.,width=5):
    x = state.grid.x.centers
    S = state.aux[0,:]
    
    # Gaussian
    p = pstar+amp*np.sin(2*np.pi*x/10)+0.5#np.exp(-(x/width)**2.) + 0.5
    state.q[0,:] = Vstar*(p/pstar * np.exp(-S))**(-1./state.problem_data['gamma'])
    state.q[1,:] = 0.


def setaux(x):
    aux = np.zeros([1,len(x)],order='F')
    return aux

def step_source(solver,state,dt):
    dt2 = dt/2.

    q = state.q

    qstar = np.empty(q.shape)

    V = q[0,:]
    press = pstar*(Vstar/V)**gamma
    qstar[0,:] = q[0,:] + dt2 * q[1,:]
    qstar[1,:] = q[1,:] + dt2 * (press-pstar)

    V = qstar[0,:]
    press = pstar*(Vstar/V)**gamma
    q[0,:] = q[0,:] + dt * qstar[1,:]
    q[1,:] = q[1,:] + dt * (press-pstar)


    
def setup(use_petsc=0,solver_type='classic',outdir='./_output',
            tfinal=500.,nout=100, mx=400, amp=1.0, L=20.,use_source=True):
    from clawpack import riemann

    if use_petsc:
        import clawpack.petclaw as pyclaw
    else:
        from clawpack import pyclaw

    rs = riemann.psystem_fwave_1D

    if solver_type=='sharpclaw':
        solver = pyclaw.SharpClawSolver1D(rs)
        solver.char_decomp=0
    else:
        solver = pyclaw.ClawSolver1D(rs)
        if use_source:
            solver.step_source = step_source
            solver.source_split = 1

    solver.bc_lower[0] = pyclaw.BC.periodic
    solver.bc_upper[0] = pyclaw.BC.periodic
    solver.aux_bc_lower[0] = pyclaw.BC.periodic
    solver.aux_bc_upper[0] = pyclaw.BC.periodic
    xlower = -L/2.
    xupper = L/2.


    x = pyclaw.Dimension(xlower,xupper,mx,name='x')
    domain = pyclaw.Domain(x)
    state = pyclaw.State(domain,solver.num_eqn,num_aux=1)

    #Set global parameters
    state.problem_data = {}
    state.problem_data['t1']    = 10.0
    state.problem_data['gamma'] = 1.4
    state.problem_data['pstar'] = 1.0
    state.problem_data['vstar'] = 1.0

    #Initialize q and aux
    xc=state.grid.x.centers
    state.aux[:,:] = setaux(xc)
    qinit(state,amp=amp,xupper=xupper)


    solver.max_steps = 5000000
    solver.fwave = True 

    claw = pyclaw.Controller()
    claw.output_style = 1
    claw.num_output_times = nout
    claw.tfinal = tfinal
    claw.write_aux_init = True
    claw.solution = pyclaw.Solution(state,domain)
    claw.solver = solver
    claw.outdir = outdir
    claw.keep_copy = True

    return claw


if __name__=="__main__":
    from clawpack.pyclaw.util import run_app_from_main
    output = run_app_from_main(setup)
