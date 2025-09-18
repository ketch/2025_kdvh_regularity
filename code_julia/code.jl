# Install packages
import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

# Load packages
using LinearAlgebra: UniformScaling, I, diag, diagind, mul!, ldiv!, lu, lu!, norm
using SparseArrays: sparse, issparse, dropzeros!
using Printf: @sprintf

using SummationByPartsOperators

using LaTeXStrings
using GLMakie # FIXME: change to CairoMakie for publication quality
# using CairoMakie
set_theme!(theme_latexfonts();
           fontsize = 26,
           linewidth = 3,
           markersize = 16,
           Lines = (cycle = Cycle([:color, :linestyle], covary = true),),
           Scatter = (cycle = Cycle([:color, :marker], covary = true),))

# FIXME: update or remove PrettyTables
# using PrettyTables: PrettyTables, pretty_table, ft_printf


const FIGDIR = joinpath(dirname(@__DIR__), "figures")
if !isdir(FIGDIR)
    mkdir(FIGDIR)
end


# TODO: upstream this to SummationByPartsOperators.jl
Base.iszero(D::FourierPolynomialDerivativeOperator) = all(iszero, D.coef)


#####################################################################
# Utility functions

function compute_eoc(Ns, errors)
    eoc = similar(errors)
    eoc[begin] = NaN # no EOC defined for the first grid
    for idx in Iterators.drop(eachindex(errors, Ns, eoc), 1)
        eoc[idx] = -( log(errors[idx] / errors[idx - 1]) /
                      log(Ns[idx] / Ns[idx - 1]) )
    end
    return eoc
end

change(x) = x .- first(x)



#####################################################################
# High-level interface of the equations and IMEX ode solver

rhs_stiff!(du, u, parameters, t) = rhs_stiff!(du, u, parameters.equation, parameters, t)
rhs_nonstiff!(du, u, parameters, t) = rhs_nonstiff!(du, u, parameters.equation, parameters, t)
operator(rhs_stiff!, parameters) = operator(rhs_stiff!, parameters.equation, parameters)
dot_entropy(u, v, parameters) = dot_entropy(u, v, parameters.equation, parameters)
linear_invariant(u, parameters) = linear_invariant(u, parameters.equation, parameters)


# IMEX Coefficients
"""
    ARS111(T = Float64)

First-order, type-II, globally stiffly accurate method of Ascher, Ruuth, and Spiteri (1997).
"""
struct ARS111{T} end
ARS111(T = Float64) = ARS111{T}()
function coefficients(::ARS111{T}) where T
    l = one(T)

    A_stiff = [0 0;
               0 l]
    b_stiff = [0, l]
    c_stiff = [0, l]
    A_nonstiff = [0 0;
                  l 0]
    b_nonstiff = [l, 0]
    c_nonstiff = [0, l]
    return A_stiff, b_stiff, c_stiff, A_nonstiff, b_nonstiff, c_nonstiff
end


"""
    ARS222(T = Float64)

Second-order, type-II, globally stiffly accurate method of Ascher, Ruuth, and Spiteri (1997).
"""
struct ARS222{T} end
ARS222(T = Float64) = ARS222{T}()
function coefficients(::ARS222{T}) where T
    two = convert(T, 2)
    γ = 1 - 1 / sqrt(two)
    δ = 1 - 1 / (2 * γ)

    A_stiff = [0 0 0;
               0 γ 0;
               0 1-γ γ]
    b_stiff = [0, 1-γ, γ]
    c_stiff = [0, γ, 1]
    A_nonstiff = [0 0 0;
                  γ 0 0;
                  δ 1-δ 0]
    b_nonstiff = [δ, 1-δ, 0]
    c_nonstiff = [0, γ, 1]
    return A_stiff, b_stiff, c_stiff, A_nonstiff, b_nonstiff, c_nonstiff
end

"""
    ARS443(T = Float64)

Third-order, type-II, globally stiffly accurate method of Ascher, Ruuth, and Spiteri (1997).
"""
struct ARS443{T} end
ARS443(T = Float64) = ARS443{T}()
function coefficients(::ARS443{T}) where T
    l = one(T)

    A_stiff = [0 0 0 0 0;
               0 l/2 0 0 0;
               0 l/6 l/2 0 0;
               0 -l/2 l/2 l/2 0;
               0 3*l/2 -3*l/2 l/2 l/2]
    b_stiff = [0, 3*l/2, -3*l/2, l/2, l/2]
    c_stiff = [0, l/2, 2*l/3, l/2, l]
    A_nonstiff = [0 0 0 0 0;
                  l/2 0 0 0 0;
                  11*l/18 l/18 0 0 0;
                  5*l/6 -5*l/6 l/2 0 0;
                  l/4 7*l/4 3*l/4 -7*l/4 0]
    b_nonstiff = [l/4 7*l/4 3*l/4 -7*l/4 0]
    c_nonstiff = [0, l/2, 2*l/3, l/2, l]
    return A_stiff, b_stiff, c_stiff, A_nonstiff, b_nonstiff, c_nonstiff
end

"""
    ARS443explicit(T = Float64)

Explicit part of the third-order, type-II, globally stiffly accurate method of Ascher, Ruuth, and Spiteri (1997).
"""
struct ARS443explicit{T} end
ARS443explicit(T = Float64) = ARS443explicit{T}()
function coefficients(::ARS443explicit{T}) where T
    l = one(T)

    A_nonstiff = [0 0 0 0 0;
                  l/2 0 0 0 0;
                  11*l/18 l/18 0 0 0;
                  5*l/6 -5*l/6 l/2 0 0;
                  l/4 7*l/4 3*l/4 -7*l/4 0]
    b_nonstiff = [l/4 7*l/4 3*l/4 -7*l/4 0]
    c_nonstiff = [0, l/2, 2*l/3, l/2, l]

    A_stiff = copy(A_nonstiff)
    b_stiff = copy(b_nonstiff)
    c_stiff = copy(c_nonstiff)

    return A_stiff, b_stiff, c_stiff, A_nonstiff, b_nonstiff, c_nonstiff
end

# IMEX ARK solver
# This assumes that the stiff part is linear and that the stiff solver is
# diagonally implicit.
function solve_imex(rhs_stiff!, rhs_stiff_operator, rhs_nonstiff!,
                    q0, tspan, parameters, alg;
                    dt,
                    relaxation = false,
                    callback = Returns(nothing),
                    save_everystep = false)
    A_stiff, b_stiff, c_stiff, A_nonstiff, b_nonstiff, c_nonstiff = coefficients(alg)

    s = length(b_stiff)
    @assert size(A_stiff, 1) == s && size(A_stiff, 2) == s &&
            length(b_stiff) == s && length(c_stiff) == s &&
            size(A_nonstiff, 1) == s && size(A_nonstiff, 2) == s &&
            length(b_nonstiff) == s && length(c_nonstiff) == s
    Base.require_one_based_indexing(A_stiff, b_stiff, c_stiff,
                                    A_nonstiff, b_nonstiff, c_nonstiff)

    q = copy(q0) # solution
    if save_everystep
        sol_q = [copy(q0)]
        sol_t = [first(tspan)]
    end
    y = similar(q) # stage value
    z = similar(q) # stage update value
    t = first(tspan)
    tmp = similar(q)
    k_stiff_q = similar(q) # derivative of the previous state
    k_stiff = Vector{typeof(q)}(undef, s) # stage derivatives
    k_nonstiff = Vector{typeof(q)}(undef, s) # stage derivatives
    for i in 1:s
        k_stiff[i] = similar(q)
        k_nonstiff[i] = similar(q)
    end

    # Setup system matrix template and factorizations
    W, factorization, factorizations = let
        a = findfirst(!iszero, diag(A_stiff))
        if isnothing(a)
            factor = zero(dt)
        else
            factor = a * dt
        end
        W = I - factor * rhs_stiff_operator

        if W isa UniformScaling
            # This happens if the stiff part is zero
            factorization = W
        elseif W isa SummationByPartsOperators.AbstractPeriodicDerivativeOperator
            factorization = W
        else
            factorization = lu(W)
        end

        # We cache the factorizations for different factors for efficiency.
        # Since we do not use adaptive time stepping, we will only have a few
        # different factors.
        if factorization isa SummationByPartsOperators.AbstractPeriodicDerivativeOperator
            factorizations = nothing
        else
            factorizations = Dict(factor => copy(factorization))
        end

        W, factorization, factorizations
    end

    while t < last(tspan)
        dt = min(dt, last(tspan) - t)

        # There are two possible formulations of a diagonally implicit RK method.
        # The "simple" one is
        #   y_i = q + h \sum_{j=1}^{i} a_{ij} f(y_j)
        # However, it can be better to use the smaller values
        #   z_i = (y_i - q) / h
        # so that the stage equations become
        #   q + h z_i = q + h \sum_{j=1}^{i} a_{ij} f(q + h z_j)
        # ⟺
        #   z_i - h a_{ii} f(q + h z_i) = \sum_{j=1}^{i-1} a_{ij} f(q + h z_j)
        # For a linear problem f(q) = T q, this becomes
        #   (I - h a_{ii} T z_i = a_{ii} T q + \sum_{j=1}^{i-1} a_{ij} T(q + h z_j)
        # We use this formulation and also avoid evaluating the stiff operator at
        # the numerical solutions (due to the large Lipschitz constant), but instead
        # rearrange the equation to obtain the required stiff RHS values as
        #   T(q + h z_i) = a_{ii}^{-1} (z_i - \sum_{j=1}^{i-1} a_{ij} f(q + h z_j))
        rhs_stiff!(k_stiff_q, q, parameters, t)

        # Compute stages
        for i in 1:s
            # RHS of linear system
            fill!(tmp, 0)
            for j in 1:(i - 1)
                @. tmp += A_stiff[i, j] * k_stiff[j] + A_nonstiff[i, j] * k_nonstiff[j]
            end
            # The right-hand side of the linear system formulated using the stages y_i
            # instead of the stage updates z_i would be
            #   @. tmp = q + dt * tmp
            # By using the stage updates z_i, we avoid the possibly different scales
            # for small dt.
            @. tmp = A_stiff[i, i] * k_stiff_q + tmp

            # Setup and solve linear system
            if iszero(rhs_stiff_operator) || iszero(A_stiff[i, i])
                copyto!(z, tmp)
            else
                factor = A_stiff[i, i] * dt

                if factorization isa SummationByPartsOperators.AbstractPeriodicDerivativeOperator
                    W = I - factor * rhs_stiff_operator
                    F = W
                else
                    F = let W = W, factor = factor,
                            factorization = factorization,
                            rhs_stiff_operator = rhs_stiff_operator
                        get!(factorizations, factor) do
                            fill!(W, 0)
                            W[diagind(W)] .= 1
                            @. W -= factor * rhs_stiff_operator
                            if issparse(W)
                                lu!(factorization, W)
                            else
                                factorization = lu!(W)
                            end
                            copy(factorization)
                        end
                    end
                end
                ldiv!(z, F, tmp)
            end

            # Compute new stage derivatives
            @. y = q + dt * z
            rhs_nonstiff!(k_nonstiff[i], y, parameters, t + c_nonstiff[i] * dt)
            if iszero(rhs_stiff_operator) || iszero(A_stiff[i, i])
                rhs_stiff!(k_stiff[i], y, parameters, t + c_stiff[i] * dt)
            else
                # The code below is equaivalent to
                #   rhs_stiff!(k_stiff[i], y, parameters, t + c_stiff[i] * dt)
                # but avoids evaluating the stiff operator at the numerical solution.
                @. tmp = z
                for j in 1:(i-1)
                    @. tmp = tmp - A_stiff[i, j] * k_stiff[j] - A_nonstiff[i, j] * k_nonstiff[j]
                end
                @. k_stiff[i] = tmp / A_stiff[i, i]
            end
        end

        # Update solution
        fill!(tmp, 0)
        for j in 1:s
            @. tmp += b_stiff[j] * k_stiff[j] + b_nonstiff[j] * k_nonstiff[j]
        end

        if relaxation # TODO? && (t + dt != last(tspan))
            @. y = dt * tmp # = qnew - q
            gamma = -2 * dot_entropy(q, y, parameters) / dot_entropy(y, y, parameters)
            if gamma < 0.5
                error("Relaxation parameter is too small (gamma = $gamma); consider reducing the time step size")
            end
            @. q = q + gamma * y
            t += gamma * dt
        else
            @. q = q + dt * tmp
            t += dt
        end
        if save_everystep
            push!(sol_q, copy(q))
            append!(sol_t, t)
        end
        callback(q, parameters, t)

        if any(isnan, q)
            @error "NaNs in solution at time $t" q @__LINE__
            error()
        end
    end

    if save_everystep
        return (; u = sol_q,
                  t = sol_t)
    else
        return (; u = (q0, q),
                  t = (first(tspan), t))
    end
end



#####################################################################
# General interface

abstract type AbstractEquation end
Base.Broadcast.broadcastable(equation::AbstractEquation) = (equation,)



#####################################################################
# KdV discretization

struct KdV <: AbstractEquation end

get_u(u, equations::KdV) = u

function rhs_stiff!(du, u, equation::KdV, parameters, t)
    (; D3) = parameters

    mul!(du, D3, u)
    @. du = -du

    return nothing
end

operator(::typeof(rhs_stiff!), equation::KdV, parameters) = parameters.minus_D3

function rhs_nonstiff!(du, u, equation::KdV, parameters, t)
    (; D1, tmp1) = parameters
    one_third = one(eltype(u)) / 3

    # This semidiscretization conserves the linear and quadratic invariants
    @. tmp1 = -one_third * u^2
    mul!(du, D1, tmp1)
    mul!(tmp1, D1, u)
    @. du = du - one_third * u * tmp1

    return nothing
end

function dot_entropy(u, v, equation::KdV, parameters)
    (; D1, tmp1) = parameters
    @. tmp1 = u * v
    return 0.5 * integrate(tmp1, D1)
end

function linear_invariant(u, equation::KdV, parameters)
    (; D1) = parameters
    return integrate(u, D1)
end

function setup(u_func, equation::KdV, tspan, D, D3 = nothing)
    if D isa PeriodicUpwindOperators && isnothing(D3)
        D1 = D.central
        D3 = sparse(D.plus) * sparse(D.central) * sparse(D.minus)
        minus_D3 = -D3
    elseif D isa PeriodicDerivativeOperator && D3 isa PeriodicDerivativeOperator
        D1 = D
        D3 = D3
        minus_D3 = -sparse(D3)
    elseif D isa FourierDerivativeOperator && isnothing(D3)
        D1 = D
        D3 = D^3
        minus_D3 = 0 * I - D3
    else
        throw(ArgumentError("Combination of operators not supported"))
    end

    x = grid(D1)
    u0 = u_func(tspan[1], x, equation)
    tmp1 = similar(u0)
    q0 = u0
    parameters = (; equation, D1, D3, minus_D3, tmp1)
    return (; q0, parameters)
end


#####################################################################
# KdVH discretization

struct KdVH{T} <: AbstractEquation
    τ::T
end

get_u(q, equation::KdVH) = get_qi(q, equation, 0)
function get_qi(q, equation::KdVH, i)
    N = length(q) ÷ 3
    return view(q, (i * N + 1):((i + 1) * N))
end

function rhs_stiff!(dq, q, equation::KdVH, parameters, t)
    (; D1) = parameters
    N = size(D1, 2)

    du = view(dq, (0 * N + 1):(1 * N))
    dv = view(dq, (1 * N + 1):(2 * N))
    dw = view(dq, (2 * N + 1):(3 * N))

    u = view(q, (0 * N + 1):(1 * N))
    v = view(q, (1 * N + 1):(2 * N))
    w = view(q, (2 * N + 1):(3 * N))

    τ = equation.τ
    inv_τ = inv(τ)

    if D1 isa PeriodicUpwindOperators
        # du .= -D₊ * w
        mul!(du, D1.plus, w, -1)

        # dv .= (D * v - w) / τ
        mul!(dv, D1.central, v)
        @. dv = inv_τ * (dv - w)

        # dw .= (-D₋ * u + v) / τ
        mul!(dw, D1.minus, u)
        @. dw = inv_τ * (-dw + v)
    else
        # du .= -D₊ * w
        mul!(du, D1, w)
        @. du = -du

        # dv .= (D * v - w) / τ
        mul!(dv, D1, v)
        @. dv = inv_τ * (dv - w)

        # dw .= (-D₋ * u + v) / τ
        mul!(dw, D1, u)
        @. dw = inv_τ * (-dw + v)
    end

    return nothing
end

function operator(::typeof(rhs_stiff!), equation::KdVH, parameters)
    D1 = parameters.D1
    τ = equation.τ
    inv_τ = inv(τ)

    if D1 isa PeriodicUpwindOperators
        Dm = sparse(D1.minus)
        D = sparse(D1.central)
        Dp = sparse(D1.plus)
        O = zero(D)
        jac = [O O -Dp;
               O inv_τ*D -inv_τ*I;
               -inv_τ*Dm inv_τ*I O]
        dropzeros!(jac)
        return jac
    else
        D = sparse(D1)
        O = zero(D)
        jac = [O O -D;
               O inv_τ*D -inv_τ*I;
               -inv_τ*D inv_τ*I O]
        dropzeros!(jac)
        return jac
    end
end

function rhs_nonstiff!(dq, q, equation::KdVH, parameters, t)
    (; D1, tmp) = parameters
    N = size(D1, 2)
    one_third = one(eltype(q)) / 3

    du = view(dq, (0*N+1):(1*N))
    dv = view(dq, (1*N+1):(2*N))
    dw = view(dq, (2*N+1):(3*N))

    u = view(q, (0*N+1):(1*N))

    if D1 isa PeriodicUpwindOperators
        D = D1.central
    else
        D = D1
    end

    @. tmp = -one_third * u^2
    mul!(du, D, tmp)
    mul!(tmp, D, u)
    @. du = du - one_third * u * tmp

    fill!(dv, zero(eltype(dv)))
    fill!(dw, zero(eltype(dw)))

    return nothing
end

function dot_entropy(q1, q2, equation::KdVH, parameters)
    (; D1, tmp) = parameters
    N = size(D1, 2)

    u1 = view(q1, (0 * N + 1):(1 * N))
    v1 = view(q1, (1 * N + 1):(2 * N))
    w1 = view(q1, (2 * N + 1):(3 * N))

    u2 = view(q2, (0 * N + 1):(1 * N))
    v2 = view(q2, (1 * N + 1):(2 * N))
    w2 = view(q2, (2 * N + 1):(3 * N))

    τ = equation.τ
    half = one(τ) / 2
    @. tmp = half * (u1 * u2 + τ * v1 * v2 + τ * w1 * w2)

    return integrate(tmp, D1)
end


function setup(u_func, equation::KdVH, tspan, D1, D3 = nothing)
    if !isnothing(D3)
        throw(ArgumentError("Combination of operators not supported"))
    end

    x = grid(D1)
    u0 = u_func(tspan[1], x, KdV())

    if D1 isa PeriodicUpwindOperators
        v0 = D1.minus * u0
        w0 = D1.central * v0
    else
        v0 = D1 * u0
        w0 = D1 * v0
    end

    q0 = vcat(u0, v0, w0)

    tmp = similar(u0)

    parameters = (; equation, D1, tmp)
    return (; q0, parameters)
end


#####################################################################
# KdVH1 discretization

struct KdVH1{T} <: AbstractEquation
    τ::T
end

get_u(q, equation::KdVH1) = q

function rhs_stiff!(du, u, equation::KdVH1, parameters, t)
    (; D1, inv_operator) = parameters

    if D1 isa FourierDerivativeOperator
        mul!(du, 0 * I - D1^3 * inv_operator, u)
    else
        throw(ArgumentError("Not supported"))
    end

    return nothing
end

function operator(::typeof(rhs_stiff!), equation::KdVH1, parameters)
    (; D1, inv_operator) = parameters
    τ = equation.τ
    inv_τ = inv(τ)

    if D1 isa FourierDerivativeOperator
        jac = 0 * I - D1^3 * inv_operator
        return jac
    else
        throw(ArgumentError("Not supported"))
    end
end

function rhs_nonstiff!(du, u, equation::KdVH1, parameters, t)
    (; D1, inv_operator, tmp1) = parameters
    one_third = one(eltype(u)) / 3

    if D1 isa FourierDerivativeOperator
        D = D1
    else
        throw(ArgumentError("Not supported"))
    end

    @. tmp1 = -one_third * u^2
    mul!(du, D, tmp1)
    mul!(tmp1, D, u)
    @. tmp1 = du - one_third * u * tmp1
    ldiv!(du, inv_operator, tmp1)

    return nothing
end

function dot_entropy(u1, u2, equation::KdVH1, parameters)
    (; D1, tmp1) = parameters

    if D1 isa FourierDerivativeOperator
        τ = equation.τ
        half = one(τ) / 2
        mul!(tmp1, I - τ * D1^2 + τ * D1^4, u2)
        @. tmp1 = half * u1 * tmp1
    else
        throw(ArgumentError("Not supported"))
    end

    return integrate(tmp1, D1)
end


function setup(u_func, equation::KdVH1, tspan, D, D3 = nothing)
    if !isnothing(D3)
        throw(ArgumentError("Combination of operators not supported"))
    end

    x = grid(D)
    u0 = u_func(tspan[1], x, KdV())
    τ = equation.τ

    if D isa PeriodicUpwindOperators
        D1 = D.central
        D_plus = sparse(D.plus)
        D_minus = sparse(D.minus)
        D2 = D_plus * D_minus
        D4 = D_plus^2 * D_minus^2
        inv_operator = lu(I - τ * D2 + τ * D4)
    elseif D isa FourierDerivativeOperator
        D1 = D
        inv_operator = I - τ * D^2 + τ * D^4
    else
        throw(ArgumentError("Combination of operators not supported"))
    end

    q0 = u0
    tmp1 = similar(u0)

    parameters = (; equation, D1, inv_operator, tmp1)
    return (; q0, parameters)
end



#####################################################################
# TODO: development
# function solitary_wave_setup(equation::KdV)
#     xmin = -45.0
#     xmax = +45.0
#     c = 1.2
#     return (; xmin, xmax, c)
# end
# function solitary_wave(t, x::Number, equation::KdV)
#     (; xmin, xmax, c) = solitary_wave_setup(equation)

#     A = 3 * c
#     K = sqrt(3 * A) / 6
#     x_t = mod(x - c * t - xmin, xmax - xmin) + xmin

#     return A / cosh(K * x_t)^2
# end
# function solitary_wave(t, x::AbstractVector, equation::KdV)
#     solitary_wave.(t, x, equation)
# end

gaussian(t, x::Number, equation::KdV) = 2 * exp(-0.02 * x^2)
gaussian(t, x::AbstractVector, equation::KdV) = gaussian.(t, x, equation)


#=
julia> test(N = 2^12, dt = 0.002, τ = 1.0, tspan = (0.0, 1.0))
julia> test(N = 2^14, dt = 0.002 / 5, τ = 0.1, tspan = (0.0, 1.0))
julia> test(N = 2^14, dt = 0.002 / 5, τ = 0.1, tspan = (0.0, 2.0))

=#
function test(; tspan = (0.0, 5.0),
                N = 2^10,
                accuracy_order = 7,
                alg = ARS443(),
                alg_kdvh = alg,
                dt = 0.05,
                τ = 1.0,
                kwargs...)
    # Initialization of physical and numerical parameters
    xmin = -50.0
    xmax = +50.0

    # TODO
    D1_kdvh = upwind_operators(periodic_derivative_operator;
                               derivative_order = 1, accuracy_order,
                               xmin, xmax, N)
    D1 = fourier_derivative_operator(xmin, xmax, N)

    u_kdv, u_ini = let equation = KdV()
        (; q0, parameters) = setup(gaussian,
                                   equation, tspan, D1)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt = dt, kwargs...)
        get_u(sol.u[end], equation), get_u(sol.u[begin], equation)
    end
    @show extrema(u_ini)

    u_kdvh = let equation = KdVH(τ)
        (; q0, parameters) = setup(gaussian,
                                   equation, tspan, D1_kdvh)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg_kdvh;
                               dt = dt, kwargs...)

        get_u(sol.u[end], equation)
    end

    u_kdvh1 = let equation = KdVH1(τ)
        (; q0, parameters) = setup(gaussian,
                                   equation, tspan, D1)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt = dt, kwargs...)

        get_u(sol.u[end], equation)
    end

    fig = Figure(size = (600, 450)) # default size is (600, 450)

    ax_sol = Axis(fig[1, 1]; xlabel = L"x")
    lines!(ax_sol, grid(D1), u_ini; label = L"initial data $u_0$")
    lines!(ax_sol, grid(D1), u_kdv; label = L"KdV solution $u$")
    lines!(ax_sol, grid(D1), u_kdvh; label = L"KdVH solution $u$")
    lines!(ax_sol, grid(D1), u_kdvh1; label = L"KdVH1 solution $u$")
    axislegend(ax_sol; position = :lt, framevisible = false)

    # FIXME
    # filename = joinpath(FIGDIR, "korteweg_de_vries_convergence.pdf")
    # save(filename, fig)
    # @info "Results saved to $filename"

    return fig
end
