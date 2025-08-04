
## FitzHugh-Nagumo (Morning Oscillator) and Goldebeter (Molecular Clock) Oscillator Models
# ---------------------------------------------------------------------------------------------------
# This code defines two oscillator models: 
# (1) FitzHugh-Nagumo for neuronal dynamics of the circadian clock network
# (2) Goldbeter for transcription-translation dynamics of the the PER protein in the molecular clock. 
# ---------------------------------------------------------------------------------------------------
# Imports 
# %%
using DifferentialEquations
# ---------------------------------------------------------------------------------------------------
# (1) FitzHugh-Nagumo Morning (M) Oscillator Model
# ---------------------------------------------------------------------------------------------------
# %%

# (1.a.) FHN Parameter Declaration 
N = 1        # mumber of neurons 
a = 0.7      # threshold parameter (affects nullcline shape, controls spiking)
b = 0.8      # recovery parameter (affects timescale of w, the recovery variable)
ε₀ = 0.08    # timescale separation (ε << 1 ⇒ fast-slow dynamics); base value at T_ref
I_ext = 0.5  # input current from outside the cell (constant input to all neurons if N > 1)
α = 5.0      # slope of the coupling sigmoid (steepness of interaction)
v₀ = a       # Resting potential / midpoint = threshold

# Hypothesized connectivity matrix C_T 
C_T = [0.0 0.0 1.0 0.0;    # M (pre; j), receives input from ICN1
	     0.0 1.0 0.0 1.0;    # E, receive input from itself and ICN2 
	     1.0 0.0 0.0 0.0;    # ICN1, receives input from M
	     1.0 1.0 0.0 0.0]    # ICN2, receives input from M and E 
	     # M  E ICN1 ICN2 (post; i)

# Placeholder for connectome-derived matrix (if added later)
C_E = []  # To be filled with empirical data later

# Set connectivity
C = C_T

# Label oscillator nodes
# M = Morning neuron oscillator, E = Evening neuron oscillator, ICN = Interclock neuron oscillator 
osc_labels = [
    (name = "M", color = :crimson),
    (name = "E", color = :dodgerblue),
    (name = "ICN1", color = :deeppink),
    (name = "ICN2", color = :indigo),
]

# Sigmoid
function σ(v, α, v₀)
    1 / (1 + exp(-α * (v - v₀)))
end

# (1.b.) FHN Function  
function fitzhugh_nagumo!(du, u, p, t)
    # Unpack parameters
    N, a, b, ε₀, I_ext, α, v₀, C = p 
    N = div(length(u), 2)            # Total number of FHN nodes should be 1/2 the length of u 
    
    # Decompose state vector u: [v₁,...,vₙ, w₁,...,wₙ]
    v = @view u[1:N]
    w = @view u[N+1:2N]
    dv = @view du[1:N]
    dw = @view du[N+1:2N]

    # Compute the coupling term, potential gradient, and recovery dynamics
    for i in 1:N
        coupling = sum(C[i, j] * σ(v[j], α, v₀) for j in 1:N)
        dv[i] = v[i] - v[i]^3 / 3 - w[i] + I_ext + coupling
        dw[i] = ε₀ * (v[i] - b * w[i])
    end
end

# (1.c.) Run the FitzHugh-Nagumo model
using Plots
u0 = [-1.0, 1.0]                                         # Initial conditions for v and w
tspan = (0.0, 200.0)                                     # Time span for the simulation
p = (N, a, b, ε₀, I_ext, α, v₀, C)                       # Parameters for the model
prob = ODEProblem(fitzhugh_nagumo!, u0, tspan, p)        # Define the ODE problem
sol = solve(prob, Tsit5(), saveat=0.1)                   # Solve the ODE problem with Tsit5 method

# Plot solutions v(t) and w(t) evolving in time 
plot(sol, label=["v(t)" "w(t)"], xlabel="Time", ylabel="State")

# Animate v(t)
anim = @animate for i in 1:length(sol.t)
    plot(sol.t[1:i], sol[1, 1:i], xlabel="Time", ylabel="v(t)",
         title="FitzHugh-Nagumo: Voltage over Time", legend=false,
         xlim=(0, maximum(sol.t)), ylim=(-2, 2))
end

# Save or display the animation
gif(anim, "fhn_v_animation.gif", fps=20)


# --------------------------------------------------------------------------------------------------
# (2) Goldebeter Molecular Clock Oscillator Model
# --------------------------------------------------------------------------------------------------
# %%
# (2.a.) Goldbeter Parameter Declaration 
v_s = 1.21        # max rate at which per (mRNA) accumulates in cytosol [1 μM/hr]
v_m = 0.65        # max rate at which per (mRNA) is degraded [1 μM/hr]
K_m = 0.50        # Michalis constant for mRNA degredation [unitless]
k_s = 0.83        # rate of synthesis of PER protein [1 μM/hr]
v_d = 2.50         # max rate at which fully phosphorylated PER is degraded [1 μM/hr]
k_1 = 1.90         # rate constant that that describes the transport of phosphorylated PER into the nucleus [1/hr]
k_2 = 1.30         # rate constant that that describes the transport of nuclear PER into the cytosol [1/hr]
k_I = 1.00         # threshold constant for repression [μM]
Kd = 0.20          # Michaelis constant for PER degredation
n = 4.00           # Hill coefficient; measures the ultrasensitivity (sigmoidality) of a Goldbeter–Koshland module [unitless]

# Maximum rate constants of kinases and phosphates involved in phosphorylation and desphosphorylation
K1 = 2.00         # [μM]
K2 = 2.00         # [μM]
K3 = 2.00         # [μM]
K4 = 2.00         # [μM]

# Maximum Michaelis constants for the kinases and phosphates involved in phosphorylation and desphosphorylation
V1 = 3.20         # [1 μM/hr]
V2 = 1.58         # [1 μM/hr]
V3 = 5.00         # [1 μM/hr]
V4 = 2.50         # [1 μM/hr]

# (2.b.) Goldbeter Function
function goldbeter_per!(du, u, p, t)
    # Unpack state variables
    M, P0, P1, P2, PN = u

    # Unpack parameters
    v_s, v_m, K_I, K_m,
    k_s, V1, K1, V2, K2,
    V3, K3, V4, K4,
    k_1, k_2, v_d, Kd, n = p

    dM = du[1] = (v_s * K_I^n / (K_I^n + PN^n)) - (v_m * M / (K_m + M))
    dP0 = du[2] = (k_s * M) - (V1 * P0 / (K1 + P0)) + (V2 * P1 / (K2 + P1))
    dP1 = du[3] = (V1 * P0 / (K1 + P0)) - (V2 * P1 / (K2 + P1)) -
            (V3 * P1 / (K3 + P1)) + (V4 * P2 / (K4 + P2))
    dP2 = du[4] = (V3 * P1 / (K3 + P1)) - (V4 * P2 / (K4 + P2)) -
            (k_1 * P2) + (k_2 * PN) - (v_d * P2 / (Kd + P2))
    dPN = du[5] = (k_1 * P2) - (k_2 * PN)
end

# Initial conditions and parameters
u0 = [0.1, 0.1, 0.1, 0.1, 0.1]
tspan = (0.0, 500.0)
p = (
    v_s, v_m, K_I, K_m,
    k_s, V1, K1, V2, K2,
    V3, K3, V4, K4,
    k_1, k_2, v_d, Kd, n
)

# Problem definition and solve
prob = ODEProblem(goldbeter_per!, u0, tspan, params)
sol = solve(prob, Euler(), dt=0.01)

# Plot the solutions
plot(sol_G, label=["M(t)" "P0(t)" "P1(t)" "P2(t)" "PN(t)"], xlabel="Time", ylabel="State")

# Animate M(t) = cystolic concentration of per mRNA
anim_gold = @animate for i in 1:length(sol_G.t)
    plot(sol_G.t[1:i], sol_G[1, 1:i], xlabel="Time", ylabel="v(t)",
         title="Goldbeter (1995): Cystolic concentration of per mRNA over Time", legend=false,
         xlim=(0, maximum(sol_G.t)), ylim=(-2, 2))
end

# Save or display the animation
gif(anim_gold, "goldbeter_animation.gif", fps=20)

