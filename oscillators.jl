
## FitzHugh-Nagumo (Morning Oscillator) and Goldebeter (Molecular Clock) Oscillator Models
# ---------------------------------------------------------------------------------------------------
# This code defines two oscillator models: 
# (1) FitzHugh-Nagumo (1961) for neuronal dynamics of the circadian clock network
# (2) Goldbeter (1995) for transcription-translation dynamics of the the PER protein in the molecular clock.
# All parameters, functions, and initial conditions are defined in this file according to the original papers.
# ---------------------------------------------------------------------------------------------------
# Imports 
# %%
using DifferentialEquations, Plots, DSP, FourierAnalysis, Unitful, Dataframes
using .my_units  # Import custom units module for μM, ms, s, hr, day

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
v_s = 1.21         # max rate at which per (mRNA) accumulates in cytosol [1 μM/hr]
v_m = 0.65         # max rate at which per (mRNA) is degraded [1 μM/hr]
K_m = 0.50         # Michalis constant for mRNA degredation [unitless]
k_s = 0.83         # rate of synthesis of PER protein [1 μM/hr]
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
    # Unpack state variables/initial conditions 
    M, P0, P1, P2, PN = u

    # Unpack parameters
    v_s, v_m, K_I, K_m,
    k_s, V1, K1, V2, K2,
    V3, K3, V4, K4,
    k_1, k_2, v_d, Kd, n = p
    
    # Replace these placeholders with the actual Goldbeter equations
    du[1] = (v_s * K_I^n / (K_I^n + PN^n)) - (v_m * M / (K_m + M))        # dM/dt
    du[2] = (k_s * M) - (V1 * P0 / (K1 + P0)) + (V2 * P1 / (K2 + P1))     # dP0/dt
    du[3] = du[3] = (V1 * P0 / (K1 + P0)) - (V2 * P1 / (K2 + P1)) -       # dP1/dt
            (V3 * P1 / (K3 + P1)) + (V4 * P2 / (K4 + P2))              
    du[4] = du[4] = (V3 * P1 / (K3 + P1)) - (V4 * P2 / (K4 + P2)) -       # dP2/dt
            (k_1 * P2) + (k_2 * PN) - (v_d * P2 / (Kd + P2))
    du[5] = du[5] = (k_1 * P2) - (k_2 * PN)                               # dPN/dt
end

# Initial concentrations [μM]
M = 1.9               # per mRNA in cytosol
P0 = 0.8              # unphosphorylated PER protein
P1 = 0.8              # monophosphorylated PER protein
P2 = 0.8              # diphosphorylated PER protein
PN = 0.8              # nuclear PER protein          


u = [M, P0, P1, P2, PN]  # state vars w/ stored initial conditions
tspan = (0.0, 500.0)     # time in hours

# Parameters for the Goldbeter model 
p = (
    v_s, v_m, K_I, K_m,
    k_s, V1, K1, V2, K2,
    V3, K3, V4, K4,
    k_1, k_2, v_d, Kd, n
)

# Problem definition and solve
prob = ODEProblem(goldbeter_per!, u0, tspan, params)
sol = solve(prob, Euler(), dt=0.01)

# Extract time and variable values
time = sol.t
M_vals  = sol[1, :]
P0_vals = sol[2, :]
P1_vals = sol[3, :]
P2_vals = sol[4, :]
PN_vals = sol[5, :]
P_t_vals = P0_vals .+ P1_vals .+ P2_vals .+ PN_vals

# Plot results w/ two aces 
# Main plot (left y-axis for mRNA)
p1 = plot(
    time, M_vals,
    label = "per mRNA",
    xlabel = "Time [hours]",
    ylabel = "[mRNA] [μM]",
    color = :black,
    linewidth = 2,
    legend = :topright,
    title = "Sustained oscillations in [PER] based on negative regulation of per mRNA synthesis by the PER protein in Drosophila"
)

# Overlay (right y-axis for PER protein concentrations)
p2 = plot(
    time, P0_vals, label = "P₀ [μM]", color = :blue, linestyle = :solid
)
plot!(p2, time, P1_vals, label = "P₁ [μM]", color = :green, linestyle = :dash)
plot!(p2, time, P2_vals, label = "P₂ [μM]", color = :orange, linestyle = :dot)
plot!(p2, time, PN_vals, label = "PN [μM]", color = :purple, linestyle = :dashdot)
plot!(p2, time, P_t_vals, label = "Pₜ (total PER) [μM]", color = :red, linewidth = 2)

# Combine both with twin axes
plot(p1, twinx(p2))

# Animate M(t) = cystolic concentration of per mRNA
gr()  # or pyplot() or plotly(), but GR is usually the fastest for GIFs

# Set up the animation
anim = @animate for i in 1:5:length(time)
    t_window = time[1:i]
    
    # Extract partial data for frame i
    M_part  = M_vals[1:i]
    P0_part = P0_vals[1:i]
    P1_part = P1_vals[1:i]
    P2_part = P2_vals[1:i]
    PN_part = PN_vals[1:i]
    Pt_part = P_t_vals[1:i]

    # Left axis plot for mRNA
    p1 = plot(
        t_window, M_part,
        label = "per mRNA", xlabel = "Time [hours]",
        ylabel = "[mRNA] [μM]", color = :black,
        linewidth = 2, legend = :topright,
        title = "Sustained oscillations in [PER] in Drosophila"
    )

    # Right axis plot for PER proteins
    p2 = plot(t_window, P0_part, label = "P₀", color = :blue)
    plot!(p2, t_window, P1_part, label = "P₁", color = :green, linestyle = :dash)
    plot!(p2, t_window, P2_part, label = "P₂", color = :orange, linestyle = :dot)
    plot!(p2, t_window, PN_part, label = "PN", color = :purple, linestyle = :dashdot)
    plot!(p2, t_window, Pt_part, label = "Pₜ", color = :red, linewidth = 2)

end

gif(anim, "per_model_oscillations.gif", fps = 20)  # Higher fps = faster animation

