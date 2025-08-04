# animations.jl 

# Imports 
using Plots
using FileIO

# Create animations folder if it doesn't exist
const ANIM_FOLDER = "animations"   # creates an "animations" folder in the current directory
mkpath(ANIM_FOLDER)                # ensure the folder exists

function animate_voltage(sol, filename="network.gif", fps=20)
    fullpath = joinpath(ANIM_FOLDER, filename)
    anim = @animate for t_idx in 1:10:length(sol.t)
        plot(sol.t[1:t_idx], sol[1,1:t_idx], label="Neuron 1", lw=2)
        for i in 2:N
            plot!(sol.t[1:t_idx], sol[i,1:t_idx], label="Neuron $i", lw=2)
        end
        xlabel!("Time"); ylabel!("Voltage"); title!("Voltage Dynamics")
    end
    gif(anim, fullpath, fps=fps)
end

function animate_phase_portrait(sol, filename="phase_portrait.gif", fps=20)
    fullpath = joinpath(ANIM_FOLDER, filename)
    anim = @animate for t_idx in 1:10:length(sol.t)
        plt = plot()
        for i in 1:N
            plot!(sol[i, t_idx], sol[N + i, t_idx], label="Neuron $i", color=colors[i])
        end
        xlabel!("v"); ylabel!("w"); title!("Phase Portrait at t=$(round(sol.t[t_idx], digits=1))")
    end
    gif(anim, fullpath, fps=fps)
end

function animate_raster(sol, filename="raster_plot.gif", fps=20)
    fullpath = joinpath(ANIM_FOLDER, filename)
    anim = @animate for t_idx in 1:10:length(sol.t)
        spike_times = [Float64[] for _ in 1:N]
        for i in 1:N
            if sol[i, t_idx] > threshold
                push!(spike_times[i], sol.t[t_idx])
            end
        end
        plt = plot()
        for i in 1:N
            scatter!(spike_times[i], fill(i, length(spike_times[i])), markersize=2, label="Neuron $i")
        end
        xlabel!("Time"); ylabel!("Neuron"); title!("Raster Plot at t=$(round(sol.t[t_idx], digits=1))")
    end
    gif(anim, fullpath, fps=fps)
end

function animate_power_spectrum(sol, i, filename="power_spectrum.gif", fps=20)
    fullpath = joinpath(ANIM_FOLDER, filename)
    anim = @animate for t_idx in 1:10:length(sol.t)
        v = sol[i, 1:t_idx] .- mean(sol[i, 1:t_idx])
        freqs = fftfreq(length(v), sol.t[2] - sol.t[1])
        P = abs.(fft(v)).^2
        plot(freqs[1:end÷2], P[1:end÷2], xlabel="Hz", ylabel="Power", title="Power Spectrum of Neuron $i at t=$(round(sol.t[t_idx], digits=1))")
    end
    gif(anim, fullpath, fps=fps)
end

function animate_order_parameter(sol, filename="order_parameter.gif", fps=20)
    fullpath = joinpath(ANIM_FOLDER, filename)
    anim = @animate for t_idx in 1:10:length(sol.t)
        phases = angle.(sol[1:N, t_idx]' .+ im)  # crude phase approximation
        R = abs(sum(exp.(im .* phases)) / N)
        plot(sol.t[1:t_idx], R, lw=2, label="Kuramoto R", xlabel="Time", ylabel="R(t)", title="Order Parameter at t=$(round(sol.t[t_idx], digits=1))")
    end
    gif(anim, fullpath, fps=fps)
end


