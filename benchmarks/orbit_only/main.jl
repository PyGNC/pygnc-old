using PyCall
using Test
using Plots
using SatellitePlayground
SP = SatellitePlayground
using SatelliteDynamics
using LinearAlgebra
using Distributions

function epoch_to_unix_time(time)
    unix_start = Epoch("1970-01-01")
    return (time.days - unix_start.days) * 86400 + time.seconds - unix_start.seconds
end


begin

    δt = 1.0
    ekf_hist = []


    x_osc_0 = [400e3 + SatelliteDynamics.R_EARTH, 0.0, deg2rad(50), deg2rad(-1.0), 0.0, 0.0] # a, e, i, Ω, ω, M
    q0 = normalize([0.030, 0.502, 0.476, 0.780])
    ω0 = 0.1 * [0.3, 0.1, -0.2]
    ω0 = ω0 / norm(ω0) * deg2rad(0.0)

    x0 = SP.state_from_osc(x_osc_0, q0, ω0)

    py"""
    import sys
    sys.path.insert(0, ".")
    """
    flight_software = pyimport("src")
    pyimport("importlib").reload(flight_software)
    EKF = flight_software.EKF([x0.position; x0.velocity])

    function measure(state, env)
        # Gaussian white noise to position
        err_model = Normal(0, 1000)
        noise = rand(err_model, 3)
        return state.position + noise
    end

    function control_law(measure)
        r = measure
        EKF.update(r, δt)

        push!(ekf_hist,
            [EKF.x[1:3]; EKF.x[4:6]]
        )
        return zero(SP.Control)
    end


    env = copy(SP.default_environment)
    env.config = SP.EnvironmentConfig(
        1, 1, false, false, false, false
    )
    (state_hist, time) = SP.simulate(control_law, max_iterations=1000, measure=measure, dt=δt, initial_condition=x0, silent=true, initial_environment=env)
    ekf_hist = [
        entry for entry in ekf_hist
    ]
    r_err = [
        norm(state_hist[i].position - ekf_hist[i][1:3]) for i in eachindex(state_hist)
    ]
    r_err /= 1e3
    v_err = [
        norm(state_hist[i].velocity - ekf_hist[i][4:6]) for i in eachindex(state_hist)
    ]
    v_err /= 1e3

    time /= 60

    display(
        plot(
            time,
            r_err,
            title="Position Error",
            xlabel="Time (minutes)",
            ylabel="Error (km)"
        )
    )

    display(
        plot(
            time,
            v_err,
            title="Velocity Error",
            xlabel="Time (minutes)",
            ylabel="Error (km/s)"
        )
    )

    position_hist = [
        (state_hist[i].position-ekf_hist[i][1:3])/1e3 for i in eachindex(state_hist)
    ]
    position_hist = reduce(hcat, position_hist)
    position_hist = position_hist'

    display(
        plot(
            time,
            position_hist,
            title="Position Estimation Error",
            xlabel="Time (minutes)",
            ylabel="Position Error (km)",
            labels=["x_err" "y_err" "z_errr"],
            linecolor=["red" "green" "blue"],
        )
    )

end