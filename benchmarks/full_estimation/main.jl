using PyCall
using Test
using Plots
using SatellitePlayground
SP = SatellitePlayground
using SatelliteDynamics
using LinearAlgebra

function randomMatrix(covariance)
    ϕ = √covariance * randn(3)
    return exp(hat(ϕ))
end

""" hat(v)

    Converts a 3-element vector into a cross-product matrix.

    Arguments:
     - v:  3-element vector                   |  [3,] 

    Returns:
     - M:  A [3 × 3] skew-symmetric matrix    |  [3, 3]
"""
function hat(v)
    M = [0.0 -v[3] v[2]
        v[3] 0.0 -v[1]
        -v[2] v[1] 0.0]

    return M
end

function qErr(q₁, q₂)
    return norm((L(q₁)'*q₂)[2:4])
end

""" L(q) 

      Converts a scalar-first unit quaternion into the left-side matrix for 
    quaternion multiplication, as described in "Planning with Attiude" (Jackson)

    Arguments:
     - q:  A scalar-first unit quaternion                         |  [4,]

    Returns: 
     - M:  Left-side matrix representing the given quaternion     |  [4, 4]
"""
function L(q)
    qₛ, qᵥ = q[1], q[2:4]

    M = [qₛ -qᵥ'
        qᵥ qₛ*I(3)+hat(qᵥ)]

    return M
end

function eulerError(e1, e2)
    return acos(dot(e1, e2) / (norm(e1) * norm(e2)))
end

function epoch_to_unix_time(time)
    unix_start = Epoch("1970-01-01")
    return (time.days - unix_start.days) * 86400 + time.seconds - unix_start.seconds
end


@testset "Estimation" begin
    py"""
    import sys
    sys.path.insert(0, ".")
    """
    flight_software = pyimport("src")
    MEKF = flight_software.MEKF()

    δt = 0.1
    mekf_hist = []


    x_osc_0 = [400e3 + SatelliteDynamics.R_EARTH, 0.0, deg2rad(50), deg2rad(-1.0), 0.0, 0.0] # a, e, i, Ω, ω, M
    q0 = normalize([ 0.030, 0.502, 0.476, 0.780])
    ω0 = 0.1 * [0.3, 0.1, -0.2]
    ω0 = ω0 / norm(ω0) * deg2rad(5.0)

    x0 = SP.state_from_osc(x_osc_0, q0, ω0)

    py"""
    import sys
    sys.path.insert(0, ".")
    """
    flight_software = pyimport("src")
    pyimport("importlib").reload(flight_software)
    EKF = flight_software.EKF([x0.position; x0.velocity])

    function measure(state, env)
        nr_sun = SatelliteDynamics.sun_position(env.time)
        unix_time = epoch_to_unix_time(env.time)

        ᵇQⁿ = SP.quaternionToMatrix(state.attitude)'

        br_sun = randomMatrix(0.01) * ᵇQⁿ * normalize(nr_sun - state.position)
        br_mag = randomMatrix(0.01) * normalize(env.b)

        gps_err = Normal(0, 5000)
        gps_noise = rand(err_model, 3)
        position = state.position + gps_noise

        return (
            position=position,
            angular_velocity=state.angular_velocity,
            br_sun=br_sun,
            br_mag=br_mag,
            time=unix_time
        )
    end

    function control_law(measure)
        (r, ω, br_sun, br_mag, time) = measure
        EKF.update(r, δt)
        r_estimated = EKF.x[1:3]
        MEKF.step(r_estimated, ω, br_mag, br_sun, δt, time)

        push!(mekf_hist, [
            MEKF.attitude; MEKF.gyro_bias
        ]) 
        return zero(SP.Control)
    end

    (state_hist, time) = SP.simulate(control_law, max_iterations=1000, measure=measure, dt=δt, initial_condition=x0)
    q_err = [
        rad2deg(qErr(state_hist[i].attitude, mekf_hist[i][1:4])) for i in eachindex(state_hist)
    ]
    time /= 60

    display(
        plot(
            time,
            q_err,
            title="Quaternion Error",
            xlabel="Time (minutes)",
            ylabel="Error (degrees)",
            label=["quaternion_error" "gyro_bias"],
            legend=:topright
        )
    )

    display(
        plot(
            time[200:end],
            q_err[200:end],
            title="Quaternion Error",
            xlabel="Time (minutes)",
            ylabel="Error (degrees)",
            label=["quaternion_error" "gyro_bias"],
            legend=:topright
        )
    )
end