using Plots
using LinearAlgebra


format(x) = transpose(hcat(x...))
initX = 5
initY = 7.5
Rx = 12
Ry = 5

R = [12,-5, 0.1,-15]
s = [initX, initY, 0.1, 0.1]

mu = [rand(), rand(),rand(),rand()]
mdash = [rand(), rand(),rand(), rand()]
lr = 1

a = zeros(4)
sigma_s = 1
sigmamdash = 1

dynamics_model(mu, R, alpha) = alpha * (R .- mu)
gradient_dynamics_model(mu,s,R,alpha) = alpha * ((s - (0.5*(s-R))) - mu)
function run_model(s, mu, mdash, a_param, lr, a, sigma_s, sigmamdash,R)
    ess = []
    emdashs = []
    emudashs = []
    dfdmus = []
    dfdmudashes = []
    dfdas = []
    mus = []
    mdashes = []
    as = []
    ss = []
    Fs = []

    for i in 1:1000
        e_s = (s .- mu) / sigma_s
        e_mdash = (a_param / sigmamdash) * (mdash - gradient_dynamics_model(mu,s, R, a_param))

        e_mudash = -(mdash .- gradient_dynamics_model(mu,s, R, a_param))/sigmamdash

        dfdmu = e_s + e_mdash
        dfdmudash = e_mudash
        dfda = -e_s

        F = sum(e_s) + sum(e_mdash) + sum(e_mudash)

        mu += lr * (mdash + dfdmu)
        mdash += lr * dfdmudash
        a += lr * dfda
        s = s + a
        push!(ess, e_s)
        push!(emdashs, e_mdash)
        push!(emudashs, e_mudash)
        push!(dfdmus, dfdmu)
        push!(dfdmudashes, dfdmudash)
        push!(dfdas, dfda)
        push!(Fs, F)
        push!(mus, mu)
        push!(mdashes, mdash)
        push!(as, a)
        push!(ss, s)
    end
    return ess, emdashs, emudashs, dfdmus, dfdmudashes, dfdas, Fs, mus, mdashes, as,ss
end
res = run_model(s, mu, mdash, a_param, lr, a, sigma_s, sigmamdash,R)
ess, emdashs, emudashs, dfdmus, dfdmudashes, dfdas, Fs, mus, mdashes, as,ss = res
plot(format(ss))

using PyCall
using Plots
pushfirst!(PyVector(pyimport("sys")["path"]),"")


np = pyimport("numpy")
a = np.zeros((5,4))

cc = pyimport("continuous_cartpole")
print(keys(cc))

env = cc.ContinuousCartPoleEnv()
s = env.reset()
os = zeros(100, 4)
as = zeros(100,1)
for i in 1:100
    a = env.action_space.sample()
    s,r,done,info = env.step(a)
    os[i,:] = s
    as[i,:] = a
    #println(r)
    if done
        s = env.reset()
    end
end

plot(os)
plot(as)


using Plots
using LinearAlgebra
using Statistics, StatsBase

env = cc.ContinuousCartPoleEnv()
s = env.reset()

format(x) = transpose(hcat(x...))

R = zeros(4)
theta_a = ones(1,4)
mu = [rand(), rand(),rand(),rand()]
mdash = [rand(), rand(),rand(), rand()]
a_param = 0.5
lr = 1
a = zeros(1)
sigma_s = 1
sigmamdash = 1
##
dynamics_model(mu, R, alpha) = alpha * (R .- mu)
gradient_dynamics_model(mu,s,R,alpha) = alpha * ((s - (0.5*(s-R))) - mu)

function ones_run_model(env,s, mu, mdash, a_param, lr, a, sigma_s, sigmamdash,R)
    ess = []
    emdashs = []
    emudashs = []
    dfdmus = []
    dfdmudashes = []
    dfdas = []
    mus = []
    mdashes = []
    as = []
    ss = []
    Fs = []
    total_r = 0
    rs = []
    divergences = []

    for i in 1:1000
        e_s = (s .- mu) / sigma_s
        e_mdash = (a_param / sigmamdash) * (mdash - dynamics_model(mu, R, a_param))
        div = dynamics_model(mu, R,a_param)
        push!(divergences, div)
        e_mudash = -(mdash .- dynamics_model(mu, R, a_param))/sigmamdash

        dfdmu = e_s + e_mdash
        dfdmudash = e_mudash
        dfda = [sum(e_s)] #

        F = sum(e_s) + sum(e_mdash) + sum(e_mudash)

        mu += lr * (mdash + dfdmu)
        mdash += lr * dfdmudash
        a += lr * dfda
        if a[1] > 1.0
            a[1] = 1.0
        end
        if a[1] < -1.0
            a[1] = -1.0
        end
        s,r,done,info = env.step(a)
        total_r +=1
        if done
            println("Reset!")
            s = env.reset()
            push!(rs, total_r)
            total_r = 0
        end
        push!(ess, e_s)
        push!(emdashs, e_mdash)
        push!(emudashs, e_mudash)
        push!(dfdmus, dfdmu)
        push!(dfdmudashes, dfdmudash)
        push!(dfdas, dfda)
        push!(Fs, F)
        push!(mus, mu)
        push!(mdashes, mdash)
        push!(as, a)
        push!(ss, s)
        env.render()
    end
    return ess, emdashs, emudashs, dfdmus, dfdmudashes, dfdas, Fs, mus, mdashes, as,ss, rs, divergences
end
res = ones_run_model(env,s, mu, mdash, a_param, lr, a, sigma_s, sigmamdash,R)
ess, emdashs, emudashs, dfdmus, dfdmudashes, dfdas, Fs, mus, mdashes, as,ss,rs, divergences = res
plot(format(as))
rs
plot(rs)
mean(rs)


##
# LINEAR LEARNED GENERATIVE MODEL.
#### FITTING THE DSDAs with JUST ACTION.

env = cc.ContinuousCartPoleEnv()
s = env.reset()

format(x) = transpose(hcat(x...))
R = zeros(4)
theta_a = randn(4,1)
mu = [rand(), rand(),rand(),rand()]
mdash = [rand(), rand(),rand(), rand()]
a_param = 0.5
lr = 0.1
a = randn(1)
sigma_s = 1
sigmamdash = 1
sigma_theta = 1
##
dynamics_model(mu, R, alpha) = alpha * (R .- mu)
function linear_PC_run_model(env,s, mu, mdash, a_param, lr, a, sigma_s, sigmamdash,R, theta_a)
    ess = []
    emdashs = []
    emudashs = []
    dfdmus = []
    dfdmudashes = []
    dfdas = []
    mus = []
    mdashes = []
    as = []
    ss = []
    Fs = []
    total_r = 0
    rs = []
    divergences = []
    thetas = []
    e_thetas = []
    dfdthetas = []
    sprev = s
    dsdahats = []
    dfda = 0
    dsdas= []

    for i in 1:3000
        e_s = (s .- mu) / sigma_s
        e_mdash = (a_param / sigmamdash) * (mdash - dynamics_model(mu, R, a_param))
        div = dynamics_model(mu, R,a_param)
        push!(divergences, div)
        e_mudash = -(mdash .- dynamics_model(mu, R, a_param))/sigmamdash
        dmuda_hat = theta_a * a
        dfdmu = e_s + e_mdash
        dfdmudash = e_mudash!
        #print(typeof(dfda))
        #println(typeof(e_s))
        F = sum(e_s) + sum(e_mdash) + sum(e_mudash)
        prevmu = mu
        mu += lr * (mdash + dfdmu)
        dmuda = lr * (mdash + dfdmu)
        mdash += lr * dfdmudash
        dsda_hat = theta_a * a

        if a[1] > 1.0
            a[1] = 1.0
        end
        if a[1] < -1.0
            a[1] = -1.0
        end
        #println(a)
        #println("dfda: $dfda")
        s,r,done,info = env.step(a)
        dsda = s - sprev
        e_theta = (dsda - dsda_hat) / sigma_theta
        dfdtheta = -(e_s + e_theta)
        theta_a +=  10 * lr * dfdtheta
        total_r +=1
        #println(typeof((transpose(dsda_hat) * e_s)))
        #println(size((transpose(theta_a) * e_theta)))
        #dfda = [(transpose(dsda) * e_s) + (transpose(theta_a) * e_theta)[1]]
        dfda = [sum(e_s)]

        push!(dsdahats, dsda_hat)
        a += lr * dfda
        if done
            println("Reset!")
            s = env.reset()
            push!(rs, total_r)
            total_r = 0
        end
        push!(ess, e_s)
        push!(emdashs, e_mdash)
        push!(emudashs, e_mudash)
        push!(dfdmus, dfdmu)
        push!(dfdmudashes, dfdmudash)
        push!(dfdas, dfda)
        push!(Fs, F)
        push!(mus, mu)
        push!(mdashes, mdash)
        push!(as, a)
        push!(ss, s)
        push!(thetas, theta_a)
        push!(e_thetas, e_theta)
        push!(dfdthetas, dfdtheta)
        push!(dsdas, dsda)
        env.render()
    end
    return ess, emdashs, emudashs, dfdmus, dfdmudashes, dfdas, Fs, mus, mdashes, as,ss, rs, divergences, thetas, e_thetas, dfdthetas, dsdahats,dsdas
end
res = linear_PC_run_model(env,s, mu, mdash, a_param, lr, a, sigma_s, sigmamdash,R,theta_a)
ess, emdashs, emudashs, dfdmus, dfdmudashes, dfdas, Fs, mus, mdashes, as,ss,rs, divergences, thetas, e_thetas, dfdthetas,dsdahats,dsdas = res
plot(format(dfdthetas))
plot(format(rs))
plot(format(dsdas[1:100])[:,4])
plot!(format(dsdahats[1:100])[:,4])
plot(format(mus))
plot!(format(ss))
plot(format(as))
##
############ STATE AND ACTION BASED DSDA ESTIMATION...
#  basic attempt to get this working!


##
env = cc.ContinuousCartPoleEnv()
s = env.reset()

format(x) = transpose(hcat(x...))
function format2(x)
    arr = zeros(length(x), 4,4)
    for (i,a) in enumerate(x)
        arr[i,:,:] = a
    end
    return arr
end
##
R = zeros(4)
theta_a = randn(4,1)
theta_s = randn(4,4)
mu = [rand(), rand(),rand(),rand()]
mdash = [rand(), rand(),rand(), rand()]
a_param = 0.01
lr = 0.5
a = randn(1)
sigma_s = 1
sigmamdash = 1
sigma_theta = 1
##

dynamics_model(mu, R, alpha) = alpha * (R .- mu)
function linear_PC_run_model(env,s, mu, mdash, a_param, lr, a, sigma_s, sigmamdash,R, theta_a,theta_s)
    ess = []
    emdashs = []
    emudashs = []
    dfdmus = []
    dfdmudashes = []
    dfdas = []
    mus = []
    mdashes = []
    as = []
    ss = []
    Fs = []
    total_r = 0
    rs = []
    divergences = []
    theta_as = []
    theta_ss = []
    e_thetas = []
    dfdthetas = []
    sprev = s
    dsdahats = []
    dfda = 0
    dsdas= []

    for i in 1:3000
        e_s = (s .- mu) / sigma_s
        e_mdash = (a_param / sigmamdash) * (mdash - dynamics_model(mu, R, a_param))
        div = dynamics_model(mu, R,a_param)
        push!(divergences, div)
        e_mudash = -(mdash .- dynamics_model(mu, R, a_param))/sigmamdash
        dfdmu = e_s + e_mdash
        dfdmudash = e_mudash
        F = sum(e_s) + sum(e_mdash) + sum(e_mudash)
        mu += lr * (mdash + dfdmu)
        mdash += lr * dfdmudash
        dsda_hat = (theta_a * a) + (theta_s * s)
        println(size(dsda_hat .* e_s))

        if a[1] > 1.0
            a[1] = 1.0
        end
        if a[1] < -1.0
            a[1] = -1.0
        end
        s,r,done,info = env.step(a)
        s[2] = s[2] / 10.0
        s[4] = s[4] / 10.0
        dsda = s - sprev
        e_theta = (dsda - dsda_hat) / sigma_theta
        dfdtheta_a = (e_theta * transpose(a))
        dfdtheta_s = (e_theta * transpose(s))
        theta_a +=  1* lr * dfdtheta_a
        theta_s += 1 * lr * dfdtheta_s
        total_r +=1
        dfda = [(transpose(dsda_hat) * dfdmu) - (e_theta * transpose(theta_a))[1]]
        push!(dsdahats, dsda_hat)
        a += lr * dfda
        if done
            println("Reset!")
            s = env.reset()
            push!(rs, total_r)
            total_r = 0
        end
        push!(ess, e_s)
        push!(emdashs, e_mdash)
        push!(emudashs, e_mudash)
        push!(dfdmus, dfdmu)
        push!(dfdmudashes, dfdmudash)
        push!(dfdas, dfda)
        push!(Fs, F)
        push!(mus, mu)
        push!(mdashes, mdash)
        push!(as, a)
        push!(ss, s)
        push!(theta_as, theta_a)
        push!(e_thetas, e_theta)
        push!(dfdthetas, dfdtheta_a)
        push!(dsdas, dsda)
        push!(theta_ss, theta_s)
        env.render()
        sprev = s
    end
    return ess, emdashs, emudashs, dfdmus, dfdmudashes, dfdas, Fs, mus, mdashes, as,ss, rs, divergences, e_thetas, dfdthetas, dsdahats,dsdas,theta_ss
end
res = linear_PC_run_model(env,s, mu, mdash, a_param, lr, a, sigma_s, sigmamdash,R,theta_a,theta_s)
ess, emdashs, emudashs, dfdmus, dfdmudashes, dfdas, Fs, mus, mdashes, as,ss,rs, divergences, e_thetas, dfdthetas,dsdahats,dsdas, theta_ss = res
plot(format(dfdthetas))
plot(format(mus)[:,1])
plot!(format(ss)[:,1])
plot(format(dsdahats))
plot(format(dsdas)[:,4])
plot!(format(dsdahats)[:,4])
plot(format(ss)[:,4])
plot(format(ss)[:,2])
plot(format(dsdas)[:,3])
plot!(format(dsdahats)[:,3])
plot(format2(theta_ss)[:,:,4])
plot(format(as))
plot(rs)
