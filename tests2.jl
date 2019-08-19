using OpenAIGym
using LinearAlgebra
using Statistics, StatsBase
using Plots

env = GymEnv("CartPole-v1")
s = reset!(env)
os = zeros(100, 4)
for i in 1:100
    a = Int(rand() > 0.5)
    if env.done
        o = reset!(env)
    else
        r,o = step!(env, a)
    end
    os[i,:] = o
end
plot(os)

function data_collection(N)
    as = zeros(N,1)
    os = zeros(N+1,4)
    odashes = zeros(N,4)
    os[1,:] = reset!(env)
    for i in 1:N
        a = Int(rand() > 0.5)
        if env.done
            odash = reset!(env)
        else
            r, odash= step!(env,a)
        end
        as[i] = a
        odashes[i,:] = odash
        os[i+1,:] = odash
    end
    return os, odashes, as
end

os, odashes, as = data_collection(10000)
plot(os)
os = os[1:end-1,:]

function blib(N)
    theta = randn(4,4)
    mu = randn(4,1)
    wa1 = randn(4,1)
    wa2 = randn(4,1)
    os = zeros(N,4)
    mus = zeros(N,4)
    preds = zeros(N,4)
    es = zeros(N,4)
    thetas = zeros(N,4,4)
    wa2s = zeros(N,4)
    wa1s = zeros(N,4)
    dmus = zeros(N,4)
    dthetas = zeros(N,4,4)
    dwa1s = zeros(N,4)
    dwa2s = zeros(N,4)
    as = zeros(N)
    lr = 0.1

    for i in 1:N
        a = Int(rand() > 0.5)
        if a == 0
            a1 = 1
            a2 = 0
        else
            a1 = 0
            a2 = 1
        end
        if env.done
            o = reset!(env)
        else
            r,o = step!(env, a)
        end
        o[1] = o[1] * 5
        o[3] = o[3] * 5
        #mu = transpose(theta) * o
        pred = (theta * mu) .+ (wa1 * a1) .+ (wa2 * a2)
        e = o - pred
        dmu = theta * e
        dtheta = e * transpose(mu)
        dwa1 = (e * a1) + (o * a1)
        dwa2 = (e * a2) + (o * a2)

        os[i,:] = o
        mus[i,:] = mu
        preds[i,:] = pred
        thetas[i,:,:] = theta
        wa1s[i,:] = wa1
        wa2s[i,:] = wa2
        es[i,:] = e
        dmus[i,:] = dmu
        dthetas[i,:,:] = dtheta
        dwa1s[i,:] = dwa1
        dwa2s[i,:] = dwa2
        as[i] = a


        mu += lr * dmu
        theta +=  lr * dtheta
        wa1 += lr * dwa1
        wa2 += lr * dwa2
    end
    return os,mus, preds, thetas, wa1s, wa2s, es, dmus, dthetas, dwa1s, dwa2s,as
end
preds
preds
os,mus, preds, thetas, wa1s, wa2s, es, dmus, dthetas, dwa1s, dwa2s,as = blib(1000)
plot(os)
plot(preds)
plot(os .- preds)
plot(thetas)
plot(mus)
plot(wa1s)
plot(wa2s)
plot(as)
plot(es)

plot(os[:,1])
plot!(preds[:,1])
os[499,:]
preds[501,:]
plot(preds[500:600,1])
plot!(os[500:600,1])
plot!(es[:,1])

plot(es[:,1])
plot!((os .- preds)[:,1])

plot(os[:,2])
plot!(preds[:,2])
plot!(es[:,2])

plot(os[:,3])
plot!(preds[:,3])
plot!(es[:,3])

plot(os[:,4])
plot!(preds[:,4])

std(os[:,1])
std(os[:,2])
std(os[:,3])
std(os[:,4])





phi = Matrix{Float32}(randn(4,1))
theta = Matrix{Float32}(randn(4,4))
pred = theta * phi


theta = randn(20,4)
bib = randn(4,1)
theta * bib

function active_inference(N)
    D = 4
    theta = randn(4,D)
    mu = randn(D,1)
    wa1 = randn(4,1)
    wa2 = randn(4,1)
    os = zeros(N,4)
    mus = zeros(N,D)
    preds = zeros(N,4)
    es = zeros(N,4)
    thetas = zeros(N,4,D)
    wa2s = zeros(N,4)
    wa1s = zeros(N,4)
    dmus = zeros(N,D)
    dthetas = zeros(N,4,D)
    dwa1s = zeros(N,4)
    dwa2s = zeros(N,4)
    as = zeros(N)
    lr = 0.1
    prior = zeros(4,1)
    diffs = zeros(N,1)
    num_resets = 0
    rewards = []
    reward_per_epoch = 0
    preda1s = zeros(N, 4)
    preda2s = zeros(N,4)
    o = randn(4,1)
    for i in 1:N
        mu = transpose(theta) * o
        preda1 = (theta * mu) .+ wa1
        preda2 =(theta * mu) .+ wa2
        #print(size(preda1))
        preda1s[i,:] = preda1
        preda2s[i,:] = preda2
        diff = sum(abs.(preda1 .- prior)) - sum(abs.(preda2 .- prior))
        if diff <=0
            a1 = 1
            a2 = 0
            a = 0
        else
            a1 = 0
            a2 = 1
            a = 1
        end
        diffs[i] = diff
        if env.done
            o = reset!(env)
            num_resets +=1
            push!(rewards, reward_per_epoch)
            reward_per_epoch = 0
        else
            r,o = step!(env, a)
            reward_per_epoch +=1
        end
        #render(env)
        o[1] = o[1] * 5
        o[3] = o[3] * 5
        pred = (theta * mu) .+ (wa1 * a1) .+ (wa2 * a2)
        e = o - pred
        dmu = theta * e
        dtheta = e * transpose(mu)
        dwa1 = (e * a1) + (o * a1)
        dwa2 = (e * a2) + (o * a2)

        os[i,:] = o
        mus[i,:] = mu
        preds[i,:] = pred
        thetas[i,:,:] = theta
        wa1s[i,:] = wa1
        wa2s[i,:] = wa2
        es[i,:] = e
        dmus[i,:] = dmu
        dthetas[i,:,:] = dtheta
        dwa1s[i,:] = dwa1
        dwa2s[i,:] = dwa2
        as[i] = a


        mu += lr * dmu
        theta +=   lr * dtheta
        wa1 +=  lr * dwa1
        wa2 +=   lr * dwa2

    end
    print("num_resets: $num_resets")
    print("mean reward: $(mean(rewards))")
    return os,mus, preds, thetas, wa1s, wa2s, es, dmus, dthetas, dwa1s, dwa2s,as, diffs, num_resets, rewards, preda1s, preda2s
end

os,mus, preds, thetas, wa1s, wa2s, es, dmus, dthetas,
 dwa1s, dwa2s,as,diffs, num_resets, rewards, preda1s, preda2s = active_inference(1000)
plot(os)
plot(preds)
plot(os .- preds)
plot(thetas[:,:,3])
plot(mus)
plot(wa1s)
plot(wa2s)
plot(as)
plot(es)
plot(diffs)

plot(os[:,1])
plot!(preds[:,1])
plot!(es[:,1])

plot(os[:,1])
plot!(preda1s[:,1])
plot!(preda2s[:,1])

plot(es[:,1])
plot!((os .- preds)[:,1])

plot(os[:,2])
plot!(preds[:,2])
plot!(es[:,2])

plot(os[:,3])
plot!(preds[:,3])
plot!(es[:,3])

plot(os[:,4])
plot!(preds[:,4])

plot(rewards)

function mu_transpose_active_inference(N)
    D = 20
    theta = randn(4,D)
    mu = randn(D,1)
    wa1 = randn(4,1)
    wa2 = randn(4,1)
    os = zeros(N,4)
    mus = zeros(N,D)
    preds = zeros(N,4)
    es = zeros(N,4)
    thetas = zeros(N,4,D)
    wa2s = zeros(N,4)
    wa1s = zeros(N,4)
    dmus = zeros(N,D)
    dthetas = zeros(N,4,D)
    dwa1s = zeros(N,4)
    dwa2s = zeros(N,4)
    as = zeros(N)
    lr = 0.001
    prior = zeros(4,1)
    diffs = zeros(N,1)
    num_resets = 0
    rewards = []
    reward_per_epoch = 0
    preda1s = zeros(N, 4)
    preda2s = zeros(N,4)
    o = randn(4,1)
    for i in 1:N
        #mu = transpose(theta) * o
        preda1 = (theta * mu) .+ wa1
        preda2 =(theta * mu) .+ wa2
        #print(size(preda1))
        preda1s[i,:] = preda1
        preda2s[i,:] = preda2
        diff = sum(abs.(preda1 .- prior)) - sum(abs.(preda2 .- prior))
        if diff >=0
            a1 = 1
            a2 = 0
            a = 0
        else
            a1 = 0
            a2 = 1
            a = 1
        end
        diffs[i] = diff
        if env.done
            o = reset!(env)
            num_resets +=1
            push!(rewards, reward_per_epoch)
            reward_per_epoch = 0
        else
            r,o = step!(env, a)
            reward_per_epoch +=1
        end
        #render(env)
        o[1] = o[1] * 5
        o[3] = o[3] * 5
        pred = (theta * mu) .+ (wa1 * a1) .+ (wa2 * a2)
        e = o - pred
        #print(size(theta),size(e))
        dmu = transpose(theta) * e
        dtheta = e * transpose(mu)
        dwa1 = (e * a1) + (o * a1)
        dwa2 = (e * a2) + (o * a2)

        os[i,:] = o
        mus[i,:] = mu
        preds[i,:] = pred
        thetas[i,:,:] = theta
        wa1s[i,:] = wa1
        wa2s[i,:] = wa2
        es[i,:] = e
        dmus[i,:] = dmu
        dthetas[i,:,:] = dtheta
        dwa1s[i,:] = dwa1
        dwa2s[i,:] = dwa2
        as[i] = a


        mu += lr * dmu
        theta +=   0 * dtheta
        wa1 +=  0 * dwa1
        wa2 +=   0 * dwa2

    end
    print("num_resets: $num_resets")
    print("mean reward: $(mean(rewards))")
    return os,mus, preds, thetas, wa1s, wa2s, es, dmus, dthetas, dwa1s, dwa2s,as, diffs, num_resets, rewards, preda1s, preda2s
end

os,mus, preds, thetas, wa1s, wa2s, es, dmus, dthetas,dwa1s, dwa2s,as,diffs, num_resets, rewards, preda1s, preda2s = mu_transpose_active_inference(1000)
plot(os[:,2])
plot!(preds[:,2])
plot!(mus[:,:])
plot(wa1s)



function Ndim_active_inference(N)
    D = 4
    theta = randn(4,D)
    mu = randn(D,1)
    wa1 = randn(4,1)
    wa2 = randn(4,1)
    os = zeros(N,4)
    mus = zeros(N,D)
    preds = zeros(N,4)
    es = zeros(N,4)
    thetas = zeros(N,4,D)
    wa2s = zeros(N,4)
    wa1s = zeros(N,4)
    dmus = zeros(N,D)
    dthetas = zeros(N,4,D)
    dwa1s = zeros(N,4)
    dwa2s = zeros(N,4)
    as = zeros(N)
    lr = 0.1
    prior = zeros(4,1)
    diffs = zeros(N,1)
    num_resets = 0
    rewards = []
    reward_per_epoch = 0
    preda1s = zeros(N, 4)
    preda2s = zeros(N,4)
    o = randn(4,1)
    a1 = 0
    a2 = 0
    for i in 1:N
        #mu = transpose(theta) * o
        for i in 1:20
            pred = (theta * mu) .+ (wa1 * a1) .+ (wa2 * a2)
            e = mu - (transpose(theta) * o)
            #e = o - pred
            #e2 = pred - o
            #print(size(theta),size(e))
            #print(mean(e))
            #print(mean(transpose(theta) * e2))
            dmu =  e #+ (transpose(theta) * e2)


            mu -= 0.1 * dmu
        end
        preda1 = (theta * mu) .+ wa1
        preda2 =(theta * mu) .+ wa2
        #print(size(preda1))
        preda1s[i,:] = preda1
        preda2s[i,:] = preda2
        diff = sum(abs.(preda1 .- prior)) - sum(abs.(preda2 .- prior))
        if diff <=0
            a1 = 1
            a2 = 0
            a = 0
        else
            a1 = 0
            a2 = 1
            a = 1
        end
        diffs[i] = diff
        if env.done
            o = reset!(env)
            num_resets +=1
            push!(rewards, reward_per_epoch)
            reward_per_epoch = 0
        else
            r,o = step!(env, a)
            reward_per_epoch +=1
        end
        #render(env)
        o[1] = o[1] * 5
        o[3] = o[3] * 5

        pred = (theta * mu) .+ (wa1 * a1) .+ (wa2 * a2)
        e = o - pred
        dtheta = e * transpose(mu)
        dwa1 = (e * a1) + (o * a1)
        dwa2 = (e * a2) + (o * a2)
        dmu = transpose(theta) * e
        mu += 0.01 * dmu

        os[i,:] = o
        mus[i,:] = mu
        preds[i,:] = pred
        thetas[i,:,:] = theta
        wa1s[i,:] = wa1
        wa2s[i,:] = wa2
        es[i,:] = e
        dmus[i,:] = dmu
        dthetas[i,:,:] = dtheta
        dwa1s[i,:] = dwa1
        dwa2s[i,:] = dwa2
        as[i] = a

        theta +=   0.1 * lr * dtheta
        wa1 +=  0.1 * lr * dwa1
        wa2 +=   0.1 * lr * dwa2

    end
    #print("num_resets: $num_resets")
    print("mean reward: $(mean(rewards))")
    return os,mus, preds, thetas, wa1s, wa2s, es, dmus, dthetas, dwa1s, dwa2s,as, diffs, num_resets, rewards, preda1s, preda2s
end


os,mus, preds, thetas, wa1s, wa2s, es, dmus, dthetas,
 dwa1s, dwa2s,as,diffs, num_resets, rewards, preda1s, preda2s = Ndim_active_inference(1000)
plot(os)
plot(preds)
plot(os .- preds)
plot(thetas[:,:,8])
plot(mus)
plot(wa1s)
plot(wa2s)
plot(as)
plot(es)
plot(diffs)

plot(os[:,1])
plot!(preds[:,1])
plot!(es[:,1])

plot(os[:,1])
plot!(preda1s[:,1])
plot!(preda2s[:,1])

os[999,:]
preds[998,:]
mus[998,:]
rs = []
for i in 1:100
    os,mus, preds, thetas, wa1s, wa2s, es, dmus, dthetas,
     dwa1s, dwa2s,as,diffs, num_resets, rewards, preda1s, preda2s = Ndim_active_inference(1000)
     push!(rs, mean(rewards))
 end
print(mean(rs))



function 2_layer_active_inference(N)
    theta = randn(4,4)
    mu = randn(4,1)
    wa1 = randn(4,1)
    wa2 = randn(4,1)
    os = zeros(N,4)
    mus = zeros(N,4)
    preds = zeros(N,4)
    es = zeros(N,4)
    thetas = zeros(N,4,4)
    wa2s = zeros(N,4)
    wa1s = zeros(N,4)
    dmus = zeros(N,4)
    dthetas = zeros(N,4,4)
    dwa1s = zeros(N,4)
    dwa2s = zeros(N,4)
    as = zeros(N)
    lr = 1
    prior = zeros(4,1)
    diffs = zeros(N,1)
    num_resets = 0
    rewards = []
    reward_per_epoch = 0
    o = randn(4,1)
    for i in 1:N
        preda1 = (theta * mu) .+ wa1
        preda2 =(theta * mu) .+ wa2
        diff = sum(abs.(preda1 .- prior)) - sum(abs.(preda2 .- prior))
        if diff <=0
            a1 = 1
            a2 = 0
            a = 0
        else
            a1 = 0
            a2 = 1
            a = 1
        end
        diffs[i] = diff

        #a = Int(rand() > 0.5)

        #mu = transpose(theta) * o
        if env.done
            o = reset!(env)
            num_resets +=1
            push!(rewards, reward_per_epoch)
            reward_per_epoch = 0
        else
            r,o = step!(env, a)
            reward_per_epoch +=1
        end
        #render(env)
        o[1] = o[1] * 5
        o[3] = o[3] * 5
        pred = (theta * mu) .+ (wa1 * a1) .+ (wa2 * a2)
        e = o - pred
        dmu = theta * e
        dtheta = e * transpose(mu)
        dwa1 = (e * a1) + (o * a1)
        dwa2 = (e * a2) + (o * a2)

        os[i,:] = o
        mus[i,:] = mu
        preds[i,:] = pred
        thetas[i,:,:] = theta
        wa1s[i,:] = wa1
        wa2s[i,:] = wa2
        es[i,:] = e
        dmus[i,:] = dmu
        dthetas[i,:,:] = dtheta
        dwa1s[i,:] = dwa1
        dwa2s[i,:] = dwa2
        as[i] = a


        mu += lr * dmu
        theta +=  0.1 * lr * dtheta
        wa1 += 0.1 * lr * dwa1
        wa2 += 0.1 * lr * dwa2

    end
    print("num_resets: $num_resets")
    print("mean reward: $(mean(rewards))")
    return os,mus, preds, thetas, wa1s, wa2s, es, dmus, dthetas, dwa1s, dwa2s,as, diffs, num_resets, rewards
end
