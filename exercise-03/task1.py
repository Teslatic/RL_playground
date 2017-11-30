#!/usr/bin/env python3
# Task 1 of Exercise 3 
import numpy as np


discount = 1.0
states = np.arange(16)

V = [0, 0,  0,  0,  0,
     0, 0,  0,  0,  0,
     0, 0,  0,  0,  0]


r =[-1,   -1,     -1,     -1,     -1,
    -1,   -1,     -1,     -1,     10,
    -100, -100,   -100,   -100, -100]

episodes = [[5,6,7,2,3,4,9],[5,6,7,12], [5,6,7,8,3,8,9]] #terminal states omitted
v_next_ep = [dict({5:6,6:7,7:2,2:3,3:4,4:9}),
             dict({5:6,6:7,7:12}),
             dict({5:6,6:7,7:8,8:3,3:8})]
             

ep_4 = [5,0,1,2,3,4]
v_next = dict({5:0,0:1,1:2,2:3,3:4,4:9})

alpha = [1,1/2,1/3]

rewards = []
#print(episodes[0])
for state in episodes[0]:
    rewards.append(r[state])
print("rewards: {}".format(rewards))

for i in range(len(episodes[0])):
    total_return = sum(rewards[i+1:-1])
    print(total_return)







            
#for ep in range(0,3):
    #list_rewards = []
    #rewards_in_ep = []
    #total_return = []
    #for idx,val in enumerate(episodes[ep]):
        ##print(v_next_ep[ep][val])
        #rewards_in_ep.append(r[v_next_ep[ep][val]])
        ##print(rewards_in_ep)
    ##print(rewards_in_ep)
    #for idx,val in enumerate(episodes[ep]):
        #print(rewards_in_ep)
        #total_return.append(sum(rewards_in_ep,idx))
    #print("Total return in episode {}: \n{}".format(ep+1,total_return))
    #visit = [False]*15
    #for idx,val in enumerate(episodes[ep]):
        ##print(visit[val])
        #if(visit[val]==False):
            #V[val] += alpha[ep]*(total_return[idx] - V[val])
            #visit[val] = True
    #print("Values after episode {}".format(ep+1))
    #print("|---------------------------------------|")
    #print("| {:.1f}\t| {:.1f}\t| {:.1f}\t| {:.1f}\t| {:.1f}  |".format(V[0],V[1],V[2],V[3],V[4]))
    #print("|---------------------------------------|")
    #print("| {:.1f}\t| {:.1f}\t| {:.1f}\t| {:.1f}\t| {:.1f}   |".format(V[5],V[6],V[7],V[8],V[9]))
    #print("|---------------------------------------|")
    #print("| {:.1f}\t| {:.1f}\t| {:.1f}\t| {:.1f}\t| {:.1f}   |".format(V[10],V[11],V[12],V[13],V[14]))
    #print("|---------------------------------------|")       
    
#delta = [0]*15
#rewards = []
#for idx,step in enumerate(ep_4):
    #delta[step] = r[step]+V[v_next[step]]-V[step]
        
#print("Temporal Difference Error in episode 4")
#print("|-----------------------------------------------|")
#print("| V_3 \t {:.1f}\t| {:.1f}\t| {:.1f}\t| {:.1f}\t| {:.1f}  |".format(V[0],V[1],V[2],V[3],V[4]))
#print("| error {:.1f}\t| {:.1f}\t| {:.1f}\t| {:.1f}\t| {:.1f} |".format(delta[0],delta[1],delta[2],delta[3],delta[4]))
#print("|-----------------------------------------------|")
#print("| V_3\t{:.1f}\t| {:.1f}\t| {:.1f}\t| {:.1f}\t| {:.1f}   |".format(V[5],V[6],V[7],V[8],V[9]))
#print("| error  {:.1f}\t| {:.1f}\t| {:.1f}\t| {:.1f}\t| {:.1f}   |".format(delta[5],delta[6],delta[7],delta[8],delta[9]))
#print("|-----------------------------------------------|")
#print("| V_3 \t {:.1f}\t| {:.1f}\t| {:.1f}\t| {:.1f}\t| {:.1f}   |".format(V[10],V[11],V[12],V[13],V[14]))
#print("| error  {:.1f}\t| {:.1f}\t| {:.1f}\t| {:.1f}\t| {:.1f}   |".format(delta[10],delta[11],delta[12],delta[13],delta[14]))
#print("|-----------------------------------------------|")



#td_lambda = [0,0.5,1]
#steps = np.arange(0,6)
#for idx,val in enumerate(td_lambda):
    #v_21 = V[5]
    #lambda_sum = 0
    #for n in steps:
        #lambda_sum += td_lambda[idx]**n * delta[5]
    #lambda_sum = (1/3)*lambda_sum
    #v_21 += lambda_sum
    #print("TD-Lambda of state S(2,1] with a lambda of {}:\t{:.1f}".format(val,v_21))



