import bot_trainer as bt
import json
import numpy as np

H = 200
D = 600

params = {}
params['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization - Shape will be H x D
params['W2'] = np.random.randn(1,H) / np.sqrt(H) # Shape will be H
params['b1'] = np.zeros((H,1))
params['b2'] = np.zeros((1,1))

#print(params)

for key in params:
    params[key] = params[key].tolist()

#np.savetxt('params.txt', params, delimiter=',')
with open('params.txt', 'w') as f:
    f.write(json.dumps(params))

with open('params.txt', 'r') as f:
    params2 = json.load(f)

for key in params2:
    params2[key] = np.array(params2[key])

#params2 = eval(params2.replace("array", "np.array"))
print(params2)