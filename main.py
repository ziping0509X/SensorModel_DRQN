import matplotlib.pyplot as plt
import math
import numpy as np
from DRQN import BrainDRQN
from ENV_R import ENVIRONMENT_R
import pandas as pd

SENSOR_NUM = 2
ACTION_NUM = 9
Time = 0

N=200000
t1 = np.linspace(0,N,num= N)
t2 = np.linspace(0,8,num= 9)
omega = math.pi/9
A = 10
sig = A * np.sin(omega*t1)

#get actionChoice matrix
def actionChoice(sig):
    actionChoice = []
    for i in range(5):
        actionChoice.append(sig[i])
    for i in range(10,14):
        actionChoice.append(sig[i])
    return actionChoice
actionChoice = actionChoice(sig)

Qnetwork_R = BrainDRQN(ACTION_NUM,SENSOR_NUM)
Env_R = ENVIRONMENT_R(SensorNum= SENSOR_NUM,ActionNum=ACTION_NUM,ActionChoice=actionChoice)

Reward = []
R_total = 0
Loss = []
ActionShow = []

while Time < N-1:

    #get satate\action\reward\nextstate
    if Time < 2100:
        print("iterations:%d" %Time)
        stateInput = Env_R.creatSensor(Power=sig[Time])
        actionIput = Qnetwork_R.getAction(actionnum=ACTION_NUM, stateInput=stateInput, time=Time)
        reward, actionshow = Env_R.getReward(stateInput=stateInput, actionInput=actionIput)
        ActionShow.append(actionshow)
        nextState = Env_R.creatSensor(Power=sig[Time + 1])
        Qnetwork_R.sendmemory(currentstate=stateInput, nextstate=nextState, action=actionIput, reward=reward)
        R_total += reward
        Reward.append(R_total)

    else :
        print("iterations:%d" % Time)
        stateInput = Env_R.creatSensor(Power=sig[Time])
        stateInputAll = []
        # for i in range(1):
        #     stateInputAll.append(Env_R.creatSensor(Power=sig[Time-4+i]))
        # stateInputAll = np.array(stateInputAll)
        # stateInputAll = stateInputAll.reshape([1,5,2])
        stateInputAll.append(stateInput)

        actionIput = Qnetwork_R.getAction(actionnum= ACTION_NUM,stateInput= stateInputAll,time= Time)
        reward,actionshow = Env_R.getReward(stateInput= stateInput,actionInput= actionIput)
        ActionShow.append(actionshow)
        nextState = Env_R.creatSensor(Power= sig[Time+1])

        Qnetwork_R.sendmemory(currentstate=stateInput,nextstate=nextState,action=actionIput,reward=reward)

        #get loss and train Qnetwork
        loss = Qnetwork_R.getLoss()

        R_total += reward
        Reward.append(R_total)

        if not loss == 0:
            Loss.append(loss)

    Time = Time + 1

Loss1 = np.array(Loss)
data1 = pd.DataFrame(Loss1,columns=['Loss'])
data1.to_csv("D:\YuanZihong\SensorModel_DRQN_1\Loss.csv")

Reward1 = np.array(Reward)
data1 = pd.DataFrame(Reward1,columns=['Reward'])
data1.to_csv("D:\YuanZihong\SensorModel_DRQN_1\Reward.csv")

plt.rcParams["font.family"]="SimHei"
plt.rcParams['axes.unicode_minus']=False

fig = plt.figure(num= 1,figsize=(12,6))
ax1 = fig.add_subplot(121)
ax1.set_xlim(0,36)
ax1.set_xlabel("时间/t")
ax1.set_ylabel("函数值")
ax1.set_title("环境状态集[0-36]")
ax1.scatter(t1,sig,c= 'b')

fig = plt.figure(num= 1,figsize=(12,6))
ax1 = fig.add_subplot(122)
ax1.set_xlim(0,8)
ax1.set_xlabel("num")
ax1.set_ylabel("choice")
ax1.set_title("ActionChoice[0,9]")
ax1.scatter(t2,actionChoice,c= 'y')

fig = plt.figure(num= 2,figsize=(12,6))
ax1 = fig.add_subplot(121)
ax1.set_xlabel("迭代次数")
ax1.set_ylabel("损失函数值")
ax1.set_title("损失函数曲线(DRQN)")
ax1.plot(Loss)

fig = plt.figure(num= 2,figsiz=(12,6))
ax1 = fig.add_subplot(122)
ax1.set_xlabel("迭代次数")
ax1.set_ylabel("总奖励")
ax1.set_title("奖励函数曲线(DRQN)")
ax1.plot(Reward)

fig = plt.figure(num=3,figsize =(12,6))
ax1 = fig.add_subplot(111)
ax1.set_xlabel("时间/t")
ax1.set_ylabel("动作值")
ax1.set_title("DRQN经过训练以后的成果")
ax1.set_xlim(199900,199928)
ax1.scatter(t1,sig,c='b',label="输入信号")
lenth = len(ActionShow)
t3 = np.linspace(0,lenth,num=lenth)
ax1.scatter(t3,ActionShow,c='y',label="输出动作")
ax1.legend(loc=1)

plt.show()