#!/usr/bin/env python
# coding: utf-8

# # Project: Reinforcement Learning
# - Bigger field - more learning

# ### Project field
# - Use Reinforcement Learning with Q-learning to find solutions to this field.
# ![Field](img/field-2.png "Field")

# ### Step 1: Import libraries

# In[1]:


import numpy as np
import random


# ### Step 2: Create a field
# 
# ![Field](img/field-3.png)

# - **__init__**:
#     - Use a list of list with integer values to represent all the states
#         - Goal end state should be 1, illegal states -1, other states 0
#     - Set the state to be random fo the size of states
# - **done**:
#     - Check if current state has non-negative values
# - **get_possible_actions**:
#     - Set a list to all possible actions **actions = [0, 1, 2, 3]**
#         - action = 0 is left
#         - action = 1 is right
#         - action = 2 is up
#         - action = 3 is down
#     - Then check if state is in a position where a possible actions should be removed.
#     - Finally, return the remaining actions
# - **update_next_state**:
#     - Get the current state
#     - Check if move is illegal, then return current state and -10 in reward
#     - Otherwise opdate state and return the reward according to new state

# In[2]:


class Field:
    def __init__(self) -> None:
        """
        Initialize field and set a random start state
        """
        self.states = [
            [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ]
        self.state = (random.randrange(0, len(self.states)), random.randrange(0, len(self.states[0])))
    
    def done(self):
        """
        Check if it isn't in a neutral state
        """
        if self.states[self.state[0]][self.state[1]] != 0:
            return True
        else:
            return False
    
    def get_possible_actions(self):
        """
        Return possible actions in state

        Action:
               0 => left
               1 => right
               2 => up
               3 => down
        """    
        actions = [0, 1, 2, 3]
        if self.state[0] == 0:
            actions.remove(2)
        if self.state[0] == len(self.states) -1:
            actions.remove(3)
        if self.state[1] == 0:
            actions.remove(0)
        if self.state[1] == len(self.states[0]) -1:
            actions.remove(1)
        return actions

    def update_next_state(self, action):
        """ 
        Update state according to action -> Return state and reward
        """
        x, y = self.state

        if action == 0:
            if y == 0:
                return self.state, -10
            self.state = x, y - 1
        if action == 1:
            if y == len(self.states) -1:
                return self.state, -10
            self.state = x, y + 1
        if action == 2:
            if x == 0:
                return self.state, -10
            self.state = x - 1, y
        if action == 3:
            if self.state == len(self.states) -1:
                return self.state, -10
            self.state = x + 1, y        
        reward = self.states[self.state[0]][self.state[1]]
        return self.state, reward


# In[3]:


field = Field()
field.state, field.done(), field.get_possible_actions()


# In[4]:


field.update_next_state(2)
field.state, field.done(), field.get_possible_actions()


# ### Step 3: Train the model
# - Create a $q$-table initialized to all 0
#     - Use **q_table = np.zeros(...)** *(insert values for ...)*
# - Set **alpha = .5, gamma = 0.5,** and **epsilon = 0.5**
# - Create *for*-loop iterating 10000
#     - Create new field
#     - While field not done
#         - Get possible actions and assign to **actions**
#         - With probability epsilon take a random action, otherwise take the best action
#             - HINT: **random.uniform(0, 1) < epsilon**
#             - HINT: Random action: **random.choice(actions)**, and best action: **np.argmax(q_table[field.state])**
#         - Get current state and assign it to **cur_x, cur_y**
#         - Update next state and get it and the reward
#         - Update **q_table[cur_x, cur_y, action] = (1 - alpha)*q_table[cur_x, cur_y, action] + alpha*(reward + gamma*np.max(q_table[next_x, next_y]))**

# In[5]:


field = Field()
q_table = np.zeros((len(field.states), len(field.states[0]), 4))

alpha = .5
epsilon = .5
gamma = .5

for _ in range(10000):
    field = Field()
    while not field.done():
        actions = field.get_possible_actions()
        if random.uniform(0, 1) < epsilon:
            action = random.choice(actions)
        else:
            action = np.argmax(q_table[field.state])
        
        cur_x, cur_y = field.state
        (next_x, next_y), reward = field.update_next_state(action)
        q_table[cur_x, cur_y, action] = (1 - alpha)*q_table[cur_x, cur_y, action] + alpha*(reward + gamma*np.max(q_table[next_x, next_y]))


# In[6]:


q_table


# ### Step 4: Solve a task
# - To see the path make a variable **path = np.zeros((3, 11))**
# - Create a field **Field()**
# - To count steps assign **steps = 1**
# - Assign the start state in the path to **np.nan**.
# - The we begin: while not solved.
#     - Get the **action** to take
#     - Get the next **state**
#     - Update **path** with **steps**
#     - Increment **steps** with one
# - see the **path**

# In[7]:


path = np.zeros((3, 11))
field = Field()
steps = 1
path[field.state[0]][field.state[1]] = np.nan

while not field.done():
    action = np.argmax(q_table[field.state])

    (next_x, next_y), _ = field.update_next_state(action)
    path[next_x][next_y] = steps
    steps +=1


# In[ ]:


path


# > ### Note
# > - The training phase (Step 3) could just take random actions
# > - Our example (Step 4) does not learn anything new

# In[ ]:




