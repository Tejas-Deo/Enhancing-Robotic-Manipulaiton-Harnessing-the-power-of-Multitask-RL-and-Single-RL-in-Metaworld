import json
from tqdm import tqdm

with open('/home/tejas/Documents/Stanford/CS 224R/Final Project/Metaworld/data/Pretrianed_policy_data/open_window/open_window_replay_buffer.json', 'r') as f:
    data = json.load(f)


keys = data.keys()
values = data.values()

print("Keys: ", keys)

print('state memory length: ', len(data['self.state_memory']))
print("new state memory length: ", len(data['self.new_state_memory']))
print("len action memory: ", len(data['self.action_memory']))
print("len reward_memory: ", len(data['self.reward_memory']))
print("len terminal memory: ", len(data['self.terminal_memory']))

print(type(data['self.state_memory']))

print(type(data['self.state_memory'][0]))
print(len(data['self.state_memory'][0]))

# print(data['self.state_memory'][0])



def count_lists_with_all_zeros(list_of_lists):
    count = 0
    for sublist in list_of_lists:
        if all(element == 0 for element in sublist):
            count += 1
    return count


output = count_lists_with_all_zeros(data['self.new_state_memory'])
print("OUTPUT: ", output)


#=========================================================


state_memory = data['self.state_memory']
new_state_memory = data['self.new_state_memory']


# to define the ANNOTATED lists
ann_state_memory = []
ann_new_state_memory = []

ann_action_memory = data['self.action_memory']
ann_reward_memory = data['self.reward_memory']
ann_terminal_memory = data['self.terminal_memory']


for i in tqdm(state_memory):
    ann_obs = i.copy()
    ann_obs.extend([1,0])
    ann_state_memory.append(ann_obs)



for j in tqdm(new_state_memory):
    ann_new_obs = j.copy()
    ann_new_obs.extend([1,0])
    ann_new_state_memory.append(ann_new_obs)


ann_arrays_dict = {
    'self.state_memory': ann_state_memory,
    'self.new_state_memory': ann_new_state_memory,
    'self.action_memory': ann_action_memory,
    'self.reward_memory': ann_reward_memory,
    'self.terminal_memory': ann_terminal_memory
}


# Convert the dictionary to a JSON string
json_string = json.dumps(ann_arrays_dict)

# Save the JSON string to a file
with open("ann_open_window_replay_buffer.json", "w") as f:
    f.write(json_string)


print("File saved!")
