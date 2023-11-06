


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['agg.path.chunksize'] = 0
mpl.rcParams.update( mpl.rc_params() )
plt.rcParams.update({'font.size': 18})
#plt.rcParams.update({'text.usetex': True})
plt.rcParams.update({'text.latex.preamble': 'bold'})
plt.rc('font', family='serif')

slrl_steps = {'sawyer_pick_place':[100000],       
            'sawyer_window_open':[309,],             
            'sawyer_window_close':[13340, 50020, 8273],
            'sawyer_drawer_open': [18585],      
            'sawyer_drawer_close':[246],      
            'sawyer_button_press': [5500, 68501],     
            'sawyer_push':[1115]}


slrl_successes = {'sawyer_pick_place':[False],       
            'sawyer_window_open':[True],             
            'sawyer_window_close':[True, False, True],
            'sawyer_drawer_open': [True],      
            'sawyer_drawer_close':[True],      
            'sawyer_button_press': [True, True],     
            'sawyer_push':[True]}



slrl_steps_masking = {'sawyer_pick_place':[100000],       
            'sawyer_window_open':[22257],             
            'sawyer_window_close':[8107],
            'sawyer_drawer_open': [20754],      
            'sawyer_drawer_close':[265],      
            'sawyer_button_press': [112343],     
            'sawyer_push':[1155]}




# Please check the order
# steps = [100000, 309, 13340, 18585, 2220, 5500, 136]

colors = plt.cm.get_cmap('viridis', 3)

# Plotting

fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111)

# fig, ax = plt.subplots()
# x = np.arange(len(keys))
# ax.barh(env_names[0], steps[0] , height= 0.5, color=colors(1), alpha=0.7,   label ='Unsuccessful')

ax.barh(env_names[1:], steps[1:] , height= 0.1, color='#8c1514ff', alpha=0.7, edgecolor='black')
ax.barh(env_names[1:], steps[1:] , height= 0.1, color='#8c1514ff', alpha=0.7, edgecolor='black')


for i, v in enumerate(steps[1:]):
    ax.text(v + 1000, i-0.05, str(v), color='black', fontweight='bold')

# Customize the plot
ax.set_yticks(np.arange(0,len(env_names[1:])))
ax.set_yticklabels(env_names[1:])
ax.set_ylabel('Environments')
ax.set_xlabel('Number of Steps')
ax.set_title('Single Life RL Experiments Using Sine Embeddings')
# ax.legend()
ax.set_xlim(0,100000)
# Show the plot
plt.tight_layout()
plt.savefig('results/SLRL_results.png')
plt.show()