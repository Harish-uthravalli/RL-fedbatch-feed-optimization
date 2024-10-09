import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style


style.use('fivethirtyeight')

fig, ax1 =  plt.subplots()
ax2 = ax1.twinx()

def animate(i):
    graph_data = open('plot.txt','r').read()
    lines = graph_data.split('\n')
    X = []
    S = []
    E = []
    T = []
    R = []
    for line in lines:
        if len(line) > 1:
            t, _, _, e, _, reward = line.split(',')
            T.append(float(t))
            E.append(float(e))
            R.append(float(reward))

    ax1.clear()
    ax2.clear()
    ax1.plot(T ,E ,color="red", label= "Enzyme Activity U/L")
    ax2.plot(T, R, color="blue", label="Reward Points")
    ax1.set_xlim(0,60)
    ax2.set_xlim(0,60)
    ax1.set_ylim(0,50)
    ax2.set_ylim(-10, 10)
    ax1.set_ylabel("Enzyme Activity U/L", color="red")
    ax1.set_xlabel("Time (hours)")
    ax2.set_ylabel("Reward Points", color="blue")
    
    ax1.tick_params(axis='y',colors="red")
    ax2.tick_params(axis='y',colors="blue")
    
    ax2.spines['right'].set_color("blue")
    
    

ani = animation.FuncAnimation(fig, animate, interval=1)
plt.show()