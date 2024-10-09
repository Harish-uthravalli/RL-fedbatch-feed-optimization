import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style


style.use('fivethirtyeight')

fig, ax1 =  plt.subplots()
ax2 = ax1.twinx()
ax3 = ax1.twinx()
def animate(i):
    graph_data = open('plot.txt','r').read()
    lines = graph_data.split('\n')
    X = []
    S = []
    E = []
    T = []
    for line in lines:
        if len(line) > 1:
            t, x, s, e, _, _ = line.split(',')
            T.append(float(t))
            X.append(float(x))
            S.append(float(s))
            E.append(float(e))

    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax1.plot(T ,E ,color="red", label= "Enzyme Activity U/L")
    ax2.plot(T, S, color="orange", label="Substrate mol/L")
    ax3.plot(T , X ,color="blue", label="Cells CDW g/L")
    ax3.spines['right'].set_position(('axes',1.15))
    # ax1.set_xlim(0,50)
    # ax2.set_xlim(0,50)
    # ax3.set_xlim(0,50)
    # ax1.set_ylim(0,3)
    ax1.set_ylabel("Enzyme Activity U/L", color="red")
    ax1.set_xlabel("Time (hours)")
    ax2.set_ylabel("Substrate mol/L ", color="orange")
    ax3.set_ylabel("Cells CDW g/L", color="blue")
    ax1.tick_params(axis='y',colors="red")
    ax2.tick_params(axis='y',colors="orange")
    ax3.tick_params(axis='y',colors="blue")
    ax2.spines['right'].set_color("orange")
    ax3.spines['right'].set_color("blue")
    ax3.spines['left'].set_color("red")
    

ani = animation.FuncAnimation(fig, animate, interval=1)
plt.show()