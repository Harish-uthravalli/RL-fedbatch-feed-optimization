import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style


style.use('fivethirtyeight')

fig, ax1 =  plt.subplots()

def animate(i):
    graph_data = open('plot.txt','r').read()
    lines = graph_data.split('\n')
    TEMP = []
    T = []
    for line in lines:
        if len(line) > 1:
            t, _ , _ ,_ ,temperature = line.split(',')
            T.append(float(t))
            TEMP.append(float(temperature))

    ax1.clear()
    
    ax1.plot(T , TEMP ,color="red", label= "Temperature")
    ax1.set_xlim(0,60)
    ax1.set_ylabel("Temperature", color="red")
    ax1.set_xlabel("Time (hours)")
    ax1.tick_params(axis='y',colors="red")
    

ani = animation.FuncAnimation(fig, animate, interval=1)
plt.show()