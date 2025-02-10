import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Arc


def main():
    poz0 = np.array([0, 0, 0]) # pozycja początkowa cząsteczki [m]
    v0 = np.array([0, 0, 0]) # prędkość początkowa cząsteczki [m/s]
    mp = 1.67e-27 # masa cząsteczki [kg]
    qp = 1.6e-19 # ładunek cząsteczki [C]
    b = 1.5 # indukcja magnetyczna [T]
    U = 50000 # napięcie [mV]
    d = 90e-6 # szerokość przerwy [m]
    r = 0.05 # promień cyklotronu [m]
    t0 = 0 # początkowy czas symulacji
    dt = 5e-12 # co ile liczone są parametry cząstki
    maxs = 20 # maksymalna liczba skoków
    proton = Czastka(poz0, v0, mp, qp)
    cyklotron = Cyklotron(b, U, d, r)
    cyklotron.symulacja(proton, t0, dt, maxs)

class Czastka:
    def __init__(self, poz, v, m, q):
        self.poz = np.array(poz)
        self.v = np.array(v)
        self.m = m
        self.q = q


class Cyklotron:
    def __init__(self, b, U, d, r):
        self.b = b
        self.U = U
        self.d = d
        self.B = np.array([0, 0, b])
        self.r = r
        self.E = np.array([U / d, 0, 0])

    def symulacja(self, czastka, t, dt, maxs):
        poz = czastka.poz
        v = czastka.v
        m = czastka.m
        q = czastka.q
        w = q * self.b / m # częstość cyklotronowa
        pozx = [poz[0]]
        pozy = [poz[1]]
        count = 0
        skoki = 0
        flaga = 0
        while np.linalg.norm(poz) < 1.5 * self.r and skoki <= maxs:
            F = np.array([0, 0, 0])
            if np.absolute(poz[0]) < self.d / 2:
                F = q * self.E * np.cos(w * t)
                if flaga == 1:
                    skoki += 1
                    flaga = 0
            elif np.linalg.norm(poz) > self.r:
                F = 0
            else:
                F = q * np.cross(v, self.B)
                flaga = 1

            a = F / m
            v = v + a * dt
            poz = poz + v * dt
            pozx.append(poz[0])
            pozy.append(poz[1])
            t = t + dt
            count += 1

        vk = np.linalg.norm(v)
        print("Końcowa prędkość:", vk, "m/s")
        print("Czas trwania symulacji:", t, "s")
        print(skoki)
        fig, ax = plt.subplots()

        ax.set_aspect('equal')
        ax.set_xlim([-1.2 * self.r, 1.2 * self.r])
        ax.set_ylim([-1.2 * self.r, 1.2 * self.r])
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

        plt.subplots_adjust(bottom=0.15)

        theta_offset = (self.d / (2 * self.r)) * (180 / np.pi)
        arc1 = Arc((0, 0), 2 * self.r, 2 * self.r, angle=0, theta1=90 + theta_offset, theta2=270 - theta_offset, color='black', lw=1.5)
        arc2 = Arc((0, 0), 2 * self.r, 2 * self.r, angle=0, theta1=-90 + theta_offset, theta2=90 - theta_offset, color='black', lw=1.5)
        ax.add_patch(arc1)
        ax.add_patch(arc2)

        x1 = self.r * np.cos(np.radians(90 + theta_offset))
        y1 = self.r * np.sin(np.radians(90 + theta_offset))
        x2 = self.r * np.cos(np.radians(-90 + theta_offset))
        y2 = self.r * np.sin(np.radians(-90 + theta_offset))
        ax.plot([-x1, x2], [y1, y2], color='black', lw=0.1)
        ax.plot([x1, -x2], [y1, y2], color='black', lw=0.1)

        trace, = ax.plot([], [], lw=1)
        running = [True]

        def toggle_animation(event):
            if running[0]:
                ani.event_source.stop()
            else:
                ani.event_source.start()
            running[0] = not running[0]

        ax_button = plt.axes([0.8, 0.1, 0.1, 0.075])
        button = Button(ax_button, 'Start/Stop')
        button.on_clicked(toggle_animation)

        ax_slider = plt.axes([0.2, 0.05, 0.55, 0.03])
        slider = Slider(ax_slider, 'Czas', 0, count-1, valinit=0, valstep=1)

        def update(frame):
            trace.set_data(pozx[:frame], pozy[:frame])
            slider.set_val(frame)
            return trace,

        def slider_update(val):
            frame = int(slider.val)
            trace.set_data(pozx[:frame], pozy[:frame])
            fig.canvas.draw_idle()

        slider.on_changed(slider_update)

        ani = FuncAnimation(fig, update, frames=range(0, count, 30), interval=1)
        plt.show()


if __name__ == '__main__':
    main()
