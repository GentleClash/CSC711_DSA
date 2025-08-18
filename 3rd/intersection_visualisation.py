
from matplotlib.animation import FuncAnimation
import numpy as np
from dataclasses import dataclass, field
from typing import List, Set, Tuple, Optional, Dict, Any
from enum import Enum
from collections import defaultdict
import heapq
from fractions import Fraction
import matplotlib.pyplot as plt

from intersection import Point, LineSegment, SweepLineIntersection, create_test_segments

def animate_sweep_line(test_case=1, save=False):
    # Create segments
    segments = create_test_segments(test_case)

    # Run the sweep line algorithm, but record the states after each event
    sweep_line = SweepLineIntersection()
    sweep_line._initialize(segments)

    event_states = []
    while sweep_line.event_queue:
        # Copy current status for animation
        event_copy = list(sweep_line.event_queue)
        current_segments = list(sweep_line.status.segments)
        current_x = sweep_line.status.current_x
        event_states.append((current_x, current_segments, list(sweep_line.intersections)))

        event = heapq.heappop(sweep_line.event_queue)
        sweep_line._handle_event_point(event)

        # After handling event, record state again
        event_states.append((event.point.x, list(sweep_line.status.segments), list(sweep_line.intersections)))

    # Prepare plot
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(segments)))
    for i, seg in enumerate(segments):
        ax.plot([float(seg.start.x), float(seg.end.x)],
                [float(seg.start.y), float(seg.end.y)],
                color=colors[i], linewidth=2, label=f'Segment {seg.id}')

    sweep_line_plot, = ax.plot([], [], 'r--', linewidth=1.5, label="Sweep Line")
    active_dots, = ax.plot([], [], 'go', markersize=8, label="Active Segment Points")
    inters_plot, = ax.plot([], [], 'r*', markersize=12, label="Intersections")

    ax.set_xlim(min(float(p.start.x) for p in segments) - 1, max(float(p.end.x) for p in segments) + 1)
    ax.set_ylim(min(float(p.start.y) for p in segments) - 1, max(float(p.end.y) for p in segments) + 1)
    ax.set_aspect('equal', adjustable='box')
    ax.legend()

    def update(frame):
        x_pos, active_segments, intersections = event_states[frame]

        # Update sweep line
        if x_pos is not None:
            sweep_line_plot.set_data([float(x_pos), float(x_pos)], ax.get_ylim())

        # Update active segment points (where they cross sweep line)
        act_xs, act_ys = [], []
        if x_pos is not None:
            for seg in active_segments:
                try:
                    y_val = seg.y_at_x(x_pos)
                    act_xs.append(float(x_pos))
                    act_ys.append(float(y_val))
                except ValueError:
                    pass
        active_dots.set_data(act_xs, act_ys)

        # Update intersections
        int_xs = [float(p.x) for p, _ in intersections]
        int_ys = [float(p.y) for p, _ in intersections]
        inters_plot.set_data(int_xs, int_ys)

        return sweep_line_plot, active_dots, inters_plot

    ani = FuncAnimation(fig, update, frames=len(event_states), interval=500, blit=True, repeat=False)

    if save:
        ani.save(f"sweep_line_testcase{test_case}.gif", writer='pillow', fps=1)
    else:
        plt.show()


if __name__ == "__main__":
    animate_sweep_line(test_case=1, save=False)
#    animate_sweep_line(test_case=2, save=True)
