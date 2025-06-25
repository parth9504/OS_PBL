import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import io
from PIL import Image
import pandas as pd
import base64
from dotenv import load_dotenv
import google.generativeai as genai
import os
try:
    import seaborn as sns
    seaborn_available = True
except ImportError:
    seaborn_available = False

# Set Matplotlib style with fallback
if seaborn_available:
    sns.set_style("whitegrid")  # Use Seaborn's whitegrid style for better visuals
else:
    plt.style.use('ggplot')  # Fallback to Matplotlib's ggplot style if seaborn is not installed

# Generate distinct colors for processes
def generate_colors(processes):
    if seaborn_available:
        colors = sns.color_palette("husl", len(processes))
    else:
        colors = plt.cm.tab10.colors[:len(processes)]  # Fallback to Matplotlib's tab10 colormap
    for i, p in enumerate(processes):
        p['color'] = colors[i]
    return processes

# Calculate metrics including I/O considerations
def calculate_metrics(processes):
    for p in processes:
        p['tat'] = p['completion'] - p['arrival']
        p['wt'] = p['tat'] - p['burst'] - (p['io_time'] if p.get('io_time') else 0)
        p['rt'] = p['start'] - p['arrival']
    
    avg_tat = sum(p['tat'] for p in processes) / len(processes)
    avg_wt = sum(p['wt'] for p in processes) / len(processes)
    avg_rt = sum(p['rt'] for p in processes) / len(processes)
    
    total_cpu_time = sum(p['burst'] for p in processes)
    cpu_utilization = (total_cpu_time / processes[-1]['completion']) * 100
    
    return processes, avg_tat, avg_wt, avg_rt, cpu_utilization

# FCFS with I/O
def fcfs(processes):
    processes.sort(key=lambda x: (x['arrival'], x['pid']))
    time = 0
    blocks = []
    for p in processes:
        p['start'] = max(time, p['arrival'])
        time = p['start'] + p['burst']
        p['completion'] = time + (p['io_time'] if p.get('io_time') else 0)
        for t in range(p['start'], p['start'] + p['burst']):
            blocks.append({"name": f"P{p['pid']}", "start": t, "end": t + 1, "color": p['color'], "type": "CPU"})
        if p.get('io_time', 0) > 0:
            for t in range(p['start'] + p['burst'], p['completion']):
                blocks.append({"name": f"P{p['pid']}(I/O)", "start": t, "end": t + 1, "color": adjust_color(p['color'], 0.7), "type": "IO"})
    return blocks

# SJF with I/O
def sjfs(processes):
    n = len(processes)
    completed = 0
    time = 0
    blocks = []
    ready = []
    while completed < n:
        for p in processes:
            if p['arrival'] <= time and 'start' not in p and p not in ready:
                ready.append(p)
        if ready:
            ready.sort(key=lambda x: (x['burst'], x['pid']))
            p = ready.pop(0)
            p['start'] = time
            time += p['burst']
            p['completion'] = time + (p['io_time'] if p.get('io_time') else 0)
            for t in range(p['start'], p['start'] + p['burst']):
                blocks.append({"name": f"P{p['pid']}", "start": t, "end": t + 1, "color": p['color'], "type": "CPU"})
            if p.get('io_time', 0) > 0:
                for t in range(p['start'] + p['burst'], p['completion']):
                    blocks.append({"name": f"P{p['pid']}(I/O)", "start": t, "end": t + 1, "color": adjust_color(p['color'], 0.7), "type": "IO"})
            completed += 1
        else:
            time += 1
    return blocks

# SRTF with I/O
def srtf(processes):
    n = len(processes)
    time = 0
    completed = 0
    for p in processes:
        p['remaining'] = p['burst']
        p['io_remaining'] = p.get('io_time', 0)
    blocks = []
    current = None
    ready = []
    while completed < n:
        for p in processes:
            if p['arrival'] == time:
                ready.append(p)
        if current and current['remaining'] == 0:
            if current['io_remaining'] > 0:
                for t in range(time, time + current['io_remaining']):
                    blocks.append({"name": f"P{current['pid']}(I/O)", "start": t, "end": t + 1, 
                                 "color": adjust_color(current['color'], 0.7), "type": "IO"})
                time += current['io_remaining']
                current['completion'] = time
                current['io_remaining'] = 0
                completed += 1
                current = None
        if ready:
            if current:
                ready.append(current)
            ready = [p for p in ready if p['remaining'] > 0]
            ready.sort(key=lambda x: (x['remaining'], x['pid']))
            current = ready.pop(0) if ready else None
            if current and 'start' not in current:
                current['start'] = time
        if current:
            blocks.append({"name": f"P{current['pid']}", "start": time, "end": time + 1, 
                          "color": current['color'], "type": "CPU"})
            current['remaining'] -= 1
        time += 1
    return blocks

# LJF with I/O
def ljfs(processes):
    n = len(processes)
    completed = 0
    time = 0
    blocks = []
    ready = []
    while completed < n:
        for p in processes:
            if p['arrival'] <= time and 'start' not in p and p not in ready:
                ready.append(p)
        if ready:
            ready.sort(key=lambda x: (-x['burst'], x['pid']))
            p = ready.pop(0)
            p['start'] = time
            time += p['burst']
            p['completion'] = time + (p['io_time'] if p.get('io_time') else 0)
            for t in range(p['start'], p['start'] + p['burst']):
                blocks.append({"name": f"P{p['pid']}", "start": t, "end": t + 1, "color": p['color'], "type": "CPU"})
            if p.get('io_time', 0) > 0:
                for t in range(p['start'] + p['burst'], p['completion']):
                    blocks.append({"name": f"P{p['pid']}(I/O)", "start": t, "end": t + 1, 
                                 "color": adjust_color(p['color'], 0.7), "type": "IO"})
            completed += 1
        else:
            time += 1
    return blocks

# LRTF with I/O
def lrtf(processes):
    n = len(processes)
    time = 0
    completed = 0
    for p in processes:
        p['remaining'] = p['burst']
        p['io_remaining'] = p.get('io_time', 0)
    blocks = []
    current = None
    ready = []
    while completed < n:
        for p in processes:
            if p['arrival'] == time:
                ready.append(p)
        if current and current['remaining'] == 0:
            if current['io_remaining'] > 0:
                for t in range(time, time + current['io_remaining']):
                    blocks.append({"name": f"P{current['pid']}(I/O)", "start": t, "end": t + 1, 
                                 "color": adjust_color(current['color'], 0.7), "type": "IO"})
                time += current['io_remaining']
                current['completion'] = time
                current['io_remaining'] = 0
                completed += 1
                current = None
        if ready:
            if current:
                ready.append(current)
            ready = [p for p in ready if p['remaining'] > 0]
            ready.sort(key=lambda x: (-x['remaining'], x['pid']))
            current = ready.pop(0) if ready else None
            if current and 'start' not in current:
                current['start'] = time
        if current:
            blocks.append({"name": f"P{current['pid']}", "start": time, "end": time + 1, 
                          "color": current['color'], "type": "CPU"})
            current['remaining'] -= 1
        time += 1
    return blocks

# Round Robin with I/O
def round_robin(processes, quantum):
    from collections import deque
    queue = deque()
    time = 0
    completed = 0
    blocks = []
    
    # Initialize processes
    for p in processes:
        p['remaining'] = p['burst']
        p['io_remaining'] = p.get('io_time', 0)
        p['started'] = False
    
    # Sort processes by arrival time for initial queueing
    processes.sort(key=lambda x: (x['arrival'], x['pid']))
    next_process_index = 0
    
    while completed < len(processes):
        # Add all processes that have arrived by current time
        while next_process_index < len(processes) and processes[next_process_index]['arrival'] <= time:
            queue.append(processes[next_process_index])
            next_process_index += 1
        
        if queue:
            p = queue.popleft()
            if not p['started']:
                p['start'] = time
                p['started'] = True
            
            # Execute for quantum or remaining time, whichever is smaller
            exec_time = min(quantum, p['remaining'])
            for t in range(time, time + exec_time):
                blocks.append({
                    "name": f"P{p['pid']}",
                    "start": t,
                    "end": t + 1,
                    "color": p['color'],
                    "type": "CPU"
                })
            time += exec_time
            p['remaining'] -= exec_time
            
            # Handle I/O if process is complete
            if p['remaining'] == 0:
                if p['io_remaining'] > 0:
                    for t in range(time, time + p['io_remaining']):
                        blocks.append({
                            "name": f"P{p['pid']}(I/O)",
                            "start": t,
                            "end": t + 1,
                            "color": adjust_color(p['color'], 0.7),
                            "type": "IO"
                        })
                    time += p['io_remaining']
                    p['io_remaining'] = 0
                p['completion'] = time
                completed += 1
            else:
                # Re-queue the process if not completed
                queue.append(p)
        else:
            # No processes ready; advance time
            time += 1
    
    return blocks


# Multi-Level Feedback Queue (MLFQ) with I/O
def mlfq(processes, queue_configs):
    from collections import deque
    queues = [deque() for _ in range(3)]  # Three queues: 0 (highest priority) to 2 (lowest)
    time = 0
    completed = 0
    blocks = []

    # Initialize processes
    for p in processes:
        p['remaining'] = p['burst']
        p['io_remaining'] = p.get('io_time', 0)
        p['started'] = False
        p['queue'] = 0  # Start in Queue 1 (highest priority)

    # Sort processes by arrival time for initial queueing
    processes.sort(key=lambda x: (x['arrival'], x['pid']))
    next_process_index = 0

    while completed < len(processes):
        # Add all processes that have arrived by current time to Queue 1
        while next_process_index < len(processes) and processes[next_process_index]['arrival'] <= time:
            p = processes[next_process_index]
            p['queue'] = 0
            queues[0].append(p)
            next_process_index += 1

        # Find the highest-priority non-empty queue
        current_queue = -1
        for q in range(3):
            if queues[q]:
                current_queue = q
                break

        if current_queue >= 0:
            algo, quantum = queue_configs[current_queue]
            p = queues[current_queue].popleft()

            if not p['started']:
                p['start'] = time
                p['started'] = True

            if algo == "Round Robin":
                # Execute for quantum or remaining time, whichever is smaller
                exec_time = min(quantum, p['remaining'])
                for t in range(time, time + exec_time):
                    blocks.append({
                        "name": f"P{p['pid']}",
                        "start": t,
                        "end": t + 1,
                        "color": p['color'],
                        "type": "CPU"
                    })
                time += exec_time
                p['remaining'] -= exec_time

                if p['remaining'] > 0:
                    # Demote to next queue if quantum used up
                    next_queue = min(p['queue'] + 1, 2)
                    p['queue'] = next_queue
                    queues[next_queue].append(p)
                else:
                    # Handle I/O if process is complete
                    if p['io_remaining'] > 0:
                        for t in range(time, time + p['io_remaining']):
                            blocks.append({
                                "name": f"P{p['pid']}(I/O)",
                                "start": t,
                                "end": t + 1,
                                "color": adjust_color(p['color'], 0.7),
                                "type": "IO"
                            })
                        time += p['io_remaining']
                        p['io_remaining'] = 0
                    p['completion'] = time
                    completed += 1
                    del p['queue']
            elif algo == "FCFS":
                # Execute entire remaining CPU burst
                exec_time = p['remaining']
                for t in range(time, time + exec_time):
                    blocks.append({
                        "name": f"P{p['pid']}",
                        "start": t,
                        "end": t + 1,
                        "color": p['color'],
                        "type": "CPU"
                    })
                time += exec_time
                p['remaining'] = 0

                # Handle I/O if process is complete
                if p['io_remaining'] > 0:
                    for t in range(time, time + p['io_remaining']):
                        blocks.append({
                            "name": f"P{p['pid']}(I/O)",
                            "start": t,
                            "end": t + 1,
                            "color": adjust_color(p['color'], 0.7),
                            "type": "IO"
                        })
                    time += p['io_remaining']
                    p['io_remaining'] = 0
                p['completion'] = time
                completed += 1
                del p['queue']
        else:
            # No processes ready; advance time
            time += 1

    return blocks

# Utility to adjust color brightness
def adjust_color(color, factor):
    return tuple(min(max(c * factor, 0), 1) for c in color[:3])

# Enhanced animation with CPU/IO distinction
def generate_animation(blocks):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_ylim(0, 1)
    max_time = max(block["end"] for block in blocks) + 1
    ax.set_xlim(0, max_time)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_yticks([])
    ax.set_title("CPU and I/O Scheduling Timeline", fontsize=14, pad=10)
    
    # Add grid for better readability
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    
    frames = []
    for i in range(len(blocks) + 10):  # Extra frames for smooth ending
        ax.clear()
        ax.set_ylim(0, 1)
        ax.set_xlim(0, max_time)
        ax.set_xlabel("Time", fontsize=12)
        ax.set_yticks([])
        ax.set_title("CPU and I/O Scheduling Timeline", fontsize=14, pad=10)
        ax.grid(True, axis='x', linestyle='--', alpha=0.3)

        for block in blocks[:min(i + 1, len(blocks))]:
            height = 0.4 if block['type'] == "CPU" else 0.2
            y_pos = 0.4 if block['type'] == "CPU" else 0.1
            ax.broken_barh([(block["start"], block["end"] - block["start"])], 
                          (y_pos, height), 
                          facecolors=block["color"], 
                          edgecolors='black', 
                          linewidth=0.5)
            ax.text(block["start"] + 0.1, y_pos + height/2, block["name"], 
                    va='center', ha='left', color='white', fontsize=8, fontweight='bold')

        # Add legend
        ax.text(0.02, 0.95, "CPU: Top | I/O: Bottom", transform=ax.transAxes, 
                fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        frame = Image.open(buf)
        frames.append(frame)

    gif_buf = io.BytesIO()
    frames[0].save(gif_buf, format="GIF", append_images=frames[1:], save_all=True, duration=500, loop=0)
    gif_buf.seek(0)
    gif_data = base64.b64encode(gif_buf.read()).decode()
    
    plt.close(fig)
    return gif_data

def main():
    st.set_page_config(page_title="Process Scheduling Simulator", layout="wide")
    st.title("🖥️ Advanced Process Scheduling Simulator")
    st.markdown("Simulate CPU scheduling algorithms with I/O burst times. Enter process details below:")

    load_dotenv()
    genai.configure(api_key=os.getenv("API_KEY"))
    model = genai.GenerativeModel("gemini-2.0-flash")

    # Modern UI layout with columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Process Configuration")
        num_processes = st.number_input("Number of Processes", min_value=1, max_value=10, value=4, key="num_procs")
        
        default_data = {
            "PID": [i + 1 for i in range(num_processes)],
            "Arrival Time": [0 for _ in range(num_processes)],
            "CPU Burst Time": [1 for _ in range(num_processes)],
            "I/O Time": [0 for _ in range(num_processes)]
        }
        df = pd.DataFrame(default_data)
        
        edited_df = st.data_editor(
            df, 
            num_rows="dynamic", 
            use_container_width=True,
            column_config={
                "PID": st.column_config.NumberColumn(min_value=1, step=1),
                "Arrival Time": st.column_config.NumberColumn(min_value=0, step=1),
                "CPU Burst Time": st.column_config.NumberColumn(min_value=1, step=1),
                "I/O Time": st.column_config.NumberColumn(min_value=0, step=1)
            }
        )

    with col2:
        st.subheader("Algorithm Settings")
        algorithm = st.selectbox(
            "Select Scheduling Algorithm", 
            ["FCFS", "SJF", "SRTF", "LJF", "LRTF", "Round Robin"],
            help="Choose a CPU scheduling algorithm to simulate"
        )
        
        quantum = 0
        if algorithm == "Round Robin":
            quantum = st.number_input(
                "Time Quantum", 
                min_value=1, 
                max_value=100, 
                value=4, 
                key="quantum",
                help="Time slice for Round Robin scheduling"
            )

    processes = []
    for _, row in edited_df.iterrows():
        process = {
            "pid": int(row["PID"]),
            "arrival": int(row["Arrival Time"]),
            "burst": int(row["CPU Burst Time"]),
            "io_time": int(row["I/O Time"])
        }
        processes.append(process)
    
    processes = generate_colors(processes)

def main():
    st.set_page_config(page_title="Process Scheduling Simulator", layout="wide")
    st.title("🖥️ AI Powered Process Scheduling Simulator")
    st.markdown("Simulate CPU scheduling algorithms with I/O burst times. Enter process details below:")

    load_dotenv()
    genai.configure(api_key=os.getenv("API_KEY"))
    model = genai.GenerativeModel("gemini-2.0-flash")

    # Modern UI layout with columns
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Process Configuration")
        num_processes = st.number_input("Number of Processes", min_value=1, max_value=10, value=4, key="num_procs")
        
        default_data = {
            "PID": [i + 1 for i in range(num_processes)],
            "Arrival Time": [0 for _ in range(num_processes)],
            "CPU Burst Time": [1 for _ in range(num_processes)],
            "I/O Time": [0 for _ in range(num_processes)]
        }
        df = pd.DataFrame(default_data)
        
        edited_df = st.data_editor(
            df, 
            num_rows="dynamic", 
            use_container_width=True,
            column_config={
                "PID": st.column_config.NumberColumn(min_value=1, step=1),
                "Arrival Time": st.column_config.NumberColumn(min_value=0, step=1),
                "CPU Burst Time": st.column_config.NumberColumn(min_value=1, step=1),
                "I/O Time": st.column_config.NumberColumn(min_value=0, step=1)
            }
        )

    with col2:
        st.subheader("Algorithm Settings")
        algorithm = st.selectbox(
            "Select Scheduling Algorithm", 
            ["FCFS", "SJF", "SRTF", "LJF", "LRTF", "Round Robin", "MLFQ"],
            help="Choose a CPU scheduling algorithm to simulate"
        )
        
        quantum = 0
        queue_configs = None
        if algorithm == "Round Robin":
            quantum = st.number_input(
                "Time Quantum", 
                min_value=1, 
                max_value=100, 
                value=4, 
                key="quantum",
                help="Time slice for Round Robin scheduling"
            )
        elif algorithm == "MLFQ":
            st.write("**MLFQ Queue Configurations**")
            queue_configs = []
            for q in range(1, 4):
                st.write(f"Queue {q} (Priority {4-q})")
                algo = st.selectbox(
                    f"Algorithm for Queue {q}",
                    ["Round Robin", "FCFS"],
                    key=f"mlfq_algo_q{q}"
                )
                quantum = 0
                if algo == "Round Robin":
                    quantum = st.number_input(
                        f"Time Quantum for Queue {q}",
                        min_value=1,
                        max_value=100,
                        value=2*q,  # Default: 2, 4, 8
                        key=f"mlfq_quantum_q{q}"
                    )
                queue_configs.append((algo, quantum))

    processes = []
    for _, row in edited_df.iterrows():
        process = {
            "pid": int(row["PID"]),
            "arrival": int(row["Arrival Time"]),
            "burst": int(row["CPU Burst Time"]),
            "io_time": int(row["I/O Time"])
        }
        processes.append(process)

    processes = generate_colors(processes)

    # AI Suggestions
    with st.expander("AI-Powered Suggestions", expanded=False):
        if st.button("Find Optimal Algorithm"):
            prompt = f"""Given processes with:
            {', '.join([f"P{p['pid']}: Arrival={p['arrival']}, CPU Burst={p['burst']}, I/O={p['io_time']}" for p in processes])}
            Which CPU scheduling algorithm (FCFS, SJF, SRTF, LJF, LRTF, Round Robin, MLFQ) is best suited and why? Be concise."""
            response = model.generate_content(prompt)
            st.info(f"💡 **AI Suggestion:** {response.text.strip()}")

        if algorithm == "Round Robin" and st.button("Optimize Time Quantum"):
            prompt = f"""Given processes with:
            {', '.join([f"P{p['pid']}: Arrival={p['arrival']}, CPU Burst={p['burst']}, I/O={p['io_time']}" for p in processes])}
            Suggest the optimal time quantum for Round Robin scheduling. Be concise."""
            response = model.generate_content(prompt)
            st.info(f"💡 **AI Suggestion:** {response.text.strip()}")

        if algorithm == "MLFQ" and st.button("Optimize MLFQ Configuration"):
            prompt = f"""Given processes with:
            {', '.join([f"P{p['pid']}: Arrival={p['arrival']}, CPU Burst={p['burst']}, I/O={p['io_time']}" for p in processes])}
            Suggest the optimal configuration for MLFQ (algorithms and time quanta for three queues). Be concise."""
            response = model.generate_content(prompt)
            st.info(f"💡 **AI Suggestion:** {response.text.strip()}")

    # Simulate and Display Results
    if st.button("Run Simulation", type="primary"):
        with st.spinner("Simulating..."):
            blocks = []
            if algorithm == "FCFS":
                blocks = fcfs(processes)
            elif algorithm == "SJF":
                blocks = sjfs(processes)
            elif algorithm == "SRTF":
                blocks = srtf(processes)
            elif algorithm == "LJF":
                blocks = ljfs(processes)
            elif algorithm == "LRTF":
                blocks = lrtf(processes)
            elif algorithm == "Round Robin":
                blocks = round_robin(processes, quantum)
            elif algorithm == "MLFQ":
                blocks = mlfq(processes, queue_configs)

            processes, avg_tat, avg_wt, avg_rt, cpu_utilization = calculate_metrics(processes)

            # Display Results in a clean format
            st.subheader(f"Results for {algorithm} Scheduling")
            col_metrics, col_gantt = st.columns([1, 2])
            
            with col_metrics:
                st.metric("Avg Turnaround Time", f"{avg_tat:.2f}")
                st.metric("Avg Waiting Time", f"{avg_wt:.2f}")
                st.metric("Avg Response Time", f"{avg_rt:.2f}")
                st.metric("CPU Utilization", f"{cpu_utilization:.2f}%")
                
                # Process details table
                st.write("**Process Details**")
                result_df = pd.DataFrame([
                    {"PID": p['pid'], "Arrival": p['arrival'], "CPU Burst": p['burst'], 
                        "I/O Time": p.get('io_time', 0), "Start": p['start'], 
                        "Completion": p['completion'], "TAT": p['tat'], 
                        "WT": p['wt'], "RT": p['rt']}
                    for p in processes
                ])
                st.dataframe(result_df, use_container_width=True)

            with col_gantt:
                gif_data = generate_animation(blocks)
                st.image(f"data:image/gif;base64,{gif_data}", 
                        caption="CPU (Top) and I/O (Bottom) Scheduling Timeline", 
                        use_column_width=True)
if __name__ == "__main__":
    main()