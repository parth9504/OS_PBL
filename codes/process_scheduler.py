import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
from matplotlib import cycler
import io
from PIL import Image
import pandas as pd
import base64
from dotenv import load_dotenv
import google.generativeai as genai
import os

# Dynamically generate colors for processes
def generate_colors(processes):
    # Use a color cycle for dynamic color generation
    color_cycle = cycler("color", plt.cm.tab10.colors)  # 'tab10' colormap
    color_iter = iter(color_cycle)  # Create an iterator for color cycling

    # Map each process PID to a unique color
    for p in processes:
        p['color'] = next(color_iter)["color"]
    return processes



#finding tat,wt,rt,cpu utilization....
def calculate_metrics(processes):
    # Function to calculate TAT, WT, and RT for each process
    for p in processes:
        p['tat'] = p['completion'] - p['arrival']  # Turnaround Time
        p['wt'] = p['tat'] - p['burst']  # Waiting Time
        p['rt'] = p['start'] - p['arrival']  # Response Time
    
    # Calculate average values
    avg_tat = sum(p['tat'] for p in processes) / len(processes)
    avg_wt = sum(p['wt'] for p in processes) / len(processes)
    avg_rt = sum(p['rt'] for p in processes) / len(processes)
    
    # Calculate CPU Utilization
    total_burst_time = sum(p['burst'] for p in processes)
    cpu_utilization = (total_burst_time / processes[-1]['completion']) * 100
    
    return processes, avg_tat, avg_wt, avg_rt, cpu_utilization

# FCFS
def fcfs(processes):
    processes.sort(key=lambda x: (x['arrival'], x['pid']))
    time = 0
    blocks = []
    for p in processes:
        p['start'] = max(time, p['arrival'])
        time = p['start'] + p['burst']
        p['completion'] = time
        # Record each process block at each time slice
        for t in range(p['start'], p['completion']):
            blocks.append({
                "name": f"P{p['pid']}",
                "start": t,
                "end": t + 1,
                "color": p['color']
            })
    return blocks

# SJF
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
            p['completion'] = time
            # Record each process block at each time slice
            for t in range(p['start'], p['completion']):
                blocks.append({
                    "name": f"P{p['pid']}",
                    "start": t,
                    "end": t + 1,
                    "color": p['color']
                })
            completed += 1
        else:
            time += 1
    return blocks

# SRTF
def srtf(processes):
    n = len(processes)
    time = 0
    completed = 0
    for p in processes:
        p['remaining'] = p['burst']
    blocks = []
    current = None
    ready = []
    while completed < n:
        for p in processes:
            if p['arrival'] == time:
                ready.append(p)
        if current and current['remaining'] == 0:
            current['completion'] = time
            completed += 1
            current = None
        if ready:
            if current:
                ready.append(current)
            ready = [p for p in ready if p['remaining'] > 0]
            ready.sort(key=lambda x: (x['remaining'], x['pid']))
            current = ready.pop(0)
            if 'start' not in current:
                current['start'] = time
        if current:
            for t in range(time, time + 1):
                blocks.append({
                    "name": f"P{current['pid']}",
                    "start": t,
                    "end": t + 1,
                    "color": current['color']
                })
            current['remaining'] -= 1
        time += 1
    return blocks


# LJF
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
            p['completion'] = time
            # Record each process block at each time slice
            for t in range(p['start'], p['completion']):
                blocks.append({
                    "name": f"P{p['pid']}",
                    "start": t,
                    "end": t + 1,
                    "color": p['color']
                })
            completed += 1
        else:
            time += 1
    return blocks


# LRTF
def lrtf(processes):
    n = len(processes)
    time = 0
    completed = 0
    for p in processes:
        p['remaining'] = p['burst']
    blocks = []
    current = None
    ready = []
    while completed < n:
        for p in processes:
            if p['arrival'] == time:
                ready.append(p)
        if current and current['remaining'] == 0:
            current['completion'] = time
            completed += 1
            current = None
        if ready:
            if current:
                ready.append(current)
            ready = [p for p in ready if p['remaining'] > 0]
            ready.sort(key=lambda x: (-x['remaining'], x['pid']))
            current = ready.pop(0)
            if 'start' not in current:
                current['start'] = time
        if current:
            for t in range(time, time + 1):
                blocks.append({
                    "name": f"P{current['pid']}",
                    "start": t,
                    "end": t + 1,
                    "color": current['color']
                })
            current['remaining'] -= 1
        time += 1
    return blocks


# Round Robin
def round_robin(processes, quantum):
    from collections import deque
    queue = deque()
    time = 0
    for p in processes:
        p['remaining'] = p['burst']
        p['started'] = False
    completed = 0
    blocks = []
    processes.sort(key=lambda x: (x['arrival'], x['pid']))
    i = 0
    current = None

    while completed < len(processes) or queue:
        while i < len(processes) and processes[i]['arrival'] <= time:
            queue.append(processes[i])
            i += 1
        if current:
            queue.append(current)
            current = None
        if queue:
            p = queue.popleft()
            if not p['started']:
                p['start'] = time
                p['started'] = True
            exec_time = min(quantum, p['remaining'])
            for t in range(time, time + exec_time):
                blocks.append({
                    "name": f"P{p['pid']}",
                    "start": t,
                    "end": t + 1,
                    "color": p['color']
                })
            time += exec_time
            p['remaining'] -= exec_time
            while i < len(processes) and processes[i]['arrival'] <= time:
                queue.append(processes[i])
                i += 1
            if p['remaining'] > 0:
                current = p
            else:
                p['completion'] = time
                completed += 1
        else:
            time += 1
    return blocks


def generate_animation(blocks):
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.set_ylim(0, 1)
    ax.set_xlim(0, max(block["end"] for block in blocks) + 1)
    ax.set_xlabel("Time")
    ax.set_yticks([])
    ax.set_title("Process Execution Timeline")

    frames = []  # List to store frames for GIF generation

    for i in range(len(blocks)):
        ax.clear()
        ax.set_ylim(0, 1)
        ax.set_xlim(0, max(block["end"] for block in blocks) + 1)
        ax.set_xlabel("Time")
        ax.set_yticks([])
        ax.set_title("Process Execution Timeline")

        # Update bars and process names in each frame
        for block in blocks[:i + 1]:
            ax.broken_barh([(block["start"], block["end"] - block["start"])], (0.4, 0.2), facecolors=block["color"])
            ax.text(block["start"] + 0.1, 0.45, block["name"], va='center', ha='left', color='white', fontsize=10, fontweight='bold')

        # Capture the current frame as an image
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        frame = Image.open(buf)
        frames.append(frame)

    # Save frames as a GIF in memory
    gif_buf = io.BytesIO()
    frames[0].save(gif_buf, format="GIF", append_images=frames[1:], save_all=True, duration=1000, loop=0)
    gif_buf.seek(0)
    
    # Convert the gif to base64 for embedding in Streamlit
    gif_data = base64.b64encode(gif_buf.read()).decode()
    return gif_data

def main():
    st.title("🧠 Process Scheduling Simulator")
    st.markdown("Enter process details in the table below:")
    load_dotenv()
    genai.configure(api_key=os.getenv("API_KEY"))
    model = genai.GenerativeModel("gemini-2.0-flash")

    # Input: Number of processes
    num_processes = st.number_input("Number of Processes", min_value=1, max_value=10, value=4)

    # Create default data
    default_data = {
        "PID": [i + 1 for i in range(num_processes)],
        "Arrival Time": [0 for _ in range(num_processes)],
        "Burst Time": [1 for _ in range(num_processes)]
    }

    df = pd.DataFrame(default_data)

    # Editable table
    edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

    processes=[]
    # Submit button for the form
    for _, row in edited_df.iterrows():
        process = {
            "pid": int(row["PID"]),
            "arrival": int(row["Arrival Time"]),
            "burst": int(row["Burst Time"]),
        }
        processes.append(process)

    # Generate unique colors for each process
    processes = generate_colors(processes)
    
    if st.button("Find the optimal algorithm"):
        prompt = f"""Given the following processes with their arrival and burst times:\n"""
        for p in processes:
            prompt += f"Process {p['pid']} - Arrival: {p['arrival']}, Burst: {p['burst']}\n"
        prompt += "Which CPU scheduling algorithm among FCFS, SJF,LJF (preemptive/non-preemptive), Round Robin would be best suited for this case and why?Be concise."
    
        # Call Gemini API here
        response = model.generate_content(prompt)
        raw = response.text.strip()
        st.info(f"💡 **AI Suggestion:** {raw}")

    # Select Scheduling Algorithm
    algorithm = st.selectbox("Select Scheduling Algorithm", 
                            ["FCFS", "SJF", "SRTF", "LJF", "LRTF", "Round Robin"])

    quantum = 0
        
    if algorithm == "Round Robin":
        quantum = st.number_input("Enter Time Quantum for Round Robin", min_value=1, max_value=100, value=4)
        
        if st.button("Find the optimal value of time quanta"):
            prompt = f"""Given the following processes with their arrival and burst times:\n"""
            for p in processes:
                prompt += f"Process {p['pid']} - Arrival: {p['arrival']}, Burst: {p['burst']}\n"
            prompt += "What should be the optimal value of time quanta for round robin algorithm here"
            response = model.generate_content(prompt)
            raw = response.text.strip()
            st.info(f"💡 **AI Suggestion:** {raw}")
    # Button to calculate and display results
    if st.button("Simulate Scheduling"):
            blocks = []
            
            # Apply the selected scheduling algorithm
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

            # Calculate metrics for the scheduling
            processes, avg_tat, avg_wt, avg_rt, cpu_utilization = calculate_metrics(processes)

            # Display results
            st.write(f"### Results for {algorithm} Scheduling")
            st.write(f"Average Turnaround Time (TAT): {avg_tat:.2f}")
            st.write(f"Average Waiting Time (WT): {avg_wt:.2f}")
            st.write(f"Average Response Time (RT): {avg_rt:.2f}")
            st.write(f"CPU Utilization: {cpu_utilization:.2f}%")

            # Generate and display the Gantt chart animation
            gif_data = generate_animation(blocks)
            st.image(f"data:image/gif;base64,{gif_data}", caption="Process Scheduling Animation", use_column_width=True)

if __name__ == "__main__":
    main()