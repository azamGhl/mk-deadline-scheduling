# Fixed-Priority Scheduling with (m,k)-Deadlines and Standby-Sparing

This project is a simple Python implementation of a fixed-priority scheduler (Rate-Monotonic style) for a standby-sparing real-time system with (m,k)-firm deadlines.

The code is based on ideas from the following paper (partial replication of one approach):

> Linwei Niu and Dakai Zhu,  
> "Fixed-Priority Scheduling for Reliable and Energy-Aware (m,k)-Deadlines Enforcement With Standby-Sparing",  
> IEEE TCAD, 2021.

## What this code does

- Reads a periodic task set from the console:  
  - Period (Pi)  
  - Worst Case Execution Time (Ci)  
  - Relative deadline (Di)  
  - (mi, ki) parameters  

- Computes the hyperperiod of the task set  
- Simulates fixed-priority scheduling over the hyperperiod  
- Generates a Gantt chart of the basic schedule  
- Builds job-level information (release, start, finish, deadline, response time)  
- Classifies jobs as **mandatory** or **optional** based on (m,k)  
- Computes:
  - flexibility degree of jobs  
  - a simple promotion time  
  - postponed release times  
- Applies a selective approach for:
  - Primary processor  
  - Spare processor  
- Draws Gantt charts for both processors.

> Note: this is a **student project / partial replication**, not an official implementation of the paper. Energy models and full fault-handling are not implemented.

## Requirements

- Python 3.8+
- `matplotlib`
- `numpy`

You can install the Python packages with:

```bash
pip install matplotlib numpy
