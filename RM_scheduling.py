#!/usr/bin/env python3
# ------------------------------------------
# RM_scheduling.py: Fixed priority Scheduler in Stand_by_sparing
import json
import copy
from sys import *
from math import gcd
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import statistics as st
from collections import deque
from collections import defaultdict
import math

tasks = dict()
RealTime_task = dict()
metrics = defaultdict(dict)
job_list = defaultdict(dict)
Mandatory_jobs = []
Optional_jobs = []
d = dict()
dList = []
T = []
C = []
U = []
# For gantt chart
y_axis = []
from_x = []
to_x = []

releaseTime = []
startTime = []
finishTime = []
avg_respTime = []


def Read_data():
    """
    Reading the details of the tasks to be scheduled from the user as
    Number of tasks n:
    Period of task P:
    Worst case excecution time WCET:
    """
    global n
    global hp
    global tasks
    global dList

    dList = {}

    n = int(input("\n \t\tEnter number of Tasks:"))
    # Storing data in a dictionary
    for i in range(n):
        dList["TASK_%d" % i] = {"start": [], "finish": []}

    dList["TASK_IDLE"] = {"start": [], "finish": []}

    for i in range(n):
        tasks[i] = {}
        print("\n\n\n Enter Period of task T", i, ":")
        p = input()
        tasks[i]["Period"] = int(p)

        print("Enter the WCET of task C", i, ":")
        w = input()
        tasks[i]["WCET"] = int(w)

        print("Enter the absolute deadline of task d", i, ":")
        d = input()
        tasks[i]["deadline"] = int(d)

        print("Enter mi for task ", i, ":")
        m = input()
        tasks[i]["m"] = int(m)

        print("Enter ki for task ", i, ":")
        m = input()
        tasks[i]["k"] = int(m)


    # Writing the dictionary into a JSON file
    with open('tasks.json', 'w') as outfile:
        json.dump(tasks, outfile, indent=4)


def Hyperperiod():
    """
    Calculates the hyper period of the tasks to be scheduled
    """
    temp = []
    for i in range(n):
        temp.append(tasks[i]["Period"])
    HP = temp[0]
    for i in temp[1:]:
        HP = HP * i // gcd(HP, i)
    print("\n Hyperperiod:", HP)
    return HP


def Schedulablity():
    """
    Calculates the utilization factor of the tasks to be scheduled
    and then checks for the schedulablity and then returns true is
    schedulable else false.
    """
    for i in range(n):
        T.append(int(tasks[i]["Period"]))
        C.append(int(tasks[i]["WCET"]))
        u = int(C[i]) / int(T[i])
        U.append(u)

    U_factor = sum(U)
    if U_factor <= 1:
        print("\nUtilization factor: ", U_factor, "underloaded tasks")

        sched_util = n * (2 ** (1 / n) - 1)
        print("Checking condition: ", sched_util)

        count = 0
        T.sort()
        for i in range(len(T)):
            if T[i] % T[0] == 0:
                count = count + 1

        # Checking the schedulablity condition
        if U_factor <= sched_util or count == len(T):
            print("\n\tTasks are schedulable by fixed_priority Scheduling!")
            return True
        else:
            print("\n\tTasks are not schedulable by fixed_priority Scheduling!")
            return False
    print("\n\tOverloaded tasks!")
    print("\n\tUtilization factor > 1")
    return False


# def estimatePriority(RealTime_task):
#     """
#     Estimates the priority of tasks at each real time period during scheduling
#     """
#     tempPeriod = hp
#     P = -1  # Returns -1 for idle tasks
#     for i in RealTime_task.keys():
#         if (RealTime_task[i]["WCET"] != 0):
#             if (tempPeriod > RealTime_task[i]["Period"] or tempPeriod > tasks[i]["Period"]):
#                 tempPeriod = tasks[i]["Period"]  # Checks the priority of each task based on period
#                 P = i
#     return P

def estimatePriority(RealTime_task):
    """
    Estimates the priority of tasks at each real time period during scheduling
    """
    # Initialize variables
    highest_priority_index = -1 # -1 for idle tasks

    # Iterate over tasks in sequence to determine the first incomplete task
    for i in sorted(RealTime_task.keys(), key=int):
        if RealTime_task[i]["WCET"] > 0:
            highest_priority_index = int(i) # Convert string key to integer index
            break

    return highest_priority_index




def Simulation(hp):
    """
    The real time schedulng based on fixed_priority scheduling is simulated here.
    """

    # Real time scheduling are carried out in RealTime_task
    global RealTime_task
    RealTime_task = copy.deepcopy(tasks)
    # validation of schedulablity neessary condition
    for i in RealTime_task.keys():
        RealTime_task[i]["DCT"] = RealTime_task[i]["WCET"]
        if (RealTime_task[i]["WCET"] > RealTime_task[i]["Period"]):
            print(" \n\t The task can not be completed in the specified time ! ", i)

    # main loop for simulator
    for t in range(hp):

        # Determine the priority of the given tasks
        priority = estimatePriority(RealTime_task)

        if (priority != -1):  # processor is not idle
            print("\nt{}-->t{} :TASK{}".format(t, t + 1, priority))
            # Update WCET after each clock cycle
            RealTime_task[priority]["WCET"] -= 1
            # For the calculation of the metrics
            dList["TASK_%d" % priority]["start"].append(t)
            dList["TASK_%d" % priority]["finish"].append(t + 1)
            # For plotting the results
            y_axis.append("TASK%d" % priority)
            from_x.append(t)
            to_x.append(t + 1)

        else:  # processor is idle
            print("\nt{}-->t{} :IDLE".format(t, t + 1))
            # For the calculation of the metrics
            dList["TASK_IDLE"]["start"].append(t)
            dList["TASK_IDLE"]["finish"].append(t + 1)
            # For plotting the results
            y_axis.append("IDLE")
            from_x.append(t)
            to_x.append(t + 1)

        # Update Period after each clock cycle
        for i in RealTime_task.keys():
            RealTime_task[i]["Period"] -= 1
            if (RealTime_task[i]["Period"] == 0):
                RealTime_task[i] = copy.deepcopy(tasks[i])

        with open('RM_sched.json', 'w') as outfile2:
            json.dump(dList, outfile2, indent=4)


def drawGantt():
    """
    The scheduled results are displayed in the form of a
    gantt chart for the user to get better understanding
    """
    colors = ['red', 'green', 'blue', 'orange', 'yellow']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # the data is plotted from_x to to_x along y_axis
    ax = plt.hlines(y_axis, from_x, to_x, linewidth=20, color=colors[n - 1])
    plt.title('fiXed-priority scheduling ')
    plt.grid(True)
    plt.xlabel("Time")
    plt.ylabel("HIGH------------------Priority--------------------->LOW")
    plt.xticks(np.arange(min(from_x), max(to_x) + 1, 1.0))
    plt.show()


def showMetrics():
    """
    Displays the resultant metrics after scheduling such as
    average response time, the average waiting time and the
    time of first deadline miss
    """
    N = []
    avg_waitTime = []

    # Calculation of number of releases and release time
    for i in tasks.keys():
        release = int(hp) / int(tasks[i]["Period"])
        N.append(release)
        temp = []
        for j in range(int(N[i])):
            temp.append(j * int(tasks[i]["Period"]))
        # temp.append(hp)
        releaseTime.append(temp)

    # # Calculation of start time of each task
    # for j, i in enumerate(tasks.keys()):
    #     start_array, end_array = filter_out(dList["TASK_%d" % i]["start"], dList["TASK_%d" % i]["finish"], N[j])
    #     startTime.append(start_array)
    #     finishTime.append(end_array)

    for j, i in enumerate(tasks.keys()):
        start_array, end_array = filter_out(dList["TASK_%d" % i]["start"], dList["TASK_%d" % i]["finish"], N[j],tasks[i]['WCET'])
        startTime.append(start_array)
        finishTime.append(end_array)


    # Calculation of average waiting time and average response time of tasks
    for i in tasks.keys():
        avg_waitTime.append(st.mean([a_i - b_i for a_i, b_i in zip(startTime[i], releaseTime[i])]))
        avg_respTime.append(st.mean([a_i - b_i for a_i, b_i in zip(finishTime[i], releaseTime[i])]))

    # Printing the resultant metrics
    for i in tasks.keys():
        metrics[i]["Releases"] = N[i]
        metrics[i]["Period"] = tasks[i]["Period"]
        metrics[i]["WCET"] = tasks[i]["WCET"]
        metrics[i]["AvgRespTime"] = avg_respTime[i]
        metrics[i]["AvgWaitTime"] = avg_waitTime[i]

        print("\n Number of releases of task %d =" % i, int(N[i]))
        print("\n Release time of task%d = " % i, releaseTime[i])
        print("\n start time of task %d = " % i, startTime[i])
        print("\n finish time of task %d = " % i, finishTime[i])
        print("\n Average Response time of task %d = " % i, avg_respTime[i])
        print("\n Average Waiting time of task %d = " % i, avg_waitTime[i])
        print("\n")

    # Storing results into a JSON file
    with open('Metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print("\n\n\t\tScheduling of %d tasks completed succesfully...." % n)


def filter_out(start_array, finish_array, release_time,execution_time):
    """A filtering function created to create the required data struture from the simulation results"""
    new_start = []
    new_finish = []
    # Calculation of finish time and start time from simulation results
    if (release_time > 1):
        for i in range(int(release_time)):
            new_start.append(start_array[(i * execution_time)])
            new_finish.append(finish_array[((i+1) * execution_time)-1])

    else:
        new_start.append(start_array[0])
        new_finish.append(finish_array[ execution_time - 1])
    return new_start, new_finish

def calculate_E_pattern(j, K, m):
    ceil_value = math.ceil((j * K) / m) * (K / m)
    floor_value = math.floor(ceil_value)

    if j == floor_value:
        return  1
    else:
        return  0

def create_jobs():
    for i, task in tasks.items():
        task_label = i
        num_jobs = hp // task['Period']
        jobs = []
        for j in range(1, num_jobs + 1):  # Start job index at 1
            Job_label = f"j{i+1}{j}"
            release_time = (j - 1) * task['Period']
            execution_time = task['WCET']
            relative_deadline = task['deadline']
            deadline = release_time + task['deadline']
            start_time = startTime[i][j-1]
            finish_time = finishTime[i][j-1]
            response_time = finish_time - release_time
            job_type = "mandatory" if 1 <= (j % task['k']) <= task['m'] else "optional"

            # job_type = "mandatory" if j == calculate_E_pattern(j - 1, task['k'], task['m']) else "optional"  ---//E-pattern

            # Calculate FD(Ji) for the job Ji
            # FD is the number of optional jobs that can miss their deadlines consecutively after this job
            fd = 0
            for future_job in range(j + 1, j + task['k']):
                future_job_type = "mandatory" if 1 <= (future_job % task['k']) <= task['m'] else "optional"
                if future_job_type == "optional":
                    fd += 1
                else:
                    # Once a mandatory job is encountered, stop counting
                    break
            job = {
                'task': task_label,
                'Job': Job_label,
                'release_time': release_time,
                'execution_time': execution_time,
                'start_time': start_time,
                'finish_time': finish_time,
                'relative_deadline': relative_deadline,
                'deadline': deadline,
                'response_time': response_time,
                'job_type': job_type,
                'flexibility_degree': fd,
                # 'promotion_time' : promotion_time
            }
            jobs.append(job)
        job_list[task_label] = jobs
    for i , j  in job_list.items():
        print("\n", i , j)


def sort_and_iterate_jobs(job_list):
    # Flatten the list of jobs
    jobs = [job for sublist in job_list.values() for job in sublist]

    # Sort the jobs by start time
    sorted_jobs = sorted(jobs, key=lambda x: x['start_time'])

    # Iterate over the sorted jobs
    for job in sorted_jobs:
        yield job


def calculate_promotion_times(job_dict):
    # Calculate the maximum response time for each task
    for task in job_dict.values():
        max_response_time = max(job['response_time'] for job in task)

        # Add the promotion time to each job
        for job in task:
            job['promotion_time'] = job['relative_deadline'] - max_response_time

    return job_dict

def postponed_release_time(job_list):
    calculate_promotion_times(job_list)
    previous_execution_time = 0
    postponed_release = 0
    postponed_release_time = []
    job_generator = sort_and_iterate_jobs(job_list)
    for i, job in enumerate(job_generator):
        inspecting_point = []
        if(job['start_time']==0):
            inspecting_point.append(job['deadline'])
        else:
            inspecting_point.append(job['deadline'])
            inspecting_point.append(postponed_release)
        print("gg",inspecting_point)
        theta_job = []
        theta_job.append(max((t - (job['execution_time'] + previous_execution_time) - job['release_time']) for t in inspecting_point))
        print("theta_job",theta_job)
        x = theta_job[0]
        theta_task = max(x,job['promotion_time'])
        print("theta_task",theta_task)
        previous_execution_time = job['execution_time']
        postponed_release = job['release_time'] + theta_task
        job['postponed_release'] = postponed_release
        #print(postponed_release)
        postponed_release_time.append(job['Job'])
        postponed_release_time.append(postponed_release)
    return postponed_release_time

def separate_jobs_by_type(job_list):
    postponed_release_time(job_list)
    Mandatory_jobs = []
    Optional_jobs = []

    for task in job_list.values():
        for job in task:
            if (job['job_type'] == "mandatory"):
                Mandatory_jobs.append(job)
            elif job['flexibility_degree'] == 1 and job['job_type'] == "optional":
                Mandatory_jobs.append(job)
            elif job['job_type'] == "optional" and job['flexibility_degree'] != 1:
                Optional_jobs.append(job)

    return Mandatory_jobs, Optional_jobs
#--------------------selective_approach for Primary processor-------------------------
def selective_approach_primary(Mandatory_jobs):
    y_axis_primary = []
    from_x_primary = []
    to_x_primary = []
    current_time = 0

    # Sort the jobs by their start time
    Mandatory_jobs.sort(key=lambda x: x['start_time'])

    # Initialize the queue with the jobs sorted by start time
    job_queue = copy.deepcopy(Mandatory_jobs)

    while True:
        # Flag to check if any job is still running
        any_job_running = False
        for i in range(len(job_queue)):
            # If the job hasn't finished yet, execute it for one step
            while job_queue[i]['execution_time'] > 0 :
                any_job_running = True
                # Record the task identifier, start time, and end time
                y_axis_primary.append(f"TASK{job_queue[i]['task']}")
                from_x_primary.append(current_time)
                to_x_primary.append(current_time + 1)

                # Decrement the execution time and update the current time
                job_queue[i]['execution_time'] -= 1
                current_time += 1
                if i < len(job_queue) - 1:
                    if job_queue[i]['execution_time'] > 0 and job_queue[i+1]['start_time']== current_time:
                        i += 1
                        job_queue.append(job_queue[i])
                    # if (current_time < job_queue[i + 1]['start_time'] and job_queue[i]['execution_time'] == 0):
                    #     x = job_queue[i + 1]['postponed_release'] - current_time
                    #     y_axis_primary.append("IDLE")
                    #     from_x_primary.append(current_time)
                    #     to_x_primary.append(current_time + x)
                    #     # job_queue[i]['execution_time'] -= 1
                    #     current_time += job_queue[i + 1]['postponed_release'] - current_time

        if not any_job_running:
            x = hp - current_time
            y_axis_primary.append("IDLE")
            from_x_primary.append(current_time)
            to_x_primary.append(current_time + x)
            break


        if not any_job_running:
            break

    return y_axis_primary, from_x_primary, to_x_primary

def drawGanttPrimary(y_axis_primary, from_x_primary, to_x_primary):
    """
    The scheduled results are displayed in the form of a
    gantt chart for the user to get better understanding
    """
    colors = ['red', 'green', 'blue', 'orange', 'yellow']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = plt.hlines(y_axis_primary, from_x_primary, to_x_primary, linewidth=20, color=colors[3])
    plt.title('selective_approach in R_pattern (Primary Processor)')
    plt.grid(True)
    plt.xlabel("Time")
    plt.ylabel("HIGH------------------Priority--------------------->LOW")
    plt.xticks(np.arange(min(from_x_primary), max(to_x_primary) + 1, 1.0))
    plt.show()
#--------------------selective_approach for spare processor-------------------------

def selective_approach_Spare(Mandatory_jobs):
    y_axis_primary = []
    from_x_primary = []
    to_x_primary = []
    current_time = 0

    # Sort the jobs by their start time
    Mandatory_jobs.sort(key=lambda x: x['postponed_release'])

    # Initialize the queue with the jobs sorted by start time
    job_queue = copy.deepcopy(Mandatory_jobs)
    current_time = job_queue[0]['postponed_release']
    while True:
        # Flag to check if any job is still running
        any_job_running = False
        for i in range(len(job_queue)):
            # current_time = job_queue[i]['postponed_release']
            # If the job hasn't finished yet, execute it for one step
            while job_queue[i]['execution_time'] > 0 :
                any_job_running = True
                # Record the task identifier, start time, and end time
                y_axis_primary.append(f"TASK{job_queue[i]['task']}")
                from_x_primary.append(current_time)
                to_x_primary.append(current_time + 1)

                # Decrement the execution time and update the current time
                job_queue[i]['execution_time'] -= 1
                current_time += 1

                if i < len(job_queue) - 1:
                    if job_queue[i]['execution_time'] > 0 and job_queue[i + 1]['postponed_release'] == current_time:
                        # Assuming 'next_job' is the job you want to insert after the current one
                        next_job = job_queue[i]  # Get the job at index i
                        job_queue.insert(i + 2, next_job)  # Insert 'next_job' after the current job
                        i += 1
                        print(job_queue)

                    if (current_time < job_queue[i + 1]['postponed_release'] and job_queue[i]['execution_time']== 0 ):
                        x = job_queue[i + 1]['postponed_release']- current_time
                        y_axis_primary.append("IDLE")
                        from_x_primary.append(current_time)
                        to_x_primary.append(current_time + x)
                        # job_queue[i]['execution_time'] -= 1
                        current_time += job_queue[i + 1]['postponed_release']- current_time

        if not any_job_running:
            x = hp - current_time
            y_axis_primary.append("IDLE")
            from_x_primary.append(current_time)
            to_x_primary.append(current_time + x)
            break

    return y_axis_primary, from_x_primary, to_x_primary
def drawGanttSpare(y_axis_Spare, from_x_Spare, to_x_Spare):
    """
    The scheduled results are displayed in the form of a
    gantt chart for the user to get better understanding
    """
    colors = ['red', 'green', 'blue', 'orange', 'yellow']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = plt.hlines(y_axis_Spare, from_x_Spare, to_x_Spare, linewidth=20, color=colors[2])
    plt.title('selective_approach in R_pattern (Spare Processor)')
    plt.grid(True)
    plt.xlabel("Time")
    plt.ylabel("HIGH------------------Priority--------------------->LOW")
    plt.xticks(np.arange(min(from_x_primary), max(to_x_primary) + 1, 1.0))
    plt.show()

if __name__ == '__main__':

    print("\n\n\t\t_Fixed_priority SCHEDULER in Stand_by_Sparing_\n")

    Read_data()
    # sched_res = Schedulablity()
    # if sched_res == True:

    hp = Hyperperiod()
    Simulation(hp)
    showMetrics()
    drawGantt()
    create_jobs()
    # print(calculate_promotion_times(job_list))

    print(postponed_release_time(job_list))
    Mandatory_jobs ,Optional_jobs = separate_jobs_by_type(job_list)
    print(Mandatory_jobs)
    print(Optional_jobs)
    tasks_dict = {item['task']: [sub for sub in Mandatory_jobs if sub['task'] == item['task']] for item in Mandatory_jobs}
    print(tasks_dict)
    print(tasks)

    # Simulation_primary(hp)
    y_axis_primary, from_x_primary, to_x_primary = selective_approach_primary(Mandatory_jobs)
    print(selective_approach_primary(Mandatory_jobs))
    drawGanttPrimary(y_axis_primary, from_x_primary, to_x_primary)
    # selective_approcach is spare
    y_axis_Spare, from_x_Spare, to_x_Spare = selective_approach_Spare(Mandatory_jobs)
    print(selective_approach_Spare(Mandatory_jobs))
    drawGanttSpare(y_axis_Spare, from_x_Spare, to_x_Spare)

    # else:
    #
    #     Read_data()
    #     sched_res = Schedulablity()
