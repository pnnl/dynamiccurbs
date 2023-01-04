# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 01:17:33 2022

@author: chase
"""

import win32com.client as com
import os
import sys
import time

experiment_directory = sys.argv[1]
experiment_name = sys.argv[1].split("\\")[-1]
print(experiment_name)

exps = os.listdir(experiment_directory)
print(exps)

for exp in exps:
    start_time = time.time()
    print("Currently running: ", exp)
    try:
        
        
        ## Connecting the COM Server => Open a new Vissim Window:
        Vissim = com.Dispatch("Vissim.Vissim")
    
        # Loads the VISSIM file and opens it
        # Locat path where the file is
    
        Vissim.LoadNet(experiment_directory + "\\" + exp + "\\" + "Seattle_Atomic_Network_" + experiment_name + "_" + exp + ".inpx")
        Vissim.LoadLayout(experiment_directory + "\\" + exp + "\\" + "Seattle_Atomic_Network_" + experiment_name + ".layx")
    
        #Vissim.LoadNet("C:\\Users\\chase\\Documents\\vissimpark\\VISSIM_configs\\baseline prototype\\base_config\\prototype.inpx")
        #Vissim.LoadLayout("C:\\Users\\chase\\Documents\\vissimpark\\VISSIM_configs\\baseline prototype\\base_config\\prototype.layx")
    
        # Delete all previous simulation runs first:
        for simRun in Vissim.Net.SimulationRuns:
            Vissim.Net.SimulationRuns.RemoveSimulationRun(simRun)
    
        #Vissim.Simulation.SetAttValue('RandSeed', 1) # Note: RandSeed 0 is not allowed
        #Vissim.Graphics.CurrentNetworkWindow.SetAttValue("QuickMode",1)
        Vissim.SuspendUpdateGUI()
        time.sleep(1)
        Vissim.Simulation.RunContinuous()
    
        #Vissim.ResumeUpdateGUI(False); # allow updating of the complete Vissim workspace (network editor, list, chart and signal time table windows)
        # # activate QuickMode
        
        end_time = time.time()
        run_time = end_time - start_time
        print("Completed. Running time (min): " + str(int(run_time/60.0)))
        
        with open(experiment_directory + "//experiment_logfile.txt", 'a') as d:
            d.write(exp.split("_")[-1] + "," + str(run_time) + ",\n")
    
        #Vissim.Close() this method doesn't exist but we'll see if results collide across exp directories
        Vissim = None #to be sure a new window is started
        
    except Exception as err:
        print("Exception: ", err)
        end_time = time.time()
        run_time = end_time - start_time
        
        with open(experiment_directory + "//experiment_logfile.txt", 'a') as d:
            d.write(exp.split("_")[-1] + "," + str(run_time) + "," + str(err) + "\n")