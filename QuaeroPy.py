# -*- coding: utf-8 -*-
"""
Created on May 17 2017

@author: jan
"""

from ctypes import *
from scipy.cluster.vq import kmeans2, whiten
import numpy as np

class QuaeroSys(object):
    
    # set max value of stimulator (i.e., 12-bit: 0 - 4095)
    MAXVAL = 4095
    # best temporal resolution (2000 Hz, i.e., every 0.5 ms)
    BEST_DT = 0.5
    
    # initialize the stimulator
    def __init__(self, lib_loc, lic, stim_card_slots = [0, 1], dt = 0.5, internal_trg = 5, wait_for_trigger = False, local_buffer_size = 100000):
        
        # store the slots in which the stim card(s) are plugged in (e.g., two cards in 0 and 1)
        self.stim_card_slots = stim_card_slots
        # initialize Stimulatot with not to wait for external trigger
        self.wait_for_trigger = wait_for_trigger
        # set temporal resolution in ms, i.e., sampling rate = 1000 Hz (dt = 1 ms) or 2000 Hz (dt = 0.5 ms)
        self.dt = dt
        # set internal trigger value
        self.internal_trg = internal_trg
        
        # load the library (stimlib0.dll or stimlib0.so)
        self.lib = cdll.LoadLibrary(lib_loc)
        
        # set correct data formats of parameters and return values
        self.lib.initStimulator.argtypes = [c_char_p]
        self.lib.initStimulator.restype  = c_int
        
        self.lib.closeStimulator.argtypes = []
        self.lib.closeStimulator.restype = c_int
        
        self.lib.resetStimulator.argtypes = []
        self.lib.resetStimulator.restype = c_int
        
        self.lib.startStimulation.argtypes = []
        self.lib.startStimulation.restype = c_int
        
        self.lib.stopStimulation.argtypes = []
        self.lib.stopStimulation.restype = c_int
        
        self.lib.setProperty.argtypes = [c_char_p, c_uint]
        self.lib.setProperty.restype = c_int
        
        self.lib.getProperty.argtypes = [c_char_p]
        self.lib.getProperty.restype = c_int
        
        self.lib.wait.argtypes = [c_ubyte, c_uint]
        self.lib.wait.restype = c_int
        
        self.lib.waitForTrigger.argtypes = [c_ubyte, c_ubyte]
        self.lib.waitForTrigger.restype = c_int
        
        self.lib.setPinBlock.argtypes = [c_ubyte, c_ubyte, c_ubyte, c_ubyte, c_ubyte, c_ubyte, c_ubyte, c_ubyte, c_ubyte, c_ubyte]
        self.lib.setPinBlock.restype = c_int
        
        self.lib.setDAC.argtypes = [c_ubyte, c_ushort]
        self.lib.setDAC.restype = c_int
        
        # set the size of the local buffer
        self.lib.setProperty(c_char_p('local_buffer_size'.encode('ascii')), c_uint(local_buffer_size))
        # initilize the stimulator with the provided license
        self.init_status = self.lib.initStimulator(c_char_p(lic.encode('ascii')))
        
        # check the initialization status 
        if self.init_status == 0:
            print('successfully loaded!')
        elif self.init_status == 2000:
            print('wrong license!')
        else:
            print('Failed with error code:', self.init_status)
    
    # set the temporal resolution of the stimulator
    def setTemporalResolution(self, dt):
        self.dt = dt
    
    # get the current temporal resolution
    def getTemporalResolution(self):
        return self.dt
    
    # get the current mode: waiting for trigger or not
    def getWaitForTriggerMode(self):
        return self.wait_for_trigger
        
    # close the stimulator -> always call this at the end!
    def close(self):
        print('closing ...')
        
        close_status = self.lib.closeStimulator()
      
        if close_status == 0:
            print('... done!')
        else:
            print('... failed!!!!')
            
        return close_status
    

    # reset the stimulator
    def reset(self):
        print('resetting ...')
        
        reset_status = self.lib.resetStimulator()
        
        if reset_status == 0:
            print('... done!')
        else:
            print('... failed!!!!')    
            
        return reset_status
    
    
    # get a property of the QuaeroSys stimulator (low-level function)
    def getProperty(self, property_name):
        return self.lib.getProperty(c_char_p(property_name.encode('ascii')))
    
    # get the status of the local buffer
    def getLocalBuffer(self):
        return self.lib.getProperty(c_char_p('local_buffer_status'.encode('ascii')))
    
    # get the status of the remote buffer (i.e., 'inside' the QuaeroSys)
    def getRemoteBuffer(self):
        return self.lib.getProperty(c_char_p('remote_buffer_status'.encode('ascii')))
        
    # start a stimulation and return status of stimulator (0 - OK, everything else - check header of lib file)
    def start(self):
        return self.lib.startStimulation()

    # stop a stimulation and return status of stimulator (0 - OK, everything else - check header of lib file)
    def stop(self):
        return self.lib.stopStimulation()
    
    # sends an INTERNAL trigger with value trg for cnt times. 
    # Other calls waiting for the trigger with value trg are executed as soon as wait sends trigger
    def wait(self, trg, cnt):
        return self.lib.wait(c_ubyte(trg), c_uint(cnt))
    
    # tell the stimulator to wait for EXTERNAL trigger (e.g., provided by experiment script via LPT port)
    # if only one controller card avialble controller_card_slot = 16 and port = 0
    def waitForTrigger(self, controller_card_slot=16, port=0):
        self.wait_for_trigger = True
        return self.lib.waitForTrigger(c_ubyte(controller_card_slot), c_ubyte(port))

    # assign each pin to a DAC and return status of stimulator (0 - OK, everything else - check header of lib file)
    def setPinBlock(self, stim_card_slot, trg, pin0, pin1, pin2, pin3, pin4, pin5, pin6, pin7):
        return self.lib.setPinBlock(c_ubyte(stim_card_slot), c_ubyte(trg), 
                               c_ubyte(pin0), c_ubyte(pin1), c_ubyte(pin2), c_ubyte(pin3), 
                               c_ubyte(pin4), c_ubyte(pin5), c_ubyte(pin6), c_ubyte(pin7))
    
    # set a specific DAC (with index idx_DAC [0 - 7]) to the value val [0 - 4095]
    def setDAC(self, idx_DAC, val):
        return self.lib.setDAC(c_ubyte(idx_DAC), c_ushort(val))
    
    
    # download a stimulus to the stimulator with 2 stim cards controlling a 2x4 display each (i.e., in total 2 x 8 pins)
    def downloadStim2x8(stimulus, mode='cluster_whole'):
        # make sure that stimulus is an instance of the class QuaeroStimulus
        assert isinstance(stimulus, QuaeroStimulus), "%r is not a QuaeroStimulus" % stimulus
                         
        if len(self.stim_card_slots) == 2:
            # do all the stuff

            # assign the stim_card_slots
            stim_card_slot1 = self.stim_card_slots[0]
            stim_card_slot2 = self.stim_card_slots[1] 
            
            # possible values of a single pin range from 0 to 4095
            maxhub = self.MAXVAL 
            
            nsamples = stimulus.nsamples
            
            # check how many pins display a unique pattern over time, i.e. how many DACs are needed
            [pin2DAC, num_DAC] = stimulus.compareStimPerPin();
            
            # if we can express the original stimulus by less than 8 DACs, we can simply assign each pin to a certain DAC and write
            # the stimulus into the according DACs
            if num_DAC < 8:
                # the easy case....
                print('<8 DAC')
                self.setPinBlock(self.stim_card_slots[0], self.internal_trg, pin2DAC(1,2), pin2DAC(1,1), pin2DAC(2,2), pin2DAC(2,1), pin2DAC(3,2), pin2DAC(3,1), pin2DAC(4,2), pin2DAC(4,1))
                self.setPinBlock(self.stim_card_slots[1], self.internal_trg, pin2DAC(1,4), pin2DAC(1,3), pin2DAC(2,4), pin2DAC(2,3), pin2DAC(3,4), pin2DAC(3,3), pin2DAC(4,4), pin2DAC(4,3))
                self.wait(self.internal_trg, self.dt/self.BEST_DT)
                
                for n in range(stimulus.nsamples):
                    for DAC_idx in range(num_DAC):
                        (row, col) = np.argwhere(pin2DAC == DAC_idx)
                        # [row,col] = find(pin2DAC==DAC_idx,1,'first'); # MATLAB  version
                        self.setDAC(DAC_idx, round(stimulus.stim_mat[row[0],col[0],n]*self.MAXVAL))
                        print(stimulus.stim_mat[:,:,n])
                        print('Set DAC ', DAC_idx, ' to: ', round(stimulus.stim_mat[row[0],col[0],n]*self.MAXVAL))
                        
                    self.wait(self.internal_trg, self.dt/self.BEST_DT)

                
            else:
                # if we need more than 8 DACs we have to discretize the stimulus into 8 classes    
                if (mode == 'discretize') or (mode == 'cluster_whole'):
                    print('>8 DAC')
                    # discretize the stimuli ...
                    if mode == 'cluster_whole':
                        discrete_stim_mat, center_vals = stimulus.clusterPinhub()
                    elif mode == 'discretize':
                        discrete_stim_mat, center_vals = stimulus.discretizePinhub()
                    
                    center_vals = np.sort(center_vals)
                    
                    for idx, val in enumerate(center_vals):                        
                        print('Set DAC ', idx, ' to: ', round(val*self.MAXVAL))
                        self.setDAC(idx, round(val*self.MAXVAL))
                        #calllib('stimlib0', 'setDAC', v_idx-1, round(center_vals(v_idx)*maxhub)); # MATLAB code
                    
                    for n in range(stimulus.nsamples):
                        pin_vals = np.squeeze(discrete_stim_mat[:,:,n]);
                        tmp_vals = np.unique(pin_vals).reshape(-1,1);

                        pin2DAC = np.zeros(pin_vals.shape)-1;
                        for tmp_val in tmp_vals:
                            pin2DAC[pin_vals==tmp_val] = np.argwhere(tmp_val==center_vals);
                        
                        print(pin2DAC)
                        
                        self.setPinBlock(self.stim_card_slots[0], self.internal_trg, pin2DAC(1,2), pin2DAC(1,1), pin2DAC(2,2), pin2DAC(2,1), pin2DAC(3,2), pin2DAC(3,1), pin2DAC(4,2), pin2DAC(4,1))
                        self.setPinBlock(self.stim_card_slots[1], self.internal_trg, pin2DAC(1,4), pin2DAC(1,3), pin2DAC(2,4), pin2DAC(2,3), pin2DAC(3,4), pin2DAC(3,3), pin2DAC(4,4), pin2DAC(4,3))
                        self.wait(self.internal_trg, self.dt/self.BEST_DT)
                        # MATLAB code
                        #quaero_status1 = calllib('stimlib0', 'setPinBlock', stim_card_slot1, 5, pin2DAC(1,2), pin2DAC(1,1), pin2DAC(2,2), pin2DAC(2,1), pin2DAC(3,2), pin2DAC(3,1), pin2DAC(4,2), pin2DAC(4,1));
                        #quaero_status2 = calllib('stimlib0', 'setPinBlock', stim_card_slot2, 5, pin2DAC(1,4), pin2DAC(1,3), pin2DAC(2,4), pin2DAC(2,3), pin2DAC(3,4), pin2DAC(3,3), pin2DAC(4,4), pin2DAC(4,3));
                        #calllib('stimlib0', 'wait', 5, dt/0.5);
                    
                    
                elif mode == 'cluster_sample':
                    
                    # ... or sample
                    for n in range(stimulus.nsamples):
                        # get the values of all pins at the given sample...
                        tmp = np.squeeze(stim_mat[:,:,n]).reshape(-1,1)
                        # .. and cluster them into max 8 groups
                        centroids, labels = kmeans2(tmp, 8)
                        # set the new pin values that correspond to the cluster means
                        pin_vals = centroids[labels].reshape(4,4)
                        # get a list of all unique sorted values
                        tmp_vals = np.unique(centroids)
                        # set the 8 DACs according to the 8 values and store the mapping in a pin2DAC matrix
                        pin2DAC = zeros(4,4)-1;
                        for idx, val in enumerate(tmp_vals):                        
                            print('Set DAC ', idx, ' to: ', round(val*self.MAXVAL))
                            self.setDAC(idx, round(val*self.MAXVAL))
                            # calllib('stimlib0', 'setDAC', v_idx-1, round(tmp_vals(v_idx)*maxhub)); # MATLAB
                            pin2DAC[pin_vals==val] = idx;
                        
                        print(pin2DAC)
                        
                        self.setPinBlock(self.stim_card_slots[0], self.internal_trg, pin2DAC(1,2), pin2DAC(1,1), pin2DAC(2,2), pin2DAC(2,1), pin2DAC(3,2), pin2DAC(3,1), pin2DAC(4,2), pin2DAC(4,1))
                        self.setPinBlock(self.stim_card_slots[1], self.internal_trg, pin2DAC(1,4), pin2DAC(1,3), pin2DAC(2,4), pin2DAC(2,3), pin2DAC(3,4), pin2DAC(3,3), pin2DAC(4,4), pin2DAC(4,3))
                        self.wait(self.internal_trg, self.dt/self.BEST_DT)
                        # calllib('stimlib0', 'setPinBlock', stim_card_slot1, 5, pin2DAC(1,2), pin2DAC(1,1), pin2DAC(2,2), pin2DAC(2,1), pin2DAC(3,2), pin2DAC(3,1), pin2DAC(4,2), pin2DAC(4,1));
                        # calllib('stimlib0', 'setPinBlock', stim_card_slot2, 5, pin2DAC(1,4), pin2DAC(1,3), pin2DAC(2,4), pin2DAC(2,3), pin2DAC(3,4), pin2DAC(3,3), pin2DAC(4,4), pin2DAC(4,3));
                        # calllib('stimlib0', 'wait', 5, dt/0.5);
                        
                else:
                    print('No such method implemented: ', mode)
                

        else:
            # do nothing
            print('not 2 stim cards available')
            return
        

class QuaeroStimulus(object):

    # initialize a stimulus object that provides all necessary information for the stimulator to download the stimulus
    def __init__(self, stim_mat):
        assert type(stim_mat) == np.ndarray
        # store the actual stimulus matrix (N x M x nsamples) with N and M referring to the number of pins of the Braille display
        self.stim_mat = stim_mat
        # get the number of samples
        self.nsamples = stim_mat.shape[2]
        
    # compare the stimulus traces of all pins to find out how many DACs we need, 
    # i.e., check if there are pins that display the same signal        
    def compareStimPerPin(self, tol=0.001):
        print('*** comparePerPin() ***')
        xpins = self.stim_mat.shape[0]
        ypins = self.stim_mat.shape[1]
        
        # create empty pin-to-DAC mapping with -1
        pin2DAC = np.zeros((xpins,ypins))-1
        num_DAC = 0
        
        for x in range(xpins):
            for y in range(ypins):
                # if the signal was already assigned to a DAC, continue with next unassigned one
                if not(pin2DAC[x,y] == -1):
                    continue
                
                # Debugging output
                print('pin: ', x, y)
                    
                # broadcast the signal from the selected single pin to the whole display
                # to allow for an easy comparison with all the other pins
                pin_signal = np.tile(self.stim_mat[x,y,:], (xpins,ypins,1))
            
                # compute root-mean-square-error between all pins and the selected pin
                rmse = np.sqrt(np.mean((self.stim_mat - pin_signal)**2,2))
            
                # all pins with a small rmse carry the same signal...
                same_sig_idx = rmse<tol;
            
                # .. assign same DAC to those pins
                pin2DAC[same_sig_idx] = num_DAC;
                num_DAC += 1;

        
        return pin2DAC, num_DAC
    
    # cluster (k-means) the values of the stimulus traces into 8 clusters max to allow a handling with 8 DACs
    def clusterPinhub(self):
        print('*** clusterPinhub() ***')
        tmp_vals = self.stim_mat.reshape(-1,1)
        
        # do the kmeans clustering with 8 clusters
        centroids, labels = kmeans2(tmp_vals, 8)
        
        clustered_stim_mat = centroids[labels];
        clustered_stim_mat = clustered_stim_mat.reshape(self.stim_mat.shape);
        
        center_vals = np.unique(centroids)
        
        return clustered_stim_mat, center_vals
    
    # discretize the values of the stimulus traces into 8 classes (with finer resolution for 'upper' classes)
    def discretizePinhub(self):
        print('*** discretizePinhub() ***')

        center_vals = np.zeros((8,1));
        discrete_stim_mat = np.zeros(self.stim_mat.shape);
        
        # class_boundaries with finer resolution for higher values, as the lower values are not reliably
        # detectable anyways
        class_boundaries = [(0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1)]
        
        for i, (lower_bound, upper_bound) in enumerate(class_boundaries):
            center_vals[i] = np.median(self.stim_mat[(self.stim_mat >= lower_bound) & (self.stim_mat <= upper_bound)])
            discrete_stim_mat[ (self.stim_mat >= lower_bound) & (self.stim_mat <= upper_bound) ] = center_vals[i]      
        
        center_vals[np.isnan(center_vals)] = 0
        
        return discrete_stim_mat, center_vals
        
    # compare two different stimulus matrices (s1 and s2) with each other, e.g., to check whether the clustering went well
    def compareStimMat(self, comp_stim):
        import matplotlib.pyplot as plt
        
        s1 = self.stim_mat
        s2 = comp_stim
        
        if not(s1.shape == s2.shape):
            print('inputs must have same dimensions')
            return

        f, axarr = plt.subplots(s1.shape[0], s1.shape[1], sharex=True, sharey=True)
        
        for i in range(s1.shape[0]):
            for j in range(s1.shape[1]):
                axarr[i, j].plot(np.squeeze(s1[i,j,:]), 'b')
                axarr[i, j].plot(np.squeeze(s2[i,j,:]), 'r--')
        
        plt.show()
    