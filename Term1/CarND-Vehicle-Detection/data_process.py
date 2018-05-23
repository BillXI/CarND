import glob
import pickle
import numpy as np

def data_process(train_r=0.7, valid_r=0.2, test_r=0.1, pickle_file='data.p'):
    '''
    data_process(): function to use vehicle and non-vehicle images from a combination of the GTI vehicle image database, the KITTI vision benchmark suite, and examples extracted from the project video itself. 
        Input:
            * train_r: training ratio
            * valid_r: validation ratio
            * test_r: test ratio
            * pickle_file: filename to save pickle data
    '''
    print('Reading data...')
    vehicles0 = glob.glob('./vehicles/GTI_Far/*.png')
    vehicles1 = glob.glob('./vehicles/GTI_MiddleClose/*.png')
    vehicles2 = glob.glob('./vehicles/GTI_Left/*.png')
    vehicles3 = glob.glob('./vehicles/GTI_Right/*.png')
    vehicles4 = glob.glob('./vehicles/KITTI_extracted/*.png')
    
    nonvehicles1 = glob.glob('./non-vehicles/Extras/*.png')
    nonvehicles1 += glob.glob('./non-vehicles/GTI/*.png')

    print('Processing data...')
    l0,l1,l2,l3,l4,l5 = len(vehicles0),len(vehicles1),len(vehicles2),len(vehicles3),len(vehicles4),len(nonvehicles1)
    # Finding end point of trainning and validation data
    train_end = (train_r*np.array([l0,l1,l2,l3,l4,l5])).astype('int')
    
    valid_end = ((train_r + valid_r)*np.array([l0,l1,l2,l3,l4,l5])).astype('int')

    vehicles_train = vehicles0[:train_end[0]] + vehicles1[:train_end[1]] + vehicles2[:train_end[2]] + vehicles3[:train_end[3]] + vehicles4[:train_end[4]]
    nonvehicles_train = nonvehicles1[:train_end[5]]

    vehicles_val = vehicles0[train_end[0]:valid_end[0]] + vehicles1[train_end[1]:valid_end[1]] + vehicles2[train_end[2]:valid_end[2]] + vehicles3[train_end[3]:valid_end[3]] + vehicles4[train_end[4]:valid_end[4]]
    nonvehicles_val = nonvehicles1[train_end[5]:valid_end[5]]

    vehicles_test = vehicles0[valid_end[0]:] + vehicles1[valid_end[1]:] + vehicles2[valid_end[2]:] + vehicles3[valid_end[3]:] + vehicles4[valid_end[4]:]
    nonvehicles_test = nonvehicles1[valid_end[5]:]
    
    print("Number of all samples in vehicles: ", sum([l0,l1,l2,l3,l4]))
    print("Number of all samples in non-vehicles: ", l5)

    print('Number of samples in vehicles training set: ', len(vehicles_train))
    print('Number of samples in nonvehicles training set: ', len(nonvehicles_train))

    print('Number of samples in vehicles validation set: ', len(vehicles_val))
    print('Number of samples in nonvehicles validation set: ', len(nonvehicles_val))

    print('Number of samples in vehicles test set: ',len(vehicles_test))
    print('Number of samples in nonvehicles test set: ',len(nonvehicles_test))

    
    print('Saving data to pickle file...')
    

    try:
        with open(pickle_file, "wb") as pfile:
            pickle.dump(
                {
                    'vehicles_train': vehicles_train,
                    'nonvehicles_train': nonvehicles_train,
                    'vehicles_val': vehicles_val,
                    'nonvehicles_val': nonvehicles_val,
                    'vehicles_test': vehicles_test,
                    'nonvehicles_test': nonvehicles_test
                },
                pfile, 
                pickle.HIGHEST_PROTOCOL
            )
    except Exception as e:
        print("Unable to save data ", pickle_file, ": ", e)
        raise
    print('Data cached in picle file...')
    print('Finished')

