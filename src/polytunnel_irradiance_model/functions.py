import numpy as np

def compute_region_means(list_of_frames):
    """Calculates the average, standard deviation, and maximum of three regions in each frame of the list.

    Divides each frame into three regions: left, central, and right. The central region will always
    be equal or larger than the left and right regions, especially when the size is not divisible by 3.
    
    Args:
        list_of_frames (list of np.ndarray): List of square frames of size N x N.
        
    Returns:
        tuple: A tuple containing three lists:
            - (avg_left, avg_center, avg_right): Averages for each region.
            - (std_left, std_center, std_right): Standard deviations for each region.
            - (max_left, max_center, max_right): Maximum values for each region.
    """
    avgl_list = []
    avgc_list = []
    avgr_list = []
    avga_list = []


    stdl_list = []
    stdc_list = []
    stdr_list = []
    stda_list = []

    maxl_list = []
    maxc_list = []
    maxr_list = []
    maxa_list = []
    
    for frame in list_of_frames:
        # Ensure the frame is a square matrix
        # frame = np.array(frame)
        

        n = frame.shape[0]
        third = n // 3  # Size of the "left" and "right" regions (basic division)

        # Left region (first third of the columns)
        l_region = frame[:, :third] 

        # Right region (last third of the columns)
        r_region = frame[:, -third:] 

        # Central region (all remaining columns)
        c_region = frame[:, third:-third]  # Central region

        # Append the computed values for each region
        try:
            avgl_list.append( np.mean(l_region) )
            avgc_list.append( np.mean(c_region) )
            avgr_list.append( np.mean(r_region) )
            avga_list.append( np.mean(frame.flatten()) )
    
            stdl_list.append(  np.std(l_region) )
            stdc_list.append(  np.std(c_region) )
            stdr_list.append(  np.std(r_region) )
            stda_list.append(  np.std(frame.flatten()) )
    
            maxl_list.append(  np.max(l_region) )
            maxc_list.append(  np.max(c_region) )
            maxr_list.append(  np.max(r_region) )
            maxa_list.append(  np.max(frame.flatten()) )

        except:
            avgl_list.append( 0 )
            avgc_list.append( 0 )
            avgr_list.append( 0 )
            avga_list.append( 0 )
            stdl_list.append( 0 )
            stdc_list.append( 0 )
            stdr_list.append( 0 )
            stda_list.append( 0 )
            maxl_list.append( 0 )
            maxc_list.append( 0 )
            maxr_list.append( 0 )
            maxa_list.append( 0 )

        
    return (avgl_list, avgc_list, avgr_list, avga_list), (stdl_list, stdc_list, stdr_list, stda_list), (maxl_list, maxc_list, maxr_list, maxa_list )

def dictionary_update(data_dict, key, data_array):
    """
    Updates a dictionary with the memory size (in bytes) of a given array.

    Args:
        data_dict (dict): The dictionary to be updated.
        key (str): The key under which the memory size will be stored.
        data_array (list or array-like): The array whose memory size will be calculated.

    Returns:
        data_dict (dict): The updated dictionary
    """
    data_dict.update({key: np.array(data_array).nbytes})
    return data_dict

