import numpy as np
import matplotlib.pyplot as plt

def find_open_pores(arr, open_pore_mean, negligible_gap, verbose = False):
    high_points = [i for i in range(len(arr)) if arr[i] > open_pore_mean]
    regions = []
    
    # make regions
    if len(high_points) > 1:
        last_s = high_points[0]
        last_e = high_points[0]
        for pt in high_points[1:]:
            if pt == last_e + 1:
                last_e = pt
            else:
                regions.append([last_s, last_e])
                last_s = pt
                last_e = pt
        regions.append([last_s, last_e])
    if verbose > 1:
        plt.plot(arr)
        plt.scatter([regions[i][0] for i in range(len(regions))], [arr[regions[i][0]] for i in range(len(regions))], color = 'green')
        plt.scatter([regions[i][1] for i in range(len(regions))], [arr[regions[i][1]] for i in range(len(regions))], color = 'orange')
        plt.title("regions")
        plt.show()
        print("pre-merge", regions)
        print([regions[i][0] for i in range(len(regions))], [regions[i][1] for i in range(len(regions))])
    # merge regions
    merged_regions = [regions[0]] if len(regions) > 0 else []
    for i, (lo, hi) in enumerate(regions):
        # skip the first high region; that was already added to the list
        if i == 0: 
            continue
        # if the low value is only a little bit bigger of the last high value, extend the last range
        if lo - negligible_gap <= merged_regions[-1][1]:
            merged_regions[-1][1] = hi
        else:
            merged_regions.append([lo, hi])
    
    # delete any regions that are only 1 point long after merging
    regions = [mr for mr in merged_regions if mr[1] > mr[0] + 3]
    
    if verbose:
        plt.plot(arr, alpha=.2)
        plt.scatter([regions[i][0] for i in range(len(regions))], [arr[regions[i][0]] for i in range(len(regions))], color = 'green')
        plt.scatter([regions[i][1] for i in range(len(regions))], [arr[regions[i][1]] for i in range(len(regions))], color = 'orange')
        plt.title("merged regions")
        print("merged regions", regions)
        plt.show()
    return regions

def find_translocation(raw_data, sampling_rate, open_pore_mean, open_pore_cutoff=85, verbose=False):
#     start_time = time.time()
    negligible_gap = int(.45*sampling_rate)
    min_translocation_length = int(2.5*sampling_rate)
    max_translocation_length = 20*sampling_rate
    running_std_window = 200
    open_pore_cutoff = 85 ##
    
    median = np.median(raw_data[::100]) # ::100 speeds this up and doesnt change end result by much
    if median > 50 or median < 1:
        if verbose:
            print('disqualified signal because unexpected median of', median)
        return []
    
    leadup_range = int(.2*sampling_rate)# right before open pore
    
    # find open pores
    # note that the if median open pore value is closer to 150,  
    # open_pore=90 makes it possible to find open pores with lots of noise
    # pre_hr_time = time.time()
    high_regions = find_open_pores(raw_data, open_pore_mean/2, negligible_gap, verbose = verbose)
    # filter inter-open pore regions
    # post_hr_time = time.time()
    candidate_loci = []  # arrays of [peptide_start, peptide_end (i.e. open_pore_start), open_pore_end]
    last_end = 0
    for (start_hi, end_hi) in high_regions:
        start_scan = int(max(start_hi - max_translocation_length, last_end))
        
        # signal needs to have a reasonable length 
        if start_hi-start_scan < min_translocation_length:
            last_end = end_hi
            continue
            
        # there shouldn't be significant neg values in the pore (prob a current flip)
        if np.mean(raw_data[start_scan:start_hi]) < 0 or (raw_data[start_scan:start_hi] < -50).sum() > 10:
            last_end = end_hi
            continue
            
        # move start_hi to be a bit earlier there is an an open pore value
        arr_end_i = min(start_hi, len(raw_data)-10)
        arr_start_i = arr_end_i - int(leadup_range/4)
        old_start_hi = start_hi
        diff_means_arr = np.diff([np.mean(raw_data[i:i+10]) for i in range(arr_start_i,arr_end_i)])
        for i, val in enumerate(diff_means_arr):
            if val >=3:
                start_hi = arr_start_i+i
                if verbose:
                    print('adjusted lo from', old_start_hi, 'to', start_hi)
                break
            
        candidate_loci.append([start_scan, start_hi, end_hi])
        last_end = end_hi
#     post_cand_time = time.time()
    if verbose:
        print("candidate_loci", candidate_loci)
    
    # go through peptide_candidates, and look for ones with filled_pore_regions
    peptide_loci = []   
    for (lo, hi, end) in candidate_loci:
        
        half = int((lo+hi)/2)
        std = np.std(raw_data[lo:half])
        lower_mean = np.mean(raw_data[lo:half])
        leadup_mean = np.mean(raw_data[max(half, hi-leadup_range) + 1:hi])
        open_pore = np.percentile(raw_data[hi:end], 90) if end != hi else raw_data[hi]
        std_wnd = int((sampling_rate)/500)
        
        if np.mean(raw_data[lo:hi]) < open_pore/15 or np.mean(raw_data[lo:hi]) > open_pore/2:
            if verbose:
                print("odd mean of ", np.mean(raw_data[lo:hi]), " in range", (lo, hi), 'relative to ', open_pore/2,  open_pore/15)
            continue
        
        if np.std(raw_data[lo:hi]) > 30 or np.std(raw_data[lo:hi]) < 1.5:
            if verbose:
                print("odd std of ", np.std(raw_data[lo:hi]), " in range", (lo, hi))
            continue
    
        if leadup_mean-lower_mean < .05*(open_pore-lower_mean):
            half = int((lo+hi)/2)
            lower_mean = np.mean(raw_data[lo:half])
            if verbose:
                print(lo, half, hi, lower_mean, leadup_mean)
                print("oddly low leadup mean of " + str(leadup_mean-lower_mean) + "in range" + str((lo,half)))
            continue
        
            
        if lower_mean + 0.5*std > leadup_mean:
            if verbose:
                print("oddly the leadup isn't higher ", leadup_mean, " in range", (lo, hi), 'relative to', lower_mean, std)
            continue
        
        std_of_means = np.std([np.mean(raw_data[i:min(i+std_wnd, hi)]) for i in range(half, hi, std_wnd)])
        if std_of_means < 2.2:
            if verbose:
                print("oddly low std of means in ", std_of_means, " in range", (half, hi), 'relative to', 2.2)
            continue
        
        if np.median(raw_data[max(half, hi-leadup_range) + 1:hi]) < 20:
            if verbose:
                print("oddly low median in ", np.median(raw_data[max(half, hi-leadup_range) + 1:hi]), " in range", (max(half, hi-leadup_range), hi), 'relative to', 20)
            continue
            
        running_std = [np.std(raw_data[i:min(i+running_std_window, hi)]) for i in range(lo, hi, running_std_window)]
        running_std_mean = np.mean(running_std)
        if (running_std_mean > 3 and sampling_rate < 5000) or running_std_mean > 6:
            if verbose:
                print('disqualified signal of high running std mean', running_std_mean)
            return []
    
        for i, x in enumerate(raw_data[lo:hi]):
            if x >= open_pore_cutoff:
                if verbose:
                    print("back from ",hi, "to", "lo+i-10")
                hi = lo+i-10
                break
                
        peptide_loci.append([lo, hi, end])
    return peptide_loci