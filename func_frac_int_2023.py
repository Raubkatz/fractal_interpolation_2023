import numpy as np
from copy import deepcopy as dc
import nolds
import hurst
import neurokit
from sklearn.preprocessing import MinMaxScaler
import os
import random

def reset_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

def re_brace(dataset): # adds an additional brackst to the inner values
    Out_Arr = []
    for i in range(len(dataset)):
        Out_Arr.append(dataset[(i):(i+1)])
    return np.array(Out_Arr)

def un_brace(dataset): # removes the inner bracket
    Out_Arr = np.empty([len(dataset)])
    for i in range(len(dataset)):
        Out_Arr[i] = dataset[i,0]
    return Out_Arr

def roundDown(n):
    return int("{:.0f}".format(n))

def calc_sub_length_and_slize(orig_data, n_sub_pieces, start_sublengths=1000, scale=True, return_scaler=False):
    """
    :param orig_data: the actual data set, univariate numpy array
    :param n_sub_pieces: the number of pieces you want to split the dataset into,
    :return: sequenced data set, sequenced data set x-coordinate, scaler, if a scaler was used, if not then scaler=None
    """
    if scale:
        ret_scaler = MinMaxScaler()
        orig_data = dc(un_brace(ret_scaler.fit_transform(re_brace(orig_data))))

    orig_data_x = np.array(list(range(len(orig_data))))
    sub_list_list = list()
    sub_list_list_x = list()
    sublengths = np.array(list(range(start_sublengths))) #generate array to reverse sort the different lengths
    sublengths = dc(sublengths[::-1]) # reverse the array
    break_out = False
    again = False
    for ii in sublengths:
            if ii > len(orig_data):
                if ii%100 == 0:
                    print('.')
            else:
                if n_sub_pieces == 1:
                    print('whole piece')
                    sub_length = len(orig_data)
                    if ii == len(orig_data):
                        sub_list_list.append(orig_data)
                        sub_list_list_x.append(orig_data_x)
                        break
                if ii>1:
                    sub_estimate = int(np.floor((len(orig_data)-1) / (ii-1))) # estimate the length of sub samples
                else:
                    print('Running the whole procedure again with reduced number of sub pieces.')
                    print('I.e., the former: n_sub_pieces=' + str(n_sub_pieces) + ", is now: n_sub_pieces=" + str(n_sub_pieces -1))
                    again = True
                    break
                if n_sub_pieces == sub_estimate: # take always the largest possible number
                    print(sub_estimate)
                    sub_length = ii
                    j = 0
                    for i in orig_data_x:
                        if (j + ii - 1) >= len(orig_data):
                            if j == (len(orig_data)-1):
                                break_out = True
                                break
                            else:
                                sub_list_list.append(list(orig_data[-ii:]))
                                sub_list_list_x.append(list(orig_data_x[-ii:]))
                                break_out = True
                                break
                        else:
                            sub_list_list.append(list(orig_data[j:(j+ii)]))
                            sub_list_list_x.append(list(orig_data_x[j:(j+ii)]))
                            j = j + ii - 1
            if break_out:
                break
    if again: #if the function didn't find a partition, do it again with reduced number of sub samples
        sub_list_list, sub_list_list_x, ret_scaler = dc(calc_sub_length_and_slize(orig_data, n_sub_pieces-1, start_sublengths=start_sublengths, scale=False))
    if scale:
        return sub_list_list, sub_list_list_x, ret_scaler
    else:
        #ret_scaler = 'lulu'
        return sub_list_list, sub_list_list_x, None

def calc_linear_parameters(sub_list):
    sub_list_x = np.array(list(range(len(sub_list))))
    delta_y = sub_list[-1] - sub_list[0]
    delta_x = sub_list_x[-1] - sub_list_x[0]
    a = delta_y/delta_x
    b = sub_list[0]
    return a, b

def linear_function(a,b,x):
    return a*x+b

def complexity(array, hurst_func=0, scale=False):
    if scale:
        scaler = MinMaxScaler()
        array = dc(un_brace(scaler.fit_transform(re_brace(array))))
    if hurst_func==0:
        return nolds.hurst_rs(array)
    if hurst_func==1:
        return hurst.compute_Hc(array, simplified=True)[0]
    if hurst_func==2:
        return hurst.compute_Hc(array, simplified=False)[0]
    if hurst_func==3:
        return neurokit.complexity_fd_petrosian(array)
    if hurst_func==4:
        return neurokit.complexity_entropy_svd(array)

def build_frac_int_time_series(sub_list_list_frac, sub_list_list_frac_x, scaler=None):
    print('running')
    print(np.shape(sub_list_list_frac))
    out_list_frac = list()
    out_list_frac_x = list()
    sub_list = dc(sub_list_list_frac[0])
    sub_list_x = dc(sub_list_list_frac_x[0])
    for i in range(len(sub_list)): #appending the first sequence
        out_list_frac.append(sub_list[i])
        out_list_frac_x.append(sub_list_x[i])
    for ii in range(len(sub_list_list_frac)-1):
        add = False
        ik = ii +1
        sub_list = dc(sub_list_list_frac[ik])
        sub_list_x = dc(sub_list_list_frac_x[ik])
        for iii in range(len(sub_list)):
            if (sub_list_x[iii] == out_list_frac_x[-1]): #criterion to not add one point two times
                add = True
            if add:
                out_list_frac.append(sub_list[iii]) #add all other sequences
                out_list_frac_x.append(sub_list_x[iii])
    if scaler != None:
        return dc(un_brace(scaler.inverse_transform(re_brace(out_list_frac)))), dc(out_list_frac_x)
    else:
        return dc(out_list_frac), dc(out_list_frac_x)

def random_selection_of_interpolation_points(sub_list, len_sub_intp):
    initial_y = list()
    for i in range(len_sub_intp):
        initial_y.append(sub_list[random.randrange(len(sub_list))])
    return dc(np.array(initial_y))


def fractal_interpolation(sub_list_list, sub_list_list_x, n_intp, hurst_func=0, hurst_iterations=500, wn_iterations=1, scaler=None, take_interpolation_points=False, epsilon=0.0001):
    frac_int_sub_list_list = list()
    frac_int_sub_list_list_x = list()
    for i in range(len(sub_list_list)): #iterate throught all sequences
        print("sequence nr: " + str(i))
        sub_list = dc(sub_list_list[i])
        sub_list_x = dc(sub_list_list_x[i])
        local_complexity = complexity(sub_list, hurst_func=hurst_func)
        sub_list_x_zero = np.array(list(range(len(sub_list))))
        len_sub_intp = ((len(sub_list) - 1) * n_intp) + 1
        data_int_x = np.array(list(range(len_sub_intp)))
        data_int_x = (data_int_x / (len(data_int_x) - 1)) * (len(sub_list) - 1)
        a,b = calc_linear_parameters(sub_list)
        if take_interpolation_points:
            initial_y = dc(random_selection_of_interpolation_points(sub_list, len_sub_intp))
        else:
            initial_y = dc(linear_function(a, b, data_int_x))  # initial y is a line between first and last data point of the sequence
        old_complexity_delta = dc(1000000000)
        best_sequence = dc(list())
        for iH in range(hurst_iterations):
            bool_intp = dc(np.empty(len(data_int_x), dtype=bool))
            bool_intp[:] = dc(True)
            eps_arr = dc(np.empty(len(data_int_x), dtype=float))
            eps_arr[:] = dc(1000)
            data_int_x = dc(data_int_x)
            data_int_y = dc(data_int_x)
            switch = True
            while switch:
                #Do this until we got a full array
                an = list()
                dn = list()
                cn = list()
                en = list()
                sn = random.uniform(-1, 1) #pick random vertical sclaing factor
                for iikk in range(len(sub_list) - 1):  # calculate parameters for fractal interpolation
                    an.append((sub_list_x_zero[iikk + 1] - sub_list_x_zero[iikk]) / (sub_list_x_zero[-1] - sub_list_x_zero[0]))
                    dn.append((sub_list_x_zero[-1] * sub_list_x_zero[iikk] - sub_list_x_zero[0] * sub_list_x_zero[iikk + 1]) / (sub_list_x_zero[-1] - sub_list_x_zero[0]))
                    cn.append((sub_list[iikk + 1] - sub_list[iikk]) / (sub_list_x_zero[-1] - sub_list_x_zero[0]) - sn * ((sub_list[-1] - sub_list[0]) / (sub_list_x_zero[-1] - sub_list_x_zero[0])))
                    en.append((sub_list_x_zero[-1] * sub_list[iikk] - sub_list_x_zero[0] * sub_list[iikk + 1]) / (sub_list_x_zero[-1] - sub_list_x_zero[0]) - sn * ((sub_list_x_zero[-1] * sub_list[0] - sub_list_x_zero[0] * sub_list[-1]) / (sub_list_x_zero[-1] - sub_list_x_zero[0])))
                #pick random transformation
                wn_n = dc(random.randrange(len(sub_list)-1))
                rand_element = dc(random.randrange(len(data_int_x)-1))
                x_prime = dc(data_int_x[rand_element]) # initialize x and y prime
                y_prime = dc(initial_y[rand_element])
                for iii in range(wn_iterations): #iteratively apply the fractla interpolation transformation
                    x_prime = dc(x_prime * an[wn_n] + dn[wn_n])
                    y_prime = dc(cn[wn_n] * x_prime + sn * y_prime + en[wn_n])
                    #print(abs(x_prime - data_int_x[ii]))
                    kk = 0 #count through original data points
                    for ikikik in range(len(initial_y)): # count thourhg the interpolated dat apoints
                        if data_int_x[ikikik] == int(data_int_x[ikikik]): #if original data point
                            bool_intp[ikikik] = False
                            data_int_y[ikikik] = sub_list[kk] #interpolated array needs to have original data points between the interpolated data points
                            kk = kk + 1 #count through original data points
                        else: #if its and interpolated data point, then we need to find one close to the equdistantly spaced abscissa
                            #print('chekcin the int')
                            if abs(x_prime - data_int_x[ikikik])<epsilon: #if the distance between an equidistant abscissa and a fractal inteproalted data point is very small, then use this one
                                if not any(bool_intp):  # have we got all data points? then break the loop
                                    print('full array found')
                                    switch = False
                                    break
                                if abs(x_prime - data_int_x[ikikik])<eps_arr[ikikik]: #also the distance error must be smaller than the previous one
                                    bool_intp[ikikik] = False
                                    data_int_y[ikikik] = dc(y_prime)
                                    if not any(bool_intp): # have we got all data points? then break the loop
                                        print('full array found')
                                        switch = False
                                        break
            frac_int_list = dc(list(data_int_y)) #make the intperoalted data points to a list
            complexity_delta = abs(local_complexity - complexity(frac_int_list, hurst_func=hurst_func)) #get the complexity delta
            if complexity_delta != complexity_delta: #sometimes there are some errors for small sub sets, resulting in nans, this cures the nans
                complexity_delta = 1
                print('Correcting for NaN')
            if complexity_delta<old_complexity_delta:
                print('############################################')
                print('############################################')
                print(' best fractal interpolation encountered for Hurst ' + str(iH))
                old_complexity_delta = dc(complexity_delta)
                best_sequence = dc(np.array(frac_int_list))
        best_sequence_x = dc(np.array(data_int_x) + sub_list_x[0]) #shift the x-coordinate with respect
        frac_int_sub_list_list.append(list(best_sequence))
        frac_int_sub_list_list_x.append(list(best_sequence_x))
    out_arr ,out_arr_x = build_frac_int_time_series(frac_int_sub_list_list, frac_int_sub_list_list_x, scaler=scaler)
    return dc(out_arr), dc(out_arr_x)








