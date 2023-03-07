"""
This program interpolates time series data as done in:
Sebastian Raubitzek, Thomas Neubauer,
A fractal interpolation approach to improve neural network predictions for difficult time series data,
Expert Systems with Applications,
Volume 169,
2021,
114474,
ISSN 0957-4174,
https://doi.org/10.1016/j.eswa.2020.114474.
(https://www.sciencedirect.com/science/article/pii/S0957417420311234)

The data set that's part of the folders is the monthly international airline passengers data set from the
Time Series Data Library by Hyndman et al. You can also load the car sales data set from the Time Series Data Library

All you need to do is to set the number of interpolation points and load the data set. However, one can choose different
splitting of the original data set such that one fractla interpolation is performed more or less fine-grained.
We recommend to use sub_pieces=10, as this is the default that was used throughout the employed research.

-Sebastian Raubitzek, 07.03.2023
"""

from matplotlib import pyplot as plt
from copy import deepcopy as dc
import numpy as np
import func_frac_int_2023 as ff

########################################################################################################################
# MAIN PARAMETERS ######################################################################################################
########################################################################################################################
data_name = "airline_passengers" #data name
#data_name = "car_sales" #data name
n_intp = 3 #nr of additional (new) inteprolated data points
sub_pieces = 10 # 10 is default, however, setting a large number will make the program choose the largest possible
                # number for each data set
x_label = "Months"
y_label = "International Airline Passengers"
########################################################################################################################
# NOT SO MAIN PARAMETERS ###############################################################################################
########################################################################################################################
scale=False #if the data is scaled before fraxctal interpolating it, if fractal inteprolation does not work too good for
            # for a particular data this might improve the procedure
hurst_func = 0 # one can choose different loss functions, 0 is the Hurst exponent by default, attention some loss
               # functions might not work on some data sets, 1 is another Hurst exponent, 2 is another hurst exponent
               # 3 is the fractal dimension by petrosian, 4 is SVD entropy from neurokit
hurst_iterations = 500 # default 500, use 50 for a quick run. How many runs using diffenent scaling factor for the
                       # optimla Hurst exponent
frac_interations = 100 # default 100, use 5 for a quick run, iterations of the fractal interpolation
########################################################################################################################
# RUNNING THE PROGRAM ##################################################################################################
########################################################################################################################
data = dc(np.genfromtxt("./DATA_ORIGINAL/" + str(data_name) + ".csv", delimiter=',')[:,1]) # load data
data_x = np.array(list(range(len(data)))) # generate x data
# slice data into sub pieces:
sequenced_data, sequenced_data_x, scaler_ret = ff.calc_sub_length_and_slize(data, sub_pieces, scale=scale)
#perform fractal interpolation:
frac_int, frac_int_x = ff.fractal_interpolation(sequenced_data, sequenced_data_x, n_intp, hurst_func=hurst_func, wn_iterations=frac_interations, scaler=scaler_ret, take_interpolation_points=False, hurst_iterations=hurst_iterations)
#plot the data
plt.plot(data_x, data, '--', color="blue")
plt.scatter(data_x, data, label="original data", color="black")
plt.plot(frac_int_x, frac_int, label="fractal interpolation", color="orange")
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.legend(bbox_to_anchor=(0.05, 0.95), loc='upper left', borderaxespad=0.)
plt.tight_layout()
plt.savefig('./plot_frac_int_' + data_name + str(n_intp) +'nintp_' + str(sub_pieces) + 'subpieces.png')  #save png
plt.savefig('./plot_frac_int_' + data_name + str(n_intp) +'nintp_' + str(sub_pieces) + 'subpieces.eps')  #save eps, use always eps for journals
plt.show()
