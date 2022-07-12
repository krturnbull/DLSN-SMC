## process dataset available at http://www.sociopatterns.org/datasets/primary-school-temporal-network-data/

library(dplyr)

######################### functions to get weighted networks ###########################

get_time_reduced_data <- function(data_full, tl, tu){
    return( dat_tmp = data_full %>% filter(time >= tl & time < tu ) )
}

get_yij <- function(dat_tmp, student_ids, i1, i2){
    ## dat_tmp is dataset on current time stamp
    ## i and j are indices of interest
    return( dim( dat_tmp %>% filter(i %in% student_ids[c(i1, i2)]) %>% filter( j %in% student_ids[c(i1, i2)] ) )[1] )
}

get_y_obs <- function(timevec, student_ids, data, mins=5){
    ## function to get aggregate interaction matrices
    ## timevec denotes the time steps within the day of interest
    ## data is full dataset (only classrooms of interest, over all time steps)

    ## function does the following:
    ## 1) loop over aggregate time periods
    ## 2) for each time period, loop over ij pairs to find yijt
    
    ## number of students
    N = length(student_ids)
    
    ## get aggregate time stamps
    len_int = 20*3*mins
    t_init = min(timevec)
    t_final = max(timevec)
    time_agg = seq( t_init, to = t_final, by=len_int)

    ## initialise observation matrix
    y_obs = array(0, dim=c(length(time_agg)-1, N, N) )

    for (t in 1:(length(time_agg)-1)){
        dat_tmp = get_time_reduced_data(data, time_agg[t], time_agg[t+1])
        for (i1 in 1:(N-1)){
            for (i2 in (i1+1):N){
                y_obs[t,i1,i2] = get_yij(dat_tmp, student_ids, i1, i2)
                y_obs[t,i2,i1] = y_obs[t,i1,i2]
            }
        }
        print( paste("Finished ", t, " of ", length(time_agg)-1) )
    }
    return( y_obs )
}

######################### process dataset ##############################

## read in data
data = read.table("primaryschool.csv", header = FALSE)
names(data) = c("time", "i", "j", "class_of_i", "class_of_j")

## get unique time stamps
unique_t = sort( unique( data$time ) )
len_unique_t = length(unique_t)

## find size of each class
data$class_of_i = as.factor( data$class_of_i )
data$class_of_j = as.factor( data$class_of_j )

## there are 9 classes
classnms = levels( data$class_of_i )
class_size = rep(0, 9)
for (ind in 1:9){
    class_size[ind] = length(unique( c(data$i[data$class_of_i == classnms[ind]],  data$j[data$class_of_j == classnms[ind]]) ))
}

## pick two classes:
class_include = classnms[c(2)] 

## reduce the dataframe to just these classes
data_red = as.data.frame(data)
data_red = data_red %>% filter(class_of_i %in% class_include) %>% filter(class_of_j %in% class_include)

## make sure sorted according to time
idsrt = sort( data_red$time, decreasing=FALSE, index.return=TRUE)$ix
sum( idsrt - sort(idsrt) ) ##yep, this is 0

## get student ids
student_ids = unique( c(data_red$i, data_red$j ) )

## find the time stamps at the start and end of each day
times_diff = unique_t[ 2:len_unique_t ] - unique_t[ 1:(len_unique_t - 1) ]
diffday = which(times_diff != 20 )

## there are 2 days with times
d1_times = unique_t[1:diffday[1]] 
d2_times = unique_t[(diffday[1]+1):len_unique_t]

## get the weighted networks
y_obs_d1 = get_y_obs(d1_times, student_ids, data_red, mins=4)
y_obs_d2 = get_y_obs(d2_times, student_ids, data_red, mins=4)

## save the output
dataset = list( studentids = student_ids, y_d1 = y_obs_d1, y_d2 = y_obs_d2 )
save(dataset, file="primschool_network_class1.RData", version=2)
