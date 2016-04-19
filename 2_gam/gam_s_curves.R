# Import libraries and help files
source("multiplot.R")
library(mgcv)
library(ggplot2)
library(reshape)

############################
# Create fictional dataset #
############################
# Add day dummies
f.add_day_dummies <- function(df) {
  list_days = c('sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday')
  for(day_number in seq(0, 6)) {
    day_name = list_days[day_number+1]
    df[paste('seasonality_',day_name, sep="")] = 0 # Create day dummy      
    df[df$day == day_number, paste('seasonality_',day_name, sep="")] = 1  # Give value to day dummy  
  }
  return(df)
}

# Generate random value from normal distribution whose mean differs from day to day
f.get_sales_on_weekday <- function(weekday, noise_factor=1) {
  # Initialize mean and sd values for days, first value is sunday
  mean_list = c(0, 4, 3, 4, 5, 6, 8)
  sd_list   = c(0, 1, 1, 1, 1, 1, 1)  
  mean      = mean_list[weekday+1]
  sd        = sd_list[weekday+1] * noise_factor
  # Random value from normal distribution, mean and sd differ on weekday
  value     = rnorm(1, mean=mean, sd=sd)
  if(value < 0) { value = 0 }
  return(value)
}

# Add fictional radio GRPs on specific date
f.add_radio_values<- function(df, radio_share=3) {
  # Add grp values on specific days
  sample_rows <- sample(seq(1, nrow(df)), nrow(df) / 3)
  df["radio_grp"] = 0
  for(row_number in sample_rows) {
    if(df[paste(row_number), "day"] != 0) {
      #df[paste(row_number), "radio_grp"] = runif(1, 0.0, 10.0)
      radio_grp_value = rnorm(1, mean=4, sd=2)
      if(radio_grp_value < 0) {
        radio_grp_value = 0
      }
      df[paste(row_number), "radio_grp"] = radio_grp_value 
    }
  }
  
  # If radio, then more sales based on the logistic function
  df["sales_radio"] <- apply(df['radio_grp'], 1, f.logistic_function)
  # Add radio sales to total sales
  df["sales_total"] <- df["sales"] + df["sales_radio"]
  return(df)
}

# S-response curve function / logistic function
f.logistic_function <- function(x) {
  L   = 10
  k   = 1.25
  x_0 = 5
  if(x > 0) {
    return(L / (1+exp(-k*(x-x_0))))
  } 
  return(0)  
}

# Create fictional dataset
f.create_fictional_dataset <- function(n_days, noise_factor, radio_share) {
  dates  = seq(c(ISOdate(2015,6,1)), by="DSTday", length.out=n_days)
  df     = data.frame(dates)
  df$day = as.POSIXlt(df$date)$wday  
  df     = f.add_day_dummies(df)
  df["sales"] = apply(df['day'], 1, function(x) f.get_sales_on_weekday(weekday=x,noise_factor=noise_factor))
  df          = f.add_radio_values(df, radio_share)
  # Slice
  dat <- df[c("sales_total", "radio_grp", 
              "seasonality_monday",   "seasonality_tuesday",
              "seasonality_wednesday", "seasonality_thursday", 
              "seasonality_friday", "seasonality_saturday" )]
  # Return
  return(dat)
}

#############
# GAM model #
#############  
f.create_model <- function(dat) {  
  # P-spline smoothers with lambda=0.5 used for radio_grp
  b1 <- mgcv::gam(sales_total ~ s(radio_grp, bs='ps', sp=0.5)
                  + seasonality_monday + seasonality_tuesday 
                  + seasonality_wednesday + seasonality_thursday 
                  + seasonality_friday + seasonality_saturday, 
                  data=dat)
  
  # Output model results and store intercept for plotting later on
  summary_model      <- summary(b1)
  model_coefficients <- summary_model$p.table
  model_intercept    <- model_coefficients["(Intercept)", 1]
  

  # Plot the smooth predictor function to obtain the radio response curve
  p    <- predict(b1, type="lpmatrix")
  beta <- coef(b1)[grepl("radio_grp", names(coef(b1)))]
  s    <- p[,grepl("radio_grp", colnames(p))] %*% beta + model_intercept
  # print(s)
  return(s)
}

##############
# PLOTS      #
##############
# Transform data for plots
n_days = 450
id = 1:n_days
data_x_full <- data.frame(id)
data_y_full <- data.frame(id)
radio_share_list  <- c(10, 3, 1)
radio_share_name_list <- c('10%', '33%', '100%')
noise_factor_list <- c(0.1, 1, 5)
for(iter_rs in 1:3) {
  for(iter_nf in noise_factor_list) {
    dat = f.create_fictional_dataset(n_days=n_days, 
                                     radio_share=radio_share_list[iter_rs],
                                     noise_factor=iter_nf)
    data_x_full[paste('Radio obs: ',radio_share_name_list[iter_rs],', St. dev.:',iter_nf)]  <- dat$radio_grp
    data_y_full[paste('Radio obs: ',radio_share_name_list[iter_rs],', St. dev.:',iter_nf)]  <- f.create_model(dat)
  }
}

data_x_full_melted <- melt(data_x_full, id=c("id"))
data_y_full_melted <- melt(data_y_full, id=c("id"))
data_melted = data.frame(data_x_full_melted, data_y_full_melted)
data_melted = data_melted[c('id', 'variable', 'value', 'value.1')]
data_melted <- rename(data_melted, c("variable"="Legend", "value"="RadioGRPs", "value.1"="AdditionalSales"))

# Define and show plot
p1   <- ggplot(data_melted, aes_string(x="RadioGRPs", y="AdditionalSales", group="Legend", colour="Legend"),size=6) 
p1   <- p1 + geom_line(size=1.25) + theme(axis.ticks = element_blank(),
                                          axis.text.x = element_text(colour="grey20",size=12,face="bold"),
                                          axis.text.y = element_text(colour="grey20",size=12,face="bold"),
                                          legend.text = element_text(colour="grey20",size=12,face="bold"),
                                          axis.title.x = element_text(colour="grey20",size=14,face="bold"),
                                          axis.title.y = element_text(colour="grey20",size=14,face="bold"),
                                          legend.title = element_text(colour="grey20",size=14,face="bold"))
p1

