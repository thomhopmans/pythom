import matplotlib.dates as dates
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
import statsmodels.tools.tools as sm_tools


class ExampleSCurves:
    """
    A practical example that shows how to optimize your media spend using S-curves
    """

    def __init__(self):
        self.df = None

    def run(self):
        """
        Execute
        :return:
        """
        # self.create_fictional_dataset()
        self.load_fictional_dataset_from_article()
        self.create_example_s_curve_plot()
        self.run_example_1()
        self.run_example_2()
        self.create_predicted_s_curve_plot()
        self.run_example_3()

    ####################################
    # CREATE/LOAD FICTIONAL DATASET    #
    ####################################
    def create_fictional_dataset(self):
        # Create fictional dataset
        self.df = pd.DataFrame(0, columns=['Sales'], index=pd.date_range('20150601',periods=31, freq='D'))
        self.df.index.name = 'datetime_NL'
        self.df = self.df.reset_index()
        self.df['day_of_the_week'] = self.df['datetime_NL'].apply(lambda x: x.strftime('%A'))
        self.df['sales'] = self.df['day_of_the_week'].apply(lambda x: self.get_random_value_based_on_weekday(x))
        self.df = self.df.set_index('datetime_NL')
        # Add fictional radio GRPs to dataset
        self.add_radio_values()
        # Add day of week dummies
        self.add_day_dummies()
        # Uncomment the line below if you want to save your fictional dataset
        self.df.to_csv('dataset.csv')

    def load_fictional_dataset_from_article(self):
        # Load earlier created fictional dataset
        self.df = pd.read_csv(filepath_or_buffer='dataset.csv',
                              index_col=['datetime_NL'],
                              parse_dates=[0])

    @staticmethod
    def get_random_value_based_on_weekday(weekday):
        """
        Assign fictional number of sales to every date based on the day of the week.
        """
        # Select the mean of the sales based on the weekday. Wednesday is the best day in this example
        weekday_mean_dict = {
            "Monday": 5,
            "Tuesday": 7,
            "Wednesday": 8,
            "Thursday": 4,
            "Friday": 4,
            "Saturday": 3,
            "Sunday": 2
        }
        mean = weekday_mean_dict[weekday]
        # Random value from normal distribution, mean differs based on weekday
        value = np.random.normal(loc=mean, scale=0.5)
        return value

    def add_radio_values(self):
        """
        Add fictional radio GRPs on specific dates
        :param df:
        :return:
        """
        self.df['radio_grp'] = int(0)
        self.df.ix['2015-06-01', 'radio_grp'] = 4
        self.df.ix['2015-06-03', 'radio_grp'] = 1
        self.df.ix['2015-06-04', 'radio_grp'] = 2
        self.df.ix['2015-06-05', 'radio_grp'] = 3
        self.df.ix['2015-06-09', 'radio_grp'] = 7.2
        self.df.ix['2015-06-11', 'radio_grp'] = 6
        self.df.ix['2015-06-13', 'radio_grp'] = 8
        self.df.ix['2015-06-17', 'radio_grp'] = 2.25
        self.df.ix['2015-06-18', 'radio_grp'] = 4.6
        self.df.ix['2015-06-20', 'radio_grp'] = 1.5
        self.df.ix['2015-06-23', 'radio_grp'] = 9
        self.df.ix['2015-06-26', 'radio_grp'] = 6.5
        self.df.ix['2015-06-30', 'radio_grp'] = 2
        # If radio, then more sales based on the logistic function
        self.df['sales'] += self.logistic_function(self.df['radio_grp'])

    def add_day_dummies(self):
        """
        Add day dummies
        :return:
        """
        list_days = ['monday', 'tuesday', 'thursday', 'friday', 'saturday', 'sunday']
        for day in list_days:
            # Create day dummy
            self.df['seasonality_' + day] = int(0)
            # Give value to day dummy
            self.df.ix[self.df['day_of_the_week'] == str.capitalize(day), ['seasonality_' + day]] = int(1)

    def logistic_function(self, x, L=10.0, k=1.25, x_0=5.0):
        """ S-response curve function / logistic function
        :param x: input value that needs to be transformed to a point on the S-curve
        :param L: the curve's maximum value
        :param k: the steepness of the curve
        :param x_0: the x-value of the sigmoid's midpoint
        :return: the value of x as a point on the S-curve
        """
        value = L / (1+np.exp(-k*(x-x_0)))
        return value

    def sales_create_radio_dummies(self):
        """ Create dummy variables for various radio GRP intervals """
        # Initialize variables and specify intervals
        self.df['radio_dummy_0.1_2.5']  = int(0)
        self.df['radio_dummy_2.5_5']    = int(0)
        self.df['radio_dummy_5_7.5']    = int(0)
        self.df['radio_dummy_7.5_10']   = int(0)
        # Create the dummy variables based on the interval specified in the variable name
        for column in self.df.columns:
            if 'radio_dummy_' in column:
                column_split = column.split("_")
                dummy_start = float(column_split[2])
                dummy_end = float(column_split[3])
                self.df.ix[self.df['radio_grp'] >= dummy_start, column] = int(1)
                self.df.ix[self.df['radio_grp'] > dummy_end, column] = int(0)


    ###############################
    # MODEL, EXAMPLES AND PLOTS   #
    ###############################
    def create_example_s_curve_plot(self):
        # Initialize plot
        fig, ax = plt.subplots(figsize=(8, 4))
        # Plot example S-response curve
        x = np.arange(0, 20100, 100)
        y = self.logistic_function(x, L=10000, k=0.0007, x_0=10000)
        ax.plot(x, y, '-', label="Radio")
        # Set plot options and show plot
        ax.legend(loc='right')
        plt.xlim([0, 20000])
        plt.xlabel('Radio spend in euros')
        plt.ylabel('Additional sales')
        plt.title('Example of S-shaped response curve')
        plt.tight_layout()
        plt.grid()
        ax.get_xaxis().set_major_formatter(tkr.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax.get_yaxis().set_major_formatter(tkr.FuncFormatter(lambda x, p: format(int(x), ',')))
        plt.show()

    def model(self, x, y, constant=True, log_transform=False):
        # Run OLS regression, print summary and return results
        if constant:
            x = sm_tools.add_constant(x)        # Add constant
        if log_transform:
            y = np.log1p(y)
        ols_model = sm.OLS(y, x)                # Initialize model
        ols_result = ols_model.fit()            # Fit model
        print(ols_result.summary())             # Print statistics summary
        return ols_result

    @staticmethod
    def apply_date_formatting_to_axis(ax):
        """ Format x-axis of input plot to a readable date format """
        ax.xaxis.set_minor_locator(dates.WeekdayLocator(byweekday=(0), interval=1))
        ax.xaxis.set_minor_formatter(dates.DateFormatter('%d\n%a'))
        ax.xaxis.grid(True, which="minor")
        ax.yaxis.grid()
        ax.xaxis.set_major_locator(dates.MonthLocator())
        ax.xaxis.set_major_formatter(dates.DateFormatter('\n\n\n%b\n%Y'))
        return ax

    def run_example_1(self, show_predictions=True):
        # Initialize model
        x = self.df[['radio_grp', 'seasonality_monday', 'seasonality_tuesday', 'seasonality_thursday', 'seasonality_friday', 'seasonality_saturday', 'seasonality_sunday']]
        y = self.df['sales']
        # Run OLS regression and store results in res
        res = self.model(x, y)
        # Initialize plot
        fig, ax = plt.subplots(figsize=(10, 6))
        # Add line chart representing sales
        ax.plot(self.df.index, y, 'o-', label="Sales")
        if show_predictions:
            ax.plot(self.df.index, res.fittedvalues, 'ro-', label="Predicted sales using model")
        # Add bar chart representing GRP values
        ax.bar(self.df.index, self.df['radio_grp'], color='y', align='center')
        # Add text box in upper left in axes coords denoting the R squared of the model
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, "Adj. R squared: 0.896", transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
        # Set plot options and show plot
        ax = self.apply_date_formatting_to_axis(ax)
        if show_predictions:
            ax.legend(['Sales', 'Predicted sales using model', 'Radio GRPs'], loc='best')
        else:
            ax.legend(['Sales', 'Radio GRPs'], loc='best')
        plt.title('Example dataset')
        plt.tight_layout()
        plt.show()


    def run_example_2(self):
        # Add radio interval dummies
        self.sales_create_radio_dummies()
        # Show new radio dataset
        print(self.df[['radio_grp', 'radio_dummy_0.1_2.5', 'radio_dummy_2.5_5', 'radio_dummy_5_7.5', 'radio_dummy_7.5_10']].head(5))
        # Initialize model
        x = self.df[['radio_dummy_0.1_2.5', 'radio_dummy_2.5_5', 'radio_dummy_5_7.5', 'radio_dummy_7.5_10',
                'seasonality_monday', 'seasonality_tuesday', 'seasonality_thursday', 'seasonality_friday',
                'seasonality_saturday', 'seasonality_sunday']]
        y = self.df['sales']
        # Run OLS regression
        res = self.model(x, y)
        # Initialize plot
        fig, ax = plt.subplots(figsize=(8, 6))
        # Add line chart representing sales
        ax.plot(self.df.index, y, 'o-', label="Sales")
        ax.plot(self.df.index, res.fittedvalues, 'ro-', label="Predicted sales using model")
        # Add bar chart representing GRP values
        ax.bar(self.df.index, self.df['radio_grp'], color='y', align='center')
        # Set plot options and show plot
        ax = self.apply_date_formatting_to_axis(ax)
        ax.legend(['Sales', 'Predicted sales using model', 'Radio GRPs'], loc='best')
        plt.title('Example dataset')
        plt.tight_layout()
        plt.show()

    def create_predicted_s_curve_plot(self):
        # Initialize plot
        fig, ax = plt.subplots(figsize=(8, 4))
        # Plot dummy coefficients
        x = [1.3, 3.75, 6.25, 8.75]
        y = [0.2485, 1.9554, 8.7667, 9.3869]
        ax.plot(x, y, 'ro', label="Dummy values", markersize=9)
        # Plot fitted predicted S-response curve
        x = np.arange(0, 11, 0.1)
        y = self.logistic_function(x, L=9.75, k=1.4, x_0=5.)
        ax.plot(x, y, '-', label="Fitted predicted S-curve")
        # Set plot options and show plot
        ax.legend(loc='upper left')
        plt.xlim([0, 10])
        plt.xlabel('Radio GRPs')
        plt.ylabel('Additional sales')
        plt.title('Predicted S-shaped radio response curve')
        plt.tight_layout()
        plt.grid()
        plt.show()

    def run_example_3(self):
        # Transform radio GRP based on our predicted S-curve
        self.df['radio_grp'] = self.logistic_function(self.df['radio_grp'], L=9.75, k=1.4, x_0=5)
        # Initialize model
        x = self.df[['radio_grp', 'seasonality_monday', 'seasonality_tuesday', 'seasonality_thursday', 'seasonality_friday', 'seasonality_saturday', 'seasonality_sunday']]
        y = self.df['sales']
        # Run OLS regression and store results in res
        res = self.model(x, y)
        # Initialize plot
        fig, ax = plt.subplots(figsize=(10, 6))
        # Add line chart representing sales
        ax.plot(self.df.index, y, 'o-', label="Sales")
        ax.plot(self.df.index, res.fittedvalues, 'ro-', label="Predicted sales using model")
        # Add bar chart representing GRP values
        ax.bar(self.df.index, self.df['radio_grp'], color='y', align='center')
        # Add text box in upper left in axes coords denoting the R squared of the model
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, "Adj. R squared: 0.984", transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
        # Set plot options and show plot
        ax = self.apply_date_formatting_to_axis(ax)
        ax.legend(['Sales', 'Predicted sales using model', 'Radio GRPs after transformation'], loc='best')
        plt.title('Example dataset')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    ExampleSCurves().run()
