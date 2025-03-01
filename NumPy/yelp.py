import os
from random import seed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Yelp:
    def __init__(self, filename, sample_size=None, seed: int = 42) -> None:
        """
        Class constructor. DO NOT MODIFY.

        Args:
            filename: The name of the file containing the data set
            sample_size: The size of the data set to be used, None as default
            seed: The seed for the random number generator

        Returns:
            None
        """

        if sample_size is not None:
            np.random.seed(seed)
            self.df = self.read_data(filename).sample(sample_size)
        else:
            self.df = self.read_data(filename)

    def read_data(self, filename: str) -> pd.DataFrame:
        """
        Reads the data from the given file path.

        Args:
            filename: The path to the file to be read

        Returns:
            A pandas DataFrame containing the data
        """

        # >> YOUR CODE HERE
        return pd.read_csv(filename)
        # END OF YOUR CODE <<

    def average_rating(self) -> np.array:
        """
        Calculates the average star ratings of restaurants in each state, and
        then save the ratings in a lists in alphabetical order of the state
        abbreviation, starting with AZ in ascending order.

        Returns:
            A 1-D np.array containing the average star ratings of restaurants.
        """

        # >> YOUR CODE HERE
        return np.array(self.df.groupby("state").mean()["stars"].sort_index().values)
        # END OF YOUR CODE <<

    def rating_stats_given_review_count(self, count: int) -> tuple:
        """
        Calculates the mean and the standard deviation of the star ratings of 
        restaurants that have at least that many ratings.

        Args:
            count: The minimum number of reviews for a restaurant to be 
            included in the calculation

        Returns:
            A tuple of two floats representing the mean and standard deviation.
        """

        # >> YOUR CODE HERE

        ratings = self.df[self.df.reviewCount >= count].stars
        mean = np.mean(ratings)
        std = np.std(ratings)

        # END OF YOUR CODE <<

        return float(mean), float(std)

    def plot_cdf(self, state: str = "NV") -> plt.Figure:
        """
        Generate plot of CDF of review counts for restaurants

        Args:
            state: The state to plot the CDF for

        Returns:
            A matplotlib Figure object
        """
        fig, ax = plt.subplots(1, figsize=(6, 4), constrained_layout=True)

        # >> YOUR CODE HERE

        x = self.df.loc[self.df.state == state, "reviewCount"].values
        points = np.unique(x)
        cdf = (x.reshape(-1, 1) <= points.reshape(1, -1)).mean(0)
        ax.plot(points, cdf)
        ax.set_title(f"CDF of review counts for {state}")
        ax.set_ylabel("Proportion")
        ax.set_xlabel("log(reviewCount)")
        ax.set_xscale('log')
        # END OF YOUR CODE <<
        return fig

    def make_boxplots(self) -> plt.Figure:
        """
        Create boxplots with distribution of number of checkins for each star 
        rating level

        Args:
            None

        Returns:
            A matplotlib Figure object
        """
        fig, ax = plt.subplots(1, figsize=(6, 4), constrained_layout=True)

        # >> YOUR CODE HERE

        star_values = np.sort(self.df.stars.unique())
        counts = [self.df.loc[self.df.stars == s, "checkins"].tolist()
                  for s in star_values]
        ax.boxplot(counts, labels=star_values)

        ax.set_title("Number of checkins for each star rating level")
        ax.set_ylabel("Number of checkins")
        ax.set_xlabel("Star rating")
        ax.set_yscale('log')
        # END OF YOUR CODE <<

        return fig


"""
-------------------------------------------------------------------------------------------
THE CODE BELOW IS FOR EVALUATION. PLEASE DO NOT CHANGE!
-------------------------------------------------------------------------------------------
"""


def evaluate_yelp():
    """
    Test your implementation in yelp.py.

    Args:
        None

    Returns:
        None
    """

    print('\n\n-------------Yelp Dataset-------------\n')
    print('This test is not exhaustive by any means. It only tests if')
    print('your implementation runs without errors.\n')

    yelp = Yelp(os.path.join(os.path.dirname(__file__), "dataset/yelp.csv"))

    yelp.average_rating()

    yelp.rating_stats_given_review_count(1)

    fig = yelp.plot_cdf()
    fig.savefig(os.path.join(os.path.dirname(__file__), "yelp_cdf.png"))

    fig = yelp.make_boxplots()
    fig.savefig(os.path.join(os.path.dirname(__file__), "yelp_boxplots.png"))

    print('Test yelp.py: passed')


if __name__ == '__main__':

    os.system('cls' if os.name == 'nt' else 'clear')

    evaluate_yelp()
