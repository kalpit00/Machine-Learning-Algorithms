import os
from math import pi

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from model import MyNetwork
from sklearn.model_selection import train_test_split

try:
    from tqdm import tqdm
    # tqdm is a progress bar library with many cool features.
    # You can install tqdm via `pip install tqdm` in the terminal.
    # Your code will still run even if you don't install tqdm.
except ImportError:
    tqdm = None


def get_device() -> torch.device:
    """
    DO NOT MODIFY.

    Set the device to GPU if available, else CPU

    Args:
        None

    Returns:
        torch.device
            'cuda' if NVIDIA GPU is available
            'mps' if Apple M1/M2 GPU is available
            'cpu' if no GPU is available
    """

    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


device = get_device()


def pre_process(file_path: str) -> dict:
    """
    Preprocess the stock data.

    First read the data from the csv file, then get a list of unique stock
    names. For each stock, sort the data by date and save the sorted data
    into a dictionary.

    Args:
        file_path (str): path to the csv file

    Returns:
        stock_dict (dict): a dictionary of stock dataframes
            key (str): stock name
            value (pd.DataFrame): sorted historical stock price dataframe,

            Example (first 5 rows of stock_dict['AAPL']):
                Date        Open     High     Low      Close     Volume    Stock
                1984-09-07  0.42388  0.42902  0.41874  0.42388   23220030  AAPL
                1984-09-10  0.42388  0.42516  0.41366  0.42134   18022532  AAPL
                1984-09-11  0.42516  0.43668  0.42516  0.42902   42498199  AAPL
                1984-09-12  0.42902  0.43157  0.41618  0.41618   37125801  AAPL
                1984-09-13  0.43927  0.44052  0.43927  0.43927   57822062  AAPL
                ...         ...      ...      ...      ...       ...       ...         
    """

    df = pd.read_csv(file_path)
    stocks = sorted(list(set(df['Stock']))) # stocks in alphabetical order

    stock_dict = dict()
    for stock in stocks:
        # >>> YOUR CODE HERE
        stock_dict[stock] = df[df['Stock'] == stock].sort_values('Date')
        # <<< END YOUR CODE

    return stock_dict


def plot_data(stock_dict: dict) -> None:
    """
    Plot the stock price vs. date for each stock.

    A 2x2 subplot is created. Each subplot shows the historical stock CLOSE
    price of each stock. The x-axis is the date and the y-axis is the stock
    CLOSE price.

    Args:
        stock_dict (dict): a dictionary of preprocessed stock dataframes

    Returns:
        None
    """

    stocks = list(stock_dict.keys())
    fig, axs = plt.subplots(2, 2)
    fig.tight_layout(pad=3.0)
    for i in range(2):
        for j in range(2):
            axs[i, j].plot(stock_dict[stocks[i*2+j]]['Close'].values)
            axs[i, j].set_title(f'{stocks[i*2+j]}')

    for ax in axs.flat:
        ax.set(xlabel='days', ylabel='close price')

    plt.savefig(os.path.join(os.path.dirname(__file__), 'stocks_history.png'))


def split_stock(stock_info: pd.DataFrame) -> tuple:
    """
    Given a stock dataframe of a single stock, split the data into train and
    valid data.

    The first 70% of the data is used for training and the rest 30% is used
    for validation. Data should not be shuffled because time series data
    should be processed in chronological order.

    Args:
        stock_info (pd.DataFrame): a dataframe containing the stock close
            price information sorted by date (ascending)

            Example: 
                stock_dict['AAPL'] is one example of stock_info

    Returns:
        tuple of 4 numpy arrays: (x_train_stock, y_train_stock, x_val_stock, y_val_stock)

            Example: 
                Assume there are 14 days' data of stock 'AAPL', then you can
                construct 10 data samples. Splitting them into a training set
                and a test set will get 7 training samples and 3 test samples.

                x_train_stock = [[AAPL_0, AAPL_1, AAPL_2, AAPL_3, AAPL_4],
                                 [AAPL_2, AAPL_3, AAPL_3, AAPL_4, AAPL_5]
                                 ...
                                 [AAPL_6, AAPL_7, AAPL_8, AAPL_9, AAPL_10]]
                y_train_stock = [AAPL_5, AAPL_6, ..., AAPL_11]

                x_val_stock = [[AAPL_7, AAPL_8, AAPL_9, AAPL_10, AAPL_11],
                                 [AAPL_8, AAPL_9, AAPL_10, AAPL_11, AAPL_12]
                                 [AAPL_9, AAPL_10, AAPL_11, AAPL_12, AAPL_13]]
                y_val_stock = [AAPL_12, AAPL_13, AAPL14]

                Finally, combine them into a tuple and return.
    """

    x = []
    y = []

    # >>> YOUR CODE HERE
    valList = stock_info['Close'].tolist()
    size = len(valList)
    splitPos = int((size - 5) * 0.7)

    for i in range(size - 5):
        x.append([valList[i],valList[i+1],valList[i+2],valList[i+3],valList[i+4]])
        y.append(valList[i+5])
    
    x_train_stock = np.array(x[0:splitPos])
    x_val_stock = np.array(x[splitPos:])
    y_train_stock = np.array(y[0:splitPos])
    y_val_stock = np.array(y[splitPos:])

    # <<< END YOUR CODE

    return (x_train_stock, y_train_stock, x_val_stock, y_val_stock)


def get_train_valid(stock_dict: dict) -> tuple:
    """
    Given a dictionary of stock dataframes, split each stock dataframe into
    train and valid data. Then combine all the train and valid data into
    a large train and a large valid dataset.

    Args:
        stock_dict (dict): a dictionary of preprocessed stock dataframes.
            Example: same as the returned value of pre_process() function

    Returns:
        tuple of np.array: (x_train, y_train, x_val, y_val)

            Example:
                Assume each stock of [AAPL, FB, TELA, MSFT] has 14 days of data,
                then each stock can construct 10 data samples. After splitting, 
                each has 7 training samples and 3 valid samples. Combine training
                samples and valid samples to get 7 * 4 = 28 training samples and
                3 * 4 = 12 valid samples.
                
                In this example, the final dimensions are:
                x_train: (28, 5)
                y_train: (28, )
                x_val: (12, 5)
                y_val: (12, )


    """
    # >>> YOUR CODE HERE
    x_train, y_train, x_val, y_val = np.array([]), np.array([]), np.array([]), np.array([])

    for stock in stock_dict:
        
        x_train_stock, y_train_stock, x_val_stock, y_val_stock = split_stock(stock_dict[stock])
        x_train = np.append(x_train, x_train_stock)
        y_train = np.append(y_train, y_train_stock)
        x_val = np.append(x_val, x_val_stock)
        y_val = np.append(y_val, y_val_stock)

    
    x_train = np.reshape(x_train, (int(np.shape(x_train)[0]/5), 5))
    x_val = np.reshape(x_val, (int(np.shape(x_val)[0]/5), 5))

    # <<< END YOUR CODE
    return (x_train, y_train, x_val, y_val)


def my_NLLloss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the negative log likelihood loss using equation (6) in the handout

    Args:
        pred (torch.Tensor): a tensor of shape (N, 2), N is the number
            of samples. For i between 0 and N-1, pred[i][0] is the 
            predicted price, and pred[i][1] is the risk of prediction.
        y (torch.Tensor): a tensor of shape (N, )

    Returns:
        nll_loss (torch.Tensor): a scalar tensor of NLL loss
            example: tensor(0.1234)
    """

    # >>> YOUR CODE HERE
    nll_loss = 0

    for i in range(pred.shape[0]):
        nll_loss += torch.log(torch.sqrt(torch.tensor(2*pi))) + 0.5*pred[i][1] + ((y[i]-pred[i][0])**2) / (2*torch.exp(pred[i][1]))
    # <<< END YOUR CODE
    return nll_loss


def train(data: tuple, max_epochs: int = 200, seed=12345) -> tuple:
    """
    Train and validate the model on the given data with your own network and
    loss function. After training, plot the training loss and validation loss
    vs. epoch number. Because of the randomness of the first few epochs, we 
    start plotting from the 5th epoch.

    Args:
        data (tuple of np.array): (x_train, y_train, x_val, y_val)
        max_epochs (int): maximum number of epochs to train
        device (torch.device): device to use for training
        seed (int): random seed for reproducibility, default is 12345

    Returns:
        None
    """
    torch.manual_seed(seed)

    if tqdm is not None:
        iterator = tqdm(range(max_epochs))
    else:
        iterator = range(max_epochs)
        
    net = MyNetwork(5, 100, 2)

    x_train, y_train, x_val, y_val = data

    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_val = torch.from_numpy(x_val)
    y_val = torch.from_numpy(y_val)

    x_train.to(device)
    y_train.to(device)
    x_val.to(device)
    y_val.to(device)
    net.to(device)

    train_losses = []
    val_losses = []

    print('---------- Training has started: -------------')
    optimizer = optim.Adam(net.parameters(), lr=0.001) # DEFINE YOUR OPTIMIZER

    for epoch in iterator: # DO NOT CHANGE THIS LINE
        # >>> YOUR CODE HERE
        train_out = net(x_train.float())
        val_out = net(x_val.float())

        train_loss = my_NLLloss(train_out, y_train)
        val_loss = my_NLLloss(val_out, y_val)
        
        optimizer.zero_grad()
        train_loss.backward()
        val_loss.backward()
        optimizer.step()

        # <<< END YOUR CODE

        # DO NOT MODIFY THE BELOW
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        if tqdm is not None:
            iterator.set_description(f' Epoch: {epoch+1}')
            iterator.set_postfix(train_loss=round(train_loss.item(), 1),
                                 val_loss=round(val_loss.item(), 1))
        else:
            print(
                f'epoch {epoch+1}: train_loss = {train_loss}, val_loss = {val_loss}')
    
    print('---------- Training ended. -------------\n')
    
    plt.figure()
    epochs = list(range(max_epochs))
    plt.plot(epochs[5:], train_losses[5:])
    plt.plot(epochs[5:], val_losses[5:])
    plt.legend(['train', 'val'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training and Validation Losses')
    plt.savefig(os.path.join(os.path.dirname(__file__), 'training_curve.png'))
    plt.close()

    return net, train_losses[-1], val_losses[-1]


def plot_predictions(model: nn.Module, stock_dict: dict) -> None:
    """
    Given a trained model and a dictionary of stock dataframes, predict the
    stock 'Close' prices for each stock. Plot the predicted 'Close' prices vs.
    the actual 'Close' prices for each stock (only plot the first 50 data samples).

    Check the handout for an example.

    Args:
        model (nn.Module): a trained model
        stock_dict (dict): a dictionary of stock dataframes

    Returns:
        None
    """

    fig, axs = plt.subplots(2, 2) # axs may be useful
    fig.tight_layout(pad=3.0) # give some space between subplots

    for k, stock in enumerate(list(stock_dict.keys())):
        (_, _, x_val, y_val) = split_stock(stock_dict[stock])

        pred = model(torch.Tensor(x_val).to(device)).detach().cpu().numpy()

        pred_prices, pred_risks = pred[:, 0], np.sqrt(np.exp(pred[:, 1]))
        rmse = np.sqrt(np.mean((pred_prices - y_val) ** 2))
        print(f'RMSE for {stock} is: {rmse}')

        i, j = k // 2, k % 2

        prices_range = [pred_prices - pred_risks, pred_prices + pred_risks]
        axs[i, j].plot(y_val[:50])
        axs[i, j].plot(pred_prices[:50])
        axs[i, j].legend(['real', 'pred'])
        axs[i, j].fill_between(list(range(50)), prices_range[0]
                               [:50], prices_range[1][:50], color=None, alpha=.15)
        axs[i, j].set_title(f'{stock}')
    
    plt.savefig(os.path.join(os.path.dirname(__file__), 'predictions.png'))
    print('Predictions plotted.')


"""
-------------------------------------------------------------------------------------------
THE CODE BELOW IS FOR EVALUATION. PLEASE DO NOT CHANGE!
-------------------------------------------------------------------------------------------
"""


if __name__ == '__main__':

    os.system('cls' if os.name == 'nt' else 'clear')

    file_path = os.path.join(os.path.dirname(
        __file__), 'datasets/stock_train.csv')
    stock_dict = pre_process(file_path)

    plot_data(stock_dict)

    data = get_train_valid(stock_dict)

    net, train_loss, val_loss = train(data, max_epochs=1000)
    plot_predictions(net, stock_dict)
