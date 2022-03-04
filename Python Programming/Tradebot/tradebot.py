import time
import random

total_balance = 1000
trade_amount = 1
return_percent = 90 /100
time_per_trade = 3

def trade(option, trade_amount, time_per_trade = 3):
    global total_balance
    if trade_amount > total_balance:
        print('Not enough money to place trade')
        exit()
    
    total_balance -= trade_amount

    trade_result = random.randint(0,1) # 0 means call and 1 mean put

    if option == 'call' or option == 'buy':
        print(f"{option} trade executed successfully.\nShowing result in {time_per_trade} seconds...")
        time.sleep(time_per_trade)
        if trade_result == 0:
            total_balance += round(trade_amount + trade_amount*return_percent, 1)
            print(f"\nOption: {option}\nTrade amount: {trade_amount}\nResult: PROFIT\nProfit: {trade_amount*return_percent}\nTotal balance: {total_balance}\n")
        else:
            print(f"\nOption: {option}\nTrade amount: {trade_amount}\nResult: LOSS\nLoss: {trade_amount}\nTotal balance: {total_balance}\n")

    elif option == 'put' or option == 'sell':
        print(f"{option} trade executed successfully.\nShowing result in {time_per_trade} seconds...")
        time.sleep(time_per_trade)
        if trade_result == 1:
            total_balance += round(trade_amount + trade_amount*return_percent, 1)
            print(f"\nOption: {option}\nTrade amount: {trade_amount}\nResult: PROFIT\nProfit: {trade_amount*return_percent}\nTotal balance: {total_balance}\n")
        else:
            print(f"\nOption: {option}\nTrade amount: {trade_amount}\nResult: LOSS\nLoss: {trade_amount}\nTotal balance: {total_balance}\n")
      

    else:
        print('Error: Invalid option')
    
    total_balance = round(total_balance, 1)


if __name__ == "__main__":
    print(type(trade('call', 1, 0)))
