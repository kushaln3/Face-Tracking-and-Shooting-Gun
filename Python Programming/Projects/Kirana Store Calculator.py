# Write a python program which keeps adding the numbers inputting by the user and when q is pressed it prints the output
sum = 0
while True:
    number = input('Enter the price: ')
    if number != 'q':
        try:
            sum += int(number)
        except:
            print("Enter a valid number.")
    else:
        print(f"Your total bill amount is: {sum}")
        break
    print(f"Your Orde Total so far: {sum}")