import random
if __name__ == "__main__":
    with open("automata.csv","a",newline='') as f:
        for i in range(10000):
            states = random.randint(3,9)
            symbols = random.randint(2,9)
            weights = ''.join([chr(ord('0')+random.randint(1,9)) for i in range(symbols)])
            transition = ','.join([''.join([chr(ord('0')+random.randint(0,9)) for i in range(states)]) for j in range(symbols)])
            line = ','.join([weights,transition])+'\n'
            #print(line)
            f.write(line)
