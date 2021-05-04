#!/usr/bin/env python3
# coding: utf-8
import random
if __name__ == "__main__":
    maxStates = 12
    maxSymbols = 10
    automata = 100000
    with open("automata.txt","w",newline='') as f:
        f.write("weights|transition\n")
        for i in range(automata):
            states = random.randint(3,maxStates)
            symbols = random.randint(2,maxSymbols)
            weights = '['+ ','.join([str(random.randint(1,maxSymbols)) for i in range(symbols)]) + ']'
            transition = '[' + '|'.join(['[' + ','.join([str(random.randint(0,maxStates)) for i in range(states)]) + ']' for j in range(symbols)]) + ']'
            line = ':'.join([weights,transition])+'\n'
            #print(line)
            f.write(line)
