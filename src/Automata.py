#!/usr/bin/env python
# coding: utf-8
import itertools
import csv #CSV import/export
import heapq
import time #Timing
import argparse #CLI
import pathlib #CLI

"""
TODOS:
    [X]--> Command Line Args
    [X]--> Timing 
    [ ]--> Heuristics 
    [/]--> Passing Automata through CLI 
    [?]--> Optimization 
    [X]--> OOP Refactor 

Legend:
    X --> Done
    / --> Partly Done
    ? --> Optional
"""

class AutomataCore:
    def __init__(self,transition:list,power:int=1):
        self._transitionFunction = transition
        self._powerTransitionFunction = list() 
        self._power = power
        self._stateToKey = dict()
        self._keyToState = dict()
        self._compute_automaton_m()

    @property
    def transitionFunction(self) -> list:
        return self._transitionFunction

    @transitionFunction.setter
    def transitionFunction(self,transition:list):
        self._transitionFunction = transition
        return 

    @property
    def powerTransitionFunction(self) -> list:
        return self._powerTransitionFunction

    @powerTransitionFunction.setter
    def powerTransitionFunction(self,transition:list):
        self._powerTransitionFunction = transition
        return 

    @property
    def power(self):
        return self._power

    @power.setter
    def power(self,Power:int):
        self._power = Power
        return

    @property
    def stateToKey(self,state:tuple) -> int:
        return self._stateToKey[state]

    @stateToKey.setter
    def stateToKey(self,state:tuple,key:int):
        self._stateToKey[state] = key
        return

    @property
    def keyToState(self,key:int) -> tuple:
        return self._keyToState[key]

    @keyToState.setter
    def keyToState(self,key:int,state:tuple):
        self._keyToState[key] = state
        return

    @property
    def keyToStateDict(self) -> dict:
        return self._keyToState

    def _compute_image(self,letter:list,tup:tuple) -> tuple:
        im = list()
        for state in tup:
            im.append(letter[state])
        return tuple(sorted(set(im)))

    def _compute_automaton_m(self):
        n = len(self._transitionFunction[0])
        states = [i for i in range(n)]
        k = 0
        self._powerTransitionFunction = list()
        for i in range(self._power):
            for tup in itertools.combinations(states, i + 1):
                self._stateToKey[tup] = k
                self._keyToState[k] = tup
                k = k+1
        for letter in self._transitionFunction:
            power_letter = list()
            for el in self._stateToKey:
                el_key = self._stateToKey[el]
                image = self._compute_image(letter, el) 
                power_letter.append(self._stateToKey[image])
            self._powerTransitionFunction.append(power_letter)
        return

class Automata(AutomataCore):

    def __init__(self,transition:list,power:int,weights:list,shouldTime=False):
        self._shouldTime = shouldTime
        self._timeInit = 0
        self._timeCompShort = 0
        self._time = 0
        if (self._shouldTime):
            startTimeInit = time.time()
        super().__init__(transition,power)
        self._weights = weights
        self._graph = dict()
        self._shortest = ""
        if (self._shouldTime):
            endTimeInit = time.time()
            self._timeInit = endTimeInit - startTimeInit

    def graph(self,transition:list):
        self._graph = dict([(i, dict()) for i in range(len(transition[0]))])
        symbols = [chr(ord('a') + i) for i in range(len(self._weights))]
        for letter, weight, symbol in sorted(zip(transition, self._weights, symbols), key = lambda x: x[1], reverse=True):
            i = 0
            for el in letter:
                self._graph[el][i] = (weight, symbol)
                i = i+1
        return 


    #TODO: maybe store this?
    def compute_shortest_words(self,starting_vertex:int) -> dict:
        distances = {vertex: (float('infinity'),"") for vertex in self._graph}
        distances[starting_vertex] = (0,"")

        pq = [(0, starting_vertex, "")]
        while len(pq) > 0:
            current_distance, current_vertex, current_word = heapq.heappop(pq)

            if current_distance > distances[current_vertex][0]:
                continue

            for neighbor, (weight,symbol) in self._graph[current_vertex].items():
                distance = current_distance + weight

                if distance < distances[neighbor][0]:
                    distances[neighbor] = (distance, symbol + current_word)
                    heapq.heappush(pq, (distance, neighbor, symbol + current_word))

        return distances

    def compute_image_by_word(self, tup:tuple, word:str) -> tuple:
        result = tup
        for symbol in word:
            letter = self._transitionFunction[ord(symbol) - 97]
            result = self._compute_image(letter, result)
        return result

    def compute_shortest_word(self) -> list:
        if (self._shouldTime):
            startTimeCompShort = time.time()
        self._power = len(self._transitionFunction[0])
        self._compute_automaton_m()
        aut = (self.powerTransitionFunction,self.keyToStateDict) 
        graph = self.graph(aut[0])
        shortest_words = self.compute_shortest_words(0)
        for i in range(1,len(self._transitionFunction[0])):
            tmp = self.compute_shortest_words(i)
            for el in tmp:
                if shortest_words[el][0] > tmp[el][0]:
                    shortest_words[el] = tmp[el]
        if (self._shouldTime):
            endTimeCompShort = time.time()
            self._timeCompShort = endTimeCompShort - startTimeCompShort
        self._shortest = list(shortest_words.items())[-1][1][1]
        return self._shortest 


    def t1(self,P:set,T:tuple,m:int) -> bool:
        return P.issubset(set(T)) and len(P) > 1

    def H1(self,P:set,T:tuple,w:str) -> float:
        image = self.compute_image_by_word(P, w[1])
        return w[0]/(len(P)-len(image))

    def H2(self,P:set,T:tuple,w:str) -> float:
        image = self.compute_image_by_word(T, w[1])
        return w[0]/(len(T)-len(image))

    def t3(self,P:set,T:tuple,m:int) -> bool:
        return set(P).issubset(set(T)) and len(P) == min(m, len(T))

    def H3(self,P:set,T:tuple,w:str) -> int:
        return  w[0]

    def H4(self,P:set,T:tuple,w:str) -> float:
        image = self.compute_image_by_word(T, w[1])
        return w[0]/((len(T)-len(image)) ** 2)


    def word_weight(self,word:str) -> int:
        weight = 0
        for letter in word:
            weight = weight + self._weights[ord(letter) - 97]
        return weight


    def approximate_weighted_synch(self,m:int, t, H) -> tuple:
        self._power = m
        if (self._shouldTime):
            startTimeAWS = time.time()
        self._compute_automaton_m()
        aut = (self.powerTransitionFunction,self.keyToStateDict) 
        self.graph(aut[0])
        shortest_words_with_inf = self.compute_shortest_words(0)
        shortest_words = dict()
        for i in range(1,len(self._transitionFunction[0])):
            tmp_words = self.compute_shortest_words(i)
            for el in tmp_words:
                if shortest_words_with_inf[el][0] > tmp_words[el][0]:
                    shortest_words_with_inf[el] = tmp_words[el]
        for el in shortest_words_with_inf:
            if shortest_words_with_inf[el][0] != float('infinity'):
                shortest_words[el] = shortest_words_with_inf[el]
        T = tuple(range(len(self._transitionFunction[0])))
        u = ""
        while len(T) > 1:
            current_shortest = float('infinity')
            w = ""
            for el in shortest_words:
                P = set(aut[1][el])
                if t(P, T, m):
                    tmp = H(P,T,shortest_words[el])
                    if tmp < current_shortest:
                        current_shortest = tmp
                        w = shortest_words[el][1]
            if w == "":
                return ""
            u = u + w
            T = self.compute_image_by_word(T, w)
        if (self._shouldTime):
            endTimeAWS = time.time()
            self._time = endTimeAWS - startTimeAWS
        return (u, len(u), self.word_weight(u))

    def getTiming(self,timing:str):
        timingType = str.lower(timing)
        if (timingType == "total"):
            return self._time
        elif (timingType == "shortest"):
            return self._timeCompShort
        elif (timingType == "init"):
            return self._timeInit
        else:
            return -1

    def getShortest(self):
        return self._shortest
    
    def __str__(self):
        return f"\n----Automata----\n\n\nTransition Function:\n{self._transitionFunction}\n\nWeights:\n{self._weights}\n\nShortest Word:\n{self._shortest}\n\nShortest Word Length:\n{str(len(self._shortest))}\n\nPower:\n{self._power}\n\nPower Transition Function:\n{self._powerTransitionFunction}\n\nGraph:\n{self._graph}\n\nKeys To States:\n{self.keyToStateDict}\n\nTime to Initilize:\n{self._timeInit}\n\nTime to Compute Shortest Word:\n{self._timeCompShort}\n\nTime to Approximate Shortest Word:\n{self._time}\n\n\n----------------\n"

#TODO: support more than 9 states
def getAutomataFromCSV(filename:str) -> list:
    automata = list()
    with open(filename, newline='') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        next(reader)
        for row in reader:
            weights = list()
            for weight in range(len(row[0])):
                weights.append(int(row[0][weight]))

            transFunct = list()
            for funct in range(len(row)-1):
                states = list()
                for state in range(len(row[funct+1])):
                    states.append(int(row[funct+1][state]))
                        


                transFunct.append(states)
            automata.append((transFunct,len(transFunct[0]),weights))
    return automata

if __name__ == "__main__":
            
    #TODO: Make a better description
    parser = argparse.ArgumentParser(description="Automata Creation and testing script")

    parser.add_argument("-e","--entry",type=ascii,help="The transition function and weights for the Automata\nUsage: -e '[weights]' '[weightOneStates,weightTwoStates,...,weightNStates]'")
    parser.add_argument("-f","--file",type=pathlib.Path,help="Specifies a file to be parsed for Automata\nUsage: -f 'path/to/file'")
    parser.add_argument("-t","--time",type=ascii,help="Specify a timing method for Automata\nUsage: -t '[total,init,shortest]'",choices=["'total'","'init'","'shortest'"])
    parser.add_argument("-v","--verbose",help="Enable verbose output for Automata\nUsage: -v",action='store_true')
    parser.add_argument("-o","--output",type=pathlib.Path,help="Specify an output file for Automata\nUsage: -o 'path/to/output'")

    args = vars(parser.parse_args())

    verbose = args['verbose']
    timing = args['time']
    output = args['output']
    inputFile = args['file']
    weights,transitionFunction = args['entry'].split()

    tests = list()
    if (inputFile.exists()):
        for automaton in getAutomataFromCSV(inputFile):
            try:
                tests.append(Automata(automaton[0],automaton[1],automaton[2],shouldTime=True))
            except Exception as e:
                print(f"Invalid Automata:\t{automaton[0]}\t{automaton[1]}\t{automaton[2]}\n")
                continue

    maxTime = (0,None)

    for test in tests:
        test.approximate_weighted_synch(4,test.t1,test.H1)
        test.approximate_weighted_synch(4,test.t1,test.H2)
        test.approximate_weighted_synch(4,test.t3,test.H3)
        test.approximate_weighted_synch(4,test.t1,test.H4)

        test.compute_shortest_word()

        if (timing is not None):
            newMax = max(maxTime[0],test.getTiming(timing.strip("'")))
            if (newMax > maxTime[0]):
                maxTime = (newMax,test)

    if (output is not None):
        with output.open('a') as outFile:
            if (verbose):
                for test in tests:
                    outFile.write(str(test)) 
            else:
                for test in tests:
                    outFile.write("Shortest:\t" + test.getShortest() + "\nLength:\t" + str(len(test.getShortest())) + "\n\n")
    else:
        for test in tests:
            print("Shortest:\t" + test.getShortest())

    #TODO: Parse -e inputs
    #print(weights,transitionFunction)


