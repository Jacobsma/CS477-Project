#!/usr/bin/env python3
# coding: utf-8
import itertools
import csv #CSV import/export
import heapq
import time #Timing
import argparse #CLI
import pathlib #CLI
import multiprocessing as mp #Multithreading

"""
TODOS:
    [X]--> Command Line Args
    [X]--> Timing 
    [X]--> Heuristics 
    [X]--> Passing Automata through CLI 
    [X]--> Optimization 
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
        self._time = list()
        if (self._shouldTime):
            self._totalTests = 0
            startTimeInit = time.time()
        super().__init__(transition,power)
        self._weights = weights
        self._graph = dict()
        self._shortest = ""
        self._shortestWords = dict()
        self._maxGraphDistances = dict()
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

        #Calculate max graph distances once
        self._maxGraphDistances = {vertex: (float('infinity'),"") for vertex in self._graph}
        return 


    def compute_shortest_words(self,starting_vertex:int) -> dict:

        #Check if we have already calculated for this starting_vertex
        if (starting_vertex in self._shortestWords):
            return self._shortestWords[starting_vertex]

        #We haven't already calculated shortest words
        distances = self._maxGraphDistances 
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

        # Store a copy of the shortest words from a given distance so that we can preform a look up for later calls
        self._shortestWords[starting_vertex] = distances
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

    def h1(self,P:set,T:tuple,w:str) -> float:
        image = self.compute_image_by_word(P, w[1])
        return w[0]/(len(P)-len(image))

    def h2(self,P:set,T:tuple,w:str) -> float:
        image = self.compute_image_by_word(T, w[1])
        return w[0]/(len(T)-len(image))

    def t3(self,P:set,T:tuple,m:int) -> bool:
        return set(P).issubset(set(T)) and len(P) == min(m, len(T))

    def h3(self,P:set,T:tuple,w:str) -> int:
        return  w[0]

    def h4(self,P:set,T:tuple,w:str) -> float:
        image = self.compute_image_by_word(T, w[1])
        return w[0]/((len(T)-len(image)) ** 2)


    def word_weight(self,word:str) -> int:
        weight = 0
        for letter in word:
            weight = weight + self._weights[ord(letter) - 97]
        return weight


    def approximate_weighted_synch(self,m:int, t, H):
        # Check if we should time
        if (self._shouldTime):
            startTimeAWS = time.time()

        # Set the power set
        self._power = m

        # Generate power automaton
        self._compute_automaton_m()
        aut = (self.powerTransitionFunction,self.keyToStateDict) 

        # Run graph so that our graph has the values for the power automaton
        self.graph(aut[0])

        # Get dummy longest words
        shortest_words_with_inf = self.compute_shortest_words(0)

        shortest_words = dict()
        for i in range(1,len(self._transitionFunction[0])):

            # Get the shortest words from the starting state i
            tmp_words = self.compute_shortest_words(i)
            for el in tmp_words:
                if shortest_words_with_inf[el][0] > tmp_words[el][0]:
                    shortest_words_with_inf[el] = tmp_words[el]

        # Copy over our computed shortest words
        for el in shortest_words_with_inf:
            if shortest_words_with_inf[el][0] != float('infinity'):
                shortest_words[el] = shortest_words_with_inf[el]

        # The total states in our automaton
        T = tuple(range(len(self._transitionFunction[0])))
        u = ""
        while len(T) > 1:
            current_shortest = float('infinity')
            w = ""
            for el in shortest_words:

                # The key represenation of the temp word
                P = set(aut[1][el])

                if t(P, T, m):
                    tmp = H(P,T,shortest_words[el])

                    # Check if the new weight is smaller
                    if tmp < current_shortest:
                        current_shortest = tmp
                        w = shortest_words[el][1]

            # Check if we found any synchronizing words
            if w == "":
                return None

            # Add our new word and reduce the remaining states
            u = u + w
            T = self.compute_image_by_word(T, w)

        if (self._shouldTime):
            self._totalTests += 1
            endTimeAWS = time.time()
        return Result(word=u, wordWeight=self.word_weight(u),time=endTimeAWS - startTimeAWS,hCount=self._totalTests)

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
        return f"\n----Automata----\n\n\nTransition Function:\n{self._transitionFunction}\n\nWeights:\n{self._weights}\n\nShortest Word:\n{self._shortest}\n\nShortest Word Length:\n{str(len(self._shortest))}\n\nPower:\n{self._power}\n\nPower Transition Function:\n{self._powerTransitionFunction}\n\nGraph:\n{self._graph}\n\nKeys To States:\n{self.keyToStateDict}\n\nTime to Initilize:\n{self._timeInit}s\n\nTime to Compute Shortest Word:\n{self._timeCompShort}s\n\nTime to Approximate Synchronizing Words:\n{', '.join(['Heuristic '+str(time[0])+': '+str(time[1])+'s' for time in self._time])}\n\nTotal Time to Approximate Synchronizing Words:\n{sum([time[1] for time in self._time])}s\n\n\n----------------\n"

class Result:
    def __init__(self,word:str,wordWeight:int,time:float,hCount:int):
        self._word = word 
        self._wordWeight = wordWeight 
        self._time = time 
        self._hCount = hCount 

    @property
    def word(self) -> str:
        return self._word

    @word.setter
    def word(self,resultWord:str):
        self._word = resultWord 
        return

    @property
    def wordWeight(self) -> int:
        return self._wordWeight

    @wordWeight.setter
    def wordWeight(self,resultWordWeight:int):
        self._wordWeight = resultWordWeight 
        return

    @property
    def time(self) -> float:
        return self._time

    @time.setter
    def time(self,resultTime:float):
        self._time = resultTime 
        return

    @property
    def hCount(self) -> int:
        return self._hCount

    @hCount.setter
    def hCount(self,resultHCount:int):
        self._hCount = resultHCount 
        return



def getAutomataFromFile(filename:str) -> list:
    automata = list()
    with open(filename, newline='') as f:
        next(f)
        for row in f:
            row = ''.join(row)
            weights,transitionFunction = row.split(':')
            weights = weights.strip("'[]")
            transitionFunction = transitionFunction.strip("'")

            x = list()
            for weight in weights.split(','):
                x.append(int(weight))

            weights = x

            x = list()
            transitionFunction = transitionFunction[1:-1]
            for symbol in transitionFunction.split('|'):
                y = list()
                for state in symbol.strip('[]').split(','):
                    y.append(int(state))

                x.append(y)
                
            transitionFunction = x
            automata.append((transitionFunction,len(transitionFunction[0]),weights))
    return automata

def addAutomaton(automaton:tuple) -> Automata:
    try:
        return Automata(transition=automaton[0],power=automaton[1],weights=automaton[2],shouldTime=True)
    except Exception as e:
        #print(e)
        #print(f"Invalid Automata:\t{automaton[2]},\t{automaton[0]},\t{automaton[1]}\n")
        return

def testHueristics(automaton:Automata) -> tuple:
    h1 = automaton.approximate_weighted_synch(4,automaton.t1,automaton.h1)
    h2 = automaton.approximate_weighted_synch(4,automaton.t1,automaton.h2)
    h3 = automaton.approximate_weighted_synch(4,automaton.t3,automaton.h3)
    h4 = automaton.approximate_weighted_synch(4,automaton.t1,automaton.h4)
    shortest = automaton.compute_shortest_word()
    return (h1,h2,h3,h4,shortest) 



if __name__ == "__main__":
            
    #TODO: Make a better description
    parser = argparse.ArgumentParser(description="Automata Creation and testing script")

    parser.add_argument("-e","--entry",type=ascii,help="The transition function and weights for the Automata\nUsage: -e '[weightOne,weightTwo,...,weightN] [[weightOneStateOne,weightOneStateTwo,...,weightNStateN]|[weightTwoStates]|...|[weightNStates]]'\nExample: -e '[1,10] [[1,2,3,4,0,0]|[1,2,4,4,5,0]]'")
    parser.add_argument("-f","--file",type=pathlib.Path,help="Specifies a file to be parsed for Automata\nUsage: -f 'path/to/file'")
    parser.add_argument("-t","--time",type=ascii,help="Specify a timing method for Automata\nUsage: -t '[total,init,shortest]'",choices=["'total'","'init'","'shortest'"],default="total")
    parser.add_argument("-v","--verbose",help="Enable verbose output for Automata\nUsage: -v",action='store_true')
    parser.add_argument("-o","--output",type=pathlib.Path,help="Specify an output file for Automata\nUsage: -o 'path/to/output'")

    args = vars(parser.parse_args())

    verbose = args['verbose']
    timing = args['time'].strip("'")
    output = args['output']
    inputFile = args['file']

    amtWorkers = 20

    tests = list()
    if (inputFile is not None and inputFile.exists()):
        with mp.Pool(processes=amtWorkers) as pool:
            tests = pool.map(addAutomaton, getAutomataFromFile(inputFile))

    else:
        weights, transitionFunction = list(),list()
        if (args['entry'] is not None):
            weights,transitionFunction = args['entry'].split()
            weights = weights.strip("'[]")
            transitionFunction = transitionFunction.strip("'")

            x = list()
            for weight in weights.split(','):
                x.append(int(weight))

            weights = x

            x = list()
            transitionFunction = transitionFunction[1:-1]
            for symbol in transitionFunction.split('|'):
                y = list()
                for state in symbol.strip('[]').split(','):
                    y.append(int(state))

                x.append(y)
                
            transitionFunction = x

            try:
                tests.append(Automata(transitionFunction,len(transitionFunction[0]),weights,shouldTime=True))
            except Exception:
                print(f"Invalid Automata:\t{weights},\t{transitionFunction},\t{len(transitionFunction[0])}\n")
                exit()
        else:
            parser.parse_args(['--help'])

    e = list()
    for x in range(len(tests)):
        if (tests[x] is not None):
            e.append(tests[x])
    tests = e

    heuristics = list()
    with mp.Pool(processes=amtWorkers) as pool:
        heuristics = pool.map(testHueristics, tests)

    e = list()
    for x in range(len(tests)):
        if (None not in heuristics[x]):
            e.append((tests[x],heuristics[x]))
    heuristics = e

    if (output is not None):
        with output.open('w') as outFile:
            for result in heuristics:
                outFile.write(f"Shortest:\t{result[1][-1]}\nLength:\t{str(len(result[1][-1]))}\nHeuristics:\n{', '.join(['Heuristic '+str(h+1)+'-> ('+result[1][h].word +' -- '+str(result[1][h].wordWeight)+')'  for h in range(len(result[1])-1)])}\nTime:\n{', '.join(['Heuristic '+str(result[1][t].hCount)+'-> '+str(result[1][t].time)+'s' for t in range(len(result[1])-1)])}\n\n")
                if (verbose):
                    outFile.write(str(result[0])) 
    else:
        for result in heuristics:
            print(f"Shortest:\t{result[1][-1]}\nLength:\t{str(len(result[1][-1]))}\nHeuristics:\n{', '.join(['Heuristic '+str(h+1)+'-> ('+result[1][h].word +' -- '+str(result[1][h].wordWeight)+')'  for h in range(len(result[1])-1)])}\nTime:\n{', '.join(['Heuristic '+str(result[1][t].hCount)+'-> '+str(result[1][t].time)+'s' for t in range(len(result[1])-1)])}\n\n")
            if (verbose):
                print(result[0])

