#!/usr/bin/env python
# coding: utf-8
import itertools

import heapq


class AutomataCore:
    def __init__(self,transition:list,power:int=1):
        self._transitionFunction = transition
        self._powerTransitionFunction = list() 
        self._power = power
        self._stateToKey = dict()
        self._keyToState = dict()
        self._compute_automaton_m()

    @property
    def transitionFunction(self):
        return self._transitionFunction

    @transitionFunction.setter
    def transitionFunction(self,transition:list):
        self._transitionFunction = transition
        return 

    @property
    def powerTransitionFunction(self):
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
    def stateToKey(self,state:tuple):
        return self._stateToKey[state]

    @stateToKey.setter
    def stateToKey(self,state:tuple,key:int):
        self._stateToKey[state] = key
        return

    @property
    def keyToState(self,key:int):
        return self._keyToState[key]

    @keyToState.setter
    def keyToState(self,key:int,state:tuple):
        self._keyToState[key] = state
        return

    @property
    def keyToStateDict(self):
        return self._keyToState

    def _compute_image(self,letter:int,tup:tuple) -> tuple:
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
        #returned power func and keyToState
        return

class Automata(AutomataCore):

    def __init__(self,transition:list,power:int,weights:list):
        super().__init__(transition,power)
        self._weights = weights
        self._graph = dict()

    def graph(self,transition:list):
      self._graph = dict([(i, dict()) for i in range(len(transition[0]))])
      symbols = [chr(ord('a') + i) for i in range(len(self._weights))]
      for letter, weight, symbol in sorted(zip(transition, self._weights, symbols), key = lambda x: x[1], reverse=True):
        i = 0
        for el in letter:
          self._graph[el][i] = (weight, symbol)
          i = i+1
      return self._graph 


    #TODO: maybe store this?
    def compute_shortest_words(self,starting_vertex):
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

    def compute_image_by_word(self, tup, word):
      result = tup
      for symbol in word:
        letter = self._transitionFunction[ord(symbol) - 97]
        result = self._compute_image(letter, result)
      return result

    def compute_shortest_word(self):
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
      return list(shortest_words.items())[-1][1]


    def t1(self,P,T,m):
      return P.issubset(set(T)) and len(P) > 1

    def H1(self,P,T,w):
      image = self.compute_image_by_word(P, w[1])
      return w[0]/(len(P)-len(image))

    def H2(self,P,T,w):
      image = self.compute_image_by_word(T, w[1])
      return w[0]/(len(T)-len(image))

    def t3(self,P,T,m):
      return set(P).issubset(set(T)) and len(P) == min(m, len(T))

    def H3(self,P,T,w):
     return  w[0]

    def H4(self,P,T,w):
      image = self.compute_image_by_word(T, w[1])
      return w[0]/((len(T)-len(image)) ** 2)


    def word_weight(self,word):
      weight = 0
      for letter in word:
        weight = weight + self._weights[ord(letter) - 97]
      return weight


    def approximate_weighted_synch(self,m, t, H):
      self._power = m
      self._compute_automaton_m()
      aut = (self.powerTransitionFunction,self.keyToStateDict) 
      graph = self.graph(aut[0])
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
      return (u, len(u), self.word_weight(u))
    
    def __str__(self):
        return f"\n----Automata----\n\n\nTransition Function:\n{self._transitionFunction}\n\nWeights:\n{self._weights}\n\nPower:\n{self._power}\n\nPower Transition Function:\n{self._powerTransitionFunction}\n\nGraph:\n{self._graph}\n\nKeys To States:\n{self.keyToStateDict}\n\n----------------\n"

if __name__ == "__main__":
    mTwo = Automata([[1,2,3,4,5,6,7,8,9,10,11,0],[0,1,2,3,4,5,6,7,8,9,10,0], [1,1,2,3,4,5,6,7,8,9,10,11], [0,1,2,3,3,5,6,7,8,9,10,11]],2, [1,6,2,7]) 
    print("m = 2")
    print("H1" ,mTwo.approximate_weighted_synch(2,mTwo.t1,mTwo.H1))
    print("H2" ,mTwo.approximate_weighted_synch(2,mTwo.t1,mTwo.H2))
    print("H3" ,mTwo.approximate_weighted_synch(2,mTwo.t3,mTwo.H3))
    print("H4" ,mTwo.approximate_weighted_synch(2,mTwo.t1,mTwo.H4))
    print("m = 3")
    print("H1" ,mTwo.approximate_weighted_synch(3,mTwo.t1,mTwo.H1))
    print("H2" ,mTwo.approximate_weighted_synch(3,mTwo.t1,mTwo.H2))
    print("H3" ,mTwo.approximate_weighted_synch(3,mTwo.t3,mTwo.H3))
    print("H4" ,mTwo.approximate_weighted_synch(3,mTwo.t1,mTwo.H4))
    print("m = 4")
    print("H1" ,mTwo.approximate_weighted_synch(4,mTwo.t1,mTwo.H1))
    print("H2" ,mTwo.approximate_weighted_synch(4,mTwo.t1,mTwo.H2))
    print("H3" ,mTwo.approximate_weighted_synch(4,mTwo.t3,mTwo.H3))
    print("H4" ,mTwo.approximate_weighted_synch(4,mTwo.t1,mTwo.H4))
    print("m = 5")
    print("H1" ,mTwo.approximate_weighted_synch(5,mTwo.t1,mTwo.H1))
    print("H2" ,mTwo.approximate_weighted_synch(5,mTwo.t1,mTwo.H2))
    print("H3" ,mTwo.approximate_weighted_synch(5,mTwo.t3,mTwo.H3))
    print("H4" ,mTwo.approximate_weighted_synch(5,mTwo.t1,mTwo.H4))
    print("m = 6")
    print("H1" ,mTwo.approximate_weighted_synch(6,mTwo.t1,mTwo.H1))
    print("H2" ,mTwo.approximate_weighted_synch(6,mTwo.t1,mTwo.H2))
    print("H3" ,mTwo.approximate_weighted_synch(6,mTwo.t3,mTwo.H3))
    print("H4" ,mTwo.approximate_weighted_synch(6,mTwo.t1,mTwo.H4))

    #print(mTwo)

    c = mTwo.compute_shortest_word()

    d = Automata([[1,2,3,4,5,6,7,8,9,10,11,0],[0,1,2,3,4,5,6,7,8,9,10,0], [1,1,2,3,4,5,6,7,8,9,10,11], [0,1,2,3,3,5,6,7,8,9,10,11]],2, [1,1,1,1]).compute_shortest_word() 

    print("least weight" ,c, len(c[1]))
    print("shortest" ,d, len(d[1]))
