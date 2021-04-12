#!/usr/bin/env python
# coding: utf-8
import itertools

import heapq


def make_graph(transition_function, weights):
  graph = dict([(i, dict()) for i in range(len(transition_function[0]))])
  symbols = [chr(ord('a') + i) for i in range(len(weights))]
  for letter, weight, symbol in sorted(zip(transition_function, weights, symbols), key = lambda x: x[1], reverse=True):
    i = 0
    for el in letter:
      graph[el][i] = (weight, symbol)
      i = i+1
  return graph


def compute_shortest_words(graph, starting_vertex):
    distances = {vertex: (float('infinity'),"") for vertex in graph}
    distances[starting_vertex] = (0,"")

    pq = [(0, starting_vertex, "")]
    while len(pq) > 0:
        current_distance, current_vertex, current_word = heapq.heappop(pq)

        if current_distance > distances[current_vertex][0]:
            continue

        for neighbor, (weight,symbol) in graph[current_vertex].items():
            distance = current_distance + weight

            if distance < distances[neighbor][0]:
                distances[neighbor] = (distance, symbol + current_word)
                heapq.heappush(pq, (distance, neighbor, symbol + current_word))

    return distances



def compute_image(letter, tup):
  im = list()
  for state in tup:
    im.append(letter[state])
  return tuple(sorted(set(im)))

def compute_image_by_word(transition_function, tup, word):
  result = tup
  for symbol in word:
    letter = transition_function[ord(symbol) - 97]
    result = compute_image(letter, result)
  return result

def compute_automaton_m(transition_function, m):
  n = len(transition_function[0])
  states = [i for i in range(n)]
  states_to_keys = dict()
  keys_to_states = dict()
  power_trans_func = list()
  k = 0
  for i in range(m):
    for tup in itertools.combinations(states, i + 1):
      states_to_keys[tup] = k
      keys_to_states[k] = tup
      k = k+1
  for letter in transition_function:
    power_letter = list()
    for el in states_to_keys:
      el_key = states_to_keys[el]
      image = compute_image(letter, el)
      power_letter.append(states_to_keys[image])
    power_trans_func.append(power_letter)
  return (power_trans_func, keys_to_states)


def compute_shortest_word(transition_function, weights):
  power_aut = compute_automaton_m(transition_function, len(transition_function[0]))
  graph = make_graph(power_aut[0], weights)
  shortest_words = compute_shortest_words(graph, 0)
  for i in range(1,len(transition_function[0])):
    tmp = compute_shortest_words(graph, i)
    for el in tmp:
      if shortest_words[el][0] > tmp[el][0]:
        shortest_words[el] = tmp[el]
  return list(shortest_words.items())[-1][1]


def t1(P,T,m):
  return P.issubset(set(T)) and len(P) > 1

def H1(P,T,w, transition_function):
  image = compute_image_by_word(transition_function, P, w[1])
  return w[0]/(len(P)-len(image))


def H2(P,T,w, transition_function):
  image = compute_image_by_word(transition_function, T, w[1])
  return w[0]/(len(T)-len(image))

def t3(P,T,m):
  return set(P).issubset(set(T)) and len(P) == min(m, len(T))

def H3(P,T,w, transition_function):
 return  w[0]

def H4(P,T,w, transition_function):
  image = compute_image_by_word(transition_function, T, w[1])
  return w[0]/((len(T)-len(image)) ** 2)


def word_weight(word, weights):
  weight = 0
  for letter in word:
    weight = weight + weights[ord(letter) - 97]
  return weight


def approximate_weighted_synch(transition_function, weights, m, t, H):
  aut = compute_automaton_m(transition_function, m)
  graph = make_graph(aut[0], weights)
  shortest_words_with_inf = compute_shortest_words(graph, 0)
  shortest_words = dict()
  for i in range(1,len(transition_function[0])):
    tmp_words = compute_shortest_words(graph, i)
    for el in tmp_words:
      if shortest_words_with_inf[el][0] > tmp_words[el][0]:
        shortest_words_with_inf[el] = tmp_words[el]
  for el in shortest_words_with_inf:
    if shortest_words_with_inf[el][0] != float('infinity'):
      shortest_words[el] = shortest_words_with_inf[el]
  T = tuple(range(len(transition_function[0])))
  u = ""
  while len(T) > 1:
    current_shortest = float('infinity')
    w = ""
    for el in shortest_words:
      P = set(aut[1][el])
      if t(P, T, m):
        tmp = H(P,T,shortest_words[el], transition_function)
        if tmp < current_shortest:
          current_shortest = tmp
          w = shortest_words[el][1]
    if w == "":
      return ""
    u = u + w
    T = compute_image_by_word(transition_function, T, w)
  return (u, len(u), word_weight(u, weights))
print("m = 2")
print("H1" ,approximate_weighted_synch([[1,2,3,4,5,6,7,8,9,10,11,0],[0,1,2,3,4,5,6,7,8,9,10,0], [1,1,2,3,4,5,6,7,8,9,10,11], [0,1,2,3,3,5,6,7,8,9,10,11]], [1,6,2,7], 2,t1,H1))
print("H2" ,approximate_weighted_synch([[1,2,3,4,5,6,7,8,9,10,11,0],[0,1,2,3,4,5,6,7,8,9,10,0], [1,1,2,3,4,5,6,7,8,9,10,11], [0,1,2,3,3,5,6,7,8,9,10,11]], [1,6,2,7], 2,t1,H2))
print("H3" ,approximate_weighted_synch([[1,2,3,4,5,6,7,8,9,10,11,0],[0,1,2,3,4,5,6,7,8,9,10,0], [1,1,2,3,4,5,6,7,8,9,10,11], [0,1,2,3,3,5,6,7,8,9,10,11]], [1,6,2,7], 2,t3,H3))
print("H4" ,approximate_weighted_synch([[1,2,3,4,5,6,7,8,9,10,11,0],[0,1,2,3,4,5,6,7,8,9,10,0], [1,1,2,3,4,5,6,7,8,9,10,11], [0,1,2,3,3,5,6,7,8,9,10,11]], [1,6,2,7], 2,t1,H4))
print("m = 3")
print("H1" ,approximate_weighted_synch([[1,2,3,4,5,6,7,8,9,10,11,0],[0,1,2,3,4,5,6,7,8,9,10,0], [1,1,2,3,4,5,6,7,8,9,10,11], [0,1,2,3,3,5,6,7,8,9,10,11]], [1,6,2,7], 3,t1,H1))
print("H2" ,approximate_weighted_synch([[1,2,3,4,5,6,7,8,9,10,11,0],[0,1,2,3,4,5,6,7,8,9,10,0], [1,1,2,3,4,5,6,7,8,9,10,11], [0,1,2,3,3,5,6,7,8,9,10,11]], [1,6,2,7], 3,t1,H2))
print("H3" ,approximate_weighted_synch([[1,2,3,4,5,6,7,8,9,10,11,0],[0,1,2,3,4,5,6,7,8,9,10,0], [1,1,2,3,4,5,6,7,8,9,10,11], [0,1,2,3,3,5,6,7,8,9,10,11]], [1,6,2,7], 3,t3,H3))
print("H4" ,approximate_weighted_synch([[1,2,3,4,5,6,7,8,9,10,11,0],[0,1,2,3,4,5,6,7,8,9,10,0], [1,1,2,3,4,5,6,7,8,9,10,11], [0,1,2,3,3,5,6,7,8,9,10,11]], [1,6,2,7], 3,t1,H4))
print("m = 4")
print("H1" ,approximate_weighted_synch([[1,2,3,4,5,6,7,8,9,10,11,0],[0,1,2,3,4,5,6,7,8,9,10,0], [1,1,2,3,4,5,6,7,8,9,10,11], [0,1,2,3,3,5,6,7,8,9,10,11]], [1,6,2,7], 4,t1,H1))
print("H2" ,approximate_weighted_synch([[1,2,3,4,5,6,7,8,9,10,11,0],[0,1,2,3,4,5,6,7,8,9,10,0], [1,1,2,3,4,5,6,7,8,9,10,11], [0,1,2,3,3,5,6,7,8,9,10,11]], [1,6,2,7], 4,t1,H2))
print("H3" ,approximate_weighted_synch([[1,2,3,4,5,6,7,8,9,10,11,0],[0,1,2,3,4,5,6,7,8,9,10,0], [1,1,2,3,4,5,6,7,8,9,10,11], [0,1,2,3,3,5,6,7,8,9,10,11]], [1,6,2,7], 4,t3,H3))
print("H4" ,approximate_weighted_synch([[1,2,3,4,5,6,7,8,9,10,11,0],[0,1,2,3,4,5,6,7,8,9,10,0], [1,1,2,3,4,5,6,7,8,9,10,11], [0,1,2,3,3,5,6,7,8,9,10,11]], [1,6,2,7], 4,t1,H4))
print("m = 5")
print("H1" ,approximate_weighted_synch([[1,2,3,4,5,6,7,8,9,10,11,0],[0,1,2,3,4,5,6,7,8,9,10,0], [1,1,2,3,4,5,6,7,8,9,10,11], [0,1,2,3,3,5,6,7,8,9,10,11]], [1,6,2,7], 5,t1,H1))
print("H2" ,approximate_weighted_synch([[1,2,3,4,5,6,7,8,9,10,11,0],[0,1,2,3,4,5,6,7,8,9,10,0], [1,1,2,3,4,5,6,7,8,9,10,11], [0,1,2,3,3,5,6,7,8,9,10,11]], [1,6,2,7], 5,t1,H2))
print("H3" ,approximate_weighted_synch([[1,2,3,4,5,6,7,8,9,10,11,0],[0,1,2,3,4,5,6,7,8,9,10,0], [1,1,2,3,4,5,6,7,8,9,10,11], [0,1,2,3,3,5,6,7,8,9,10,11]], [1,6,2,7], 5,t3,H3))
print("H4" ,approximate_weighted_synch([[1,2,3,4,5,6,7,8,9,10,11,0],[0,1,2,3,4,5,6,7,8,9,10,0], [1,1,2,3,4,5,6,7,8,9,10,11], [0,1,2,3,3,5,6,7,8,9,10,11]], [1,6,2,7], 5,t1,H4))
print("m = 6")
print("H1" ,approximate_weighted_synch([[1,2,3,4,5,6,7,8,9,10,11,0],[0,1,2,3,4,5,6,7,8,9,10,0], [1,1,2,3,4,5,6,7,8,9,10,11], [0,1,2,3,3,5,6,7,8,9,10,11]], [1,6,2,7], 6,t1,H1))
print("H2" ,approximate_weighted_synch([[1,2,3,4,5,6,7,8,9,10,11,0],[0,1,2,3,4,5,6,7,8,9,10,0], [1,1,2,3,4,5,6,7,8,9,10,11], [0,1,2,3,3,5,6,7,8,9,10,11]], [1,6,2,7], 6,t1,H2))
print("H3" ,approximate_weighted_synch([[1,2,3,4,5,6,7,8,9,10,11,0],[0,1,2,3,4,5,6,7,8,9,10,0], [1,1,2,3,4,5,6,7,8,9,10,11], [0,1,2,3,3,5,6,7,8,9,10,11]], [1,6,2,7], 6,t3,H3))
print("H4" ,approximate_weighted_synch([[1,2,3,4,5,6,7,8,9,10,11,0],[0,1,2,3,4,5,6,7,8,9,10,0], [1,1,2,3,4,5,6,7,8,9,10,11], [0,1,2,3,3,5,6,7,8,9,10,11]], [1,6,2,7], 6,t1,H4))

c = compute_shortest_word([[1,2,3,4,5,6,7,8,9,10,11,0],[0,1,2,3,4,5,6,7,8,9,10,0], [1,1,2,3,4,5,6,7,8,9,10,11], [0,1,2,3,3,5,6,7,8,9,10,11]], [1,6,2,7])
d = compute_shortest_word([[1,2,3,4,5,6,7,8,9,10,11,0],[0,1,2,3,4,5,6,7,8,9,10,0], [1,1,2,3,4,5,6,7,8,9,10,11], [0,1,2,3,3,5,6,7,8,9,10,11]], [1,1,1,1])

print("least weight" ,c, len(c[1]))
print("shortest" ,d, len(d[1]))



# In[ ]:


from itertools import permutations  

def relabel(trans_func, perm):
  pairs = [(i,trans_func[i]) for i in range(len(trans_func))]
  result = [0 for i in trans_func]
  for pair in pairs:
    result[perm[pair[0]]] = perm[pair[1]]
  return result


#print(rotate([1,2,3,4,0,0],2))


#perms = permutations([0,1, 2, 3, 4])  
#for perm in perms:
# t = relabel([1,2,0,2,3], list(perm))
# print(t,approximate_weighted_synch([[1,2,0,2,3],t],[1,1], 2, t4, H4))
# t = relabel([1,2,0,1,2], list(perm))
# print(t,approximate_weighted_synch([[1,2,0,1,2],t],[1,1], 2, t4, H4))




#print(compute_shortest_word([[1,2,3,4,0,0],[1,2,4,4,5,0]],[1,10]))
#print(compute_shortest_word([[1,2,3,4,0,0],[1,2,4,4,5,0]],[1,10]))

#print(compute_shortest_word([[1,2,0,4,5,3,5],[4,5,5,0,3,6,1]],[1,1]))
#print(approximate_weighted_synch([[3,2,4,5,3,6,4,5,6] ,[1,7,4,7,6,6,1,4,0]],[1,1], 4, t4, H4))
#print(approximate_weighted_synch([[3,2,4,5,3,6,4,5,6] ,[1,7,4,7,6,6,1,4,0]],[1,1], 4, t3, H3))
#print(approximate_weighted_synch([[3,2,4,5,3,6,4,5,6] ,[1,7,4,7,6,6,1,4,0]],[1,1], 4, t2, H2))
#print(approximate_weighted_synch([[3,2,4,5,3,6,4,5,6] ,[1,7,4,7,6,6,1,4,0]],[1,1], 4, t1, H1))
#print(approximate_weighted_synch([[1,2,3,4,5,6,7,8,9,10,11,0] ,[0,0,2,3,4,6,7,5,8,9,10,11]],[1,1], 2, t1, H1))



#print("xd",approximate_weighted_synch([[1,2,3,0] ,[2,1,2,3]],[1,1], 4, t1, H4))

