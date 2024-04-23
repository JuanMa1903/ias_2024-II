#INTEGRANTES
#Airton Jairo Sampayo Solano
#Keyner David Barrios Mercado
#Eliecer farid Ureche Torres
#Juan Diego Marin Soler

import random
import math
import timeit
import matplotlib.pyplot as plt

fit = lambda x, y: (x**4) + (y**4) - (1.8*x**3) - (1.5*y**3) - (2*x**2) - (2*y**2) + 3*x + 3*y

def eval_fitness(population):
    return [fit(x, y) for x, y in population]

def mutar(parent, sigma):
    x, y = parent
    if sigma <= epsilon:
        sigma = epsilon
    return (x + sigma * random.gauss(0, 0.5), y + sigma * random.gauss(0, 0.5))

def recombinarLocal(parent1, parent2):
    numRandom = random.uniform(0,1)
    return ((parent1[0] * numRandom) + (parent2[0] * (1 - numRandom)),
            (parent1[1] * numRandom) + (parent2[1] * (1 - numRandom)))

epsilon = math.pow(10, -5)
euler = math.e
n = 2
t= 1/math.sqrt(n)

sigmaIni = 0.5
sigmaAct = sigmaIni * (euler)**(t*random.gauss(0,1))

miu = 5
lda = miu * 7
gens = 100
start = -5
end = -2

sesenta = int(round(lda * 0.60))
cuarenta = int(round(lda * 0.40))

random.seed(1)
parents = [(random.uniform(start, end), random.uniform(start, end)) for _ in range(miu)]

TiempoInicio = timeit.default_timer()
MejorMin = 0
MejorXY = None
MejorIteracion = 0
HistorialFitness = [] 

for i in range(gens):
    # offspringsRecom = [recombinarGlobal(parents) for _ in range(cuarenta)]
    offspringsRecom = [recombinarLocal(random.choice(parents), random.choice(parents)) for _ in range(cuarenta)]
    # offspringsRecom = [recombinarDiscreta(parents) for _ in range(cuarenta)]
    offspringsMutar = [mutar(random.choice(parents), sigmaAct) for _ in range(sesenta)]
    population = parents + offspringsMutar + offspringsRecom
    fitness = eval_fitness(population)
    MejorMinimoGen = min(fitness)
    HistorialFitness.append(MejorMinimoGen)
    if MejorMinimoGen < MejorMin:
        MejorMin = MejorMinimoGen
        MejorXY = population[fitness.index(MejorMinimoGen)]
        MejorIteracion = i
    parents = [population[j] for j in sorted(range(len(population)), key=lambda x: fitness[x])[:miu]]

TiempoFinal = timeit.default_timer()
Duracion = TiempoFinal - TiempoInicio

print("El tiempo en encontrar la mejor solucion fue de: ", Duracion)
print("La mejor solucion encontrada:", MejorMin)
print("Valores con los que se obtuvo esa solucion:", MejorXY)
print("Mejor solucion encontrada en la iteracion: ", MejorIteracion)

plt.figure(figsize=(10, 5))
plt.plot(range(gens), HistorialFitness, marker='o', linestyle='-', color='b')
plt.title('Convergencia del Algoritmo a lo largo de las generaciones')
plt.xlabel('Generación')
plt.ylabel('Mejor Fitness')
plt.grid(True)
plt.show()

