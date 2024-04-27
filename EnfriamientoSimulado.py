#INTEGRANTES
#Airton Jairo Sampayo Solano
#Keyner David Barrios Mercado
#Eliecer farid Ureche Torres
#Juan Diego Marin Soler

import random
import math
import timeit
import matplotlib.pyplot as plt

def Funcion(x, y):
    return -(y + 47) * math.sin(math.sqrt(abs(x/2 + (y + 47)))) - x * math.sin(math.sqrt(abs(x - (y + 47))))

TiempoInicio = timeit.default_timer()

def simulated_annealing(funcion, limites, iteraciones, salto, temp, enfriamientoRate, tempFinal):

    xIni, yIni = random.uniform(*limites), random.uniform(*limites)
    evalIni = funcion(xIni, yIni)
    xMejor, yMejor, evalMejor = xIni, yIni, evalIni
    iteracionMejor = 0
    valores_por_iteracion = []
    
    random.seed(1)
    for i in range(iteraciones):
        temp *= enfriamientoRate
        if temp>tempFinal:
            xNueva, yNueva = xIni + salto * (random.uniform(-1.0, 1.0)), yIni + salto * (random.uniform(-1.0, 1.0))
            xNueva, yNueva = max(limites[0], min(xNueva, limites[1])), max(limites[0], min(yNueva, limites[1]))
            evalNueva = funcion(xNueva, yNueva)
            diff = evalNueva - evalIni
            valores_por_iteracion.append(evalNueva)
            
            if diff < 0 or random.random() < math.exp(-diff / temp):
                xIni, yIni, evalIni = xNueva, yNueva, evalNueva
                if evalIni < evalMejor:
                    xMejor, yMejor, evalMejor = xIni, yIni, evalIni
                    iteracionMejor = i
        else: break

    return (xMejor, yMejor, evalMejor, iteracionMejor, valores_por_iteracion, Duracion)

TiempoFinal = timeit.default_timer()
Duracion = TiempoFinal - TiempoInicio

Limites = (-512, 512)
Iteraciones = 1200
Salto = 20           
TempIni = 78000
EnfriamientoRate = 0.78
TempFinal = 0.0001

xMejor, yMejor, evalMejor, iteracionMejor, valores_por_iteracion, Duracion = simulated_annealing(Funcion, Limites, Iteraciones, Salto, TempIni, EnfriamientoRate, TempFinal)

print('El tiempo en encontrar la mejor solucion fue de: %.17f ' % Duracion)
print("La mejor solucion encontrada:", evalMejor)
print('Valores con los que se obtuvo esa solucion: x=%.6f, y=%.6f)' % (xMejor, yMejor))
print("Mejor solucion encontrada en la iteracion: ", iteracionMejor)

plt.figure(figsize=(10, 5))
plt.plot(valores_por_iteracion, marker='o', linestyle='-', color='b')
plt.title('Valor de la función objetivo por iteración')
plt.xlabel('Iteración')
plt.ylabel('Valor de la función objetivo')
plt.grid(True)
plt.show()
