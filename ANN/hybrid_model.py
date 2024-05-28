import numpy as np
import random

# Definimos la función de activación relu
def relu(x):
    return np.maximum(0, x)

# Definimos la función softmax para la salida
def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps, axis=0)

# Función de forward pass para calcular la salida de la red neuronal
def forward_pass(weights, x):
    # Reshape de los pesos
    W1 = weights[:16].reshape(2, 8)
    W2 = weights[16:].reshape(8, 4)
    
    # Capa oculta
    hidden_layer = relu(np.dot(x, W1))
    
    # Capa de salida
    output_layer = softmax(np.dot(hidden_layer, W2))
    
    return output_layer

# Función de fitness (Mean Squared Error)
def fitness(weights, X, Y):
    predictions = np.array([forward_pass(weights, x) for x in X])
    mse = np.mean((predictions - Y) ** 2)
    return mse

# Inicializamos la población de posibles soluciones
def initialize_population(pop_size, weight_dim):
    return np.random.randn(pop_size, weight_dim)

# Estrategia de evolución diferencial
def differential_evolution(X, Y, generations, pop_size=20, F=0.8, CR=0.9, tolerance=1e-5):
    weight_dim = 48  # 16 pesos en W1 y 32 pesos en W2
    population = initialize_population(pop_size, weight_dim)
    best_individual = None
    best_fitness = float('inf')
    
    for generation in range(generations):
        new_population = []
        for i in range(pop_size):
            indices = list(range(pop_size))
            indices.remove(i)
            a, b, c = random.sample(indices, 3)
            mutant = population[a] + F * (population[b] - population[c])
            cross_points = np.random.rand(weight_dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, weight_dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            f = fitness(trial, X, Y)
            if f < fitness(population[i], X, Y):
                new_population.append(trial)
                if f < best_fitness:
                    best_fitness = f
                    best_individual = trial
            else:
                new_population.append(population[i])
        population = np.array(new_population)
        
        
        # Condición de parada si el fitness es menor que el umbral de tolerancia
        if best_fitness < tolerance:
            print(f'Stopping early at generation {generation} with fitness {best_fitness}')
            break
    
    return best_individual

# Generamos datos de entrada para entrenamiento
def generate_training_data(num_points=1000):
    X = np.random.uniform(-1, 1, (num_points, 2))
    Y = []
    for x1, x2 in X:
        if x1 > 0 and x2 > 0:
            Y.append([1, 0, 0, 0])
        elif x1 < 0 and x2 > 0:
            Y.append([0, 1, 0, 0])
        elif x1 < 0 and x2 < 0:
            Y.append([0, 0, 1, 0])
        else:
            Y.append([0, 0, 0, 1])
    return X, np.array(Y)

# Generamos los datos de entrenamiento
X_train, Y_train = generate_training_data()

# Solicitamos al usuario el número máximo de generaciones
generations = int(input("Ingrese el número máximo de generaciones: "))

# Ejecutamos la evolución diferencial para encontrar los mejores pesos
best_weights = differential_evolution(X_train, Y_train, generations)

# Probar los pesos obtenidos con un punto de prueba
def classify_point(x, weights):
    output = forward_pass(weights, x)
    return np.argmax(output)

# Ejemplo de clasificación de un punto
test_point = np.array([120, -120])
quadrant = classify_point(test_point, best_weights)
print(f'El punto {test_point} pertenece al cuadrante {quadrant + 1}')
