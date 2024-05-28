import numpy as np
import random

def relu(x):
    return np.maximum(0, x)


def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps, axis=0)

# Funciónpara calcular la salida de la red neuronal
def forward_pass(weights, x):
    W1 = weights[:16].reshape(2, 8)
    W2 = weights[16:].reshape(8, 4)

    # Capa oculta
    hidden_layer = relu(np.dot(x, W1))
    
    # Capa de salida
    output_layer = softmax(np.dot(hidden_layer, W2))
    
    return output_layer

# Función de fitness (M S E)
def fitness(weights, X, Y):
    predictions = np.array([forward_pass(weights, x) for x in X])
    mse = np.mean((predictions - Y) ** 2)
    return mse

# Inicializamos la población
def initialize_population(pop_size, weight_dim):
    return np.random.randn(pop_size, weight_dim)

def differential_evolution(X, Y, generations, pop_size=20, F=0.8, CR=0.9, tolerance=1e-5):
    weight_dim = 48 
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
        
        # Condición de parada 
        if best_fitness < tolerance:
            print(f'Paramos en la generacion {generation} Con un fitness {best_fitness}')
            break
    
    return best_individual

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

X_train, Y_train = generate_training_data()

x = float(input("Ingrese la coordenada X del punto de prueba: "))
y = float(input("Ingrese la coordenada Y del punto de prueba: "))
test_point = np.array([x, y])

generations = int(input("Ingrese el número máximo de generaciones: "))

# Ejecutamos la evolución diferencial para encontrar los mejores pesos
best_weights = differential_evolution(X_train, Y_train, generations)

# Probar los pesos 
def classify_point(x, weights):
    output = forward_pass(weights, x)
    return np.argmax(output)

# Clasificar el punto de prueba
quadrant = classify_point(test_point, best_weights)
print(f'El punto {test_point} pertenece al cuadrante {quadrant + 1}')
