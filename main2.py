import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# =============================================
# Выбор задачи
# =============================================
print("Выберите задачу:")
print("1. Поиск минимума функции (вариант 15)")
print("2. Генерация бинарных последовательностей с заданным PSL (вариант 15)")
choice = int(input("Введите номер задачи (1 или 2): "))

if choice == 1:
    # =============================================
    # ЗАДАЧА 1: Поиск минимума функции (вариант 15)
    # =============================================
    # Параметры алгоритма
    POPULATION_SIZE = 50
    CHROMOSOME_LENGTH = 20
    SEARCH_MIN, SEARCH_MAX = -5.0, 5.0
    MUTATION_PROB = 0.2
    CROSSOVER_PROB = 0.9
    MAX_GENERATIONS = 50
    TOURNAMENT_SIZE = 3
    EARLY_STOP_GEN = 10


    # Целевая функция (вариант 15)
    def target_function(x):
        return 9 * x - 9.7 * np.abs(np.sin(37 * x)) * np.cos(27 * x)


    # Двоичное кодирование и декодирование
    def binary_encode(x):
        x_int = int((x - SEARCH_MIN) / (SEARCH_MAX - SEARCH_MIN) * (2 ** CHROMOSOME_LENGTH - 1))
        return format(x_int, f'0{CHROMOSOME_LENGTH}b')


    def binary_decode(chromosome):
        x_int = int(chromosome, 2)
        return SEARCH_MIN + (SEARCH_MAX - SEARCH_MIN) * x_int / (2 ** CHROMOSOME_LENGTH - 1)


    # Инициализация популяции
    def initialize_population():
        return [binary_encode(np.random.uniform(SEARCH_MIN, SEARCH_MAX)) for _ in range(POPULATION_SIZE)]


    # Мутация и скрещивание
    def mutate(chromosome):
        return ''.join(
            [bit if np.random.random() > MUTATION_PROB else '1' if bit == '0' else '0' for bit in chromosome])


    def crossover(p1, p2):
        if np.random.random() < CROSSOVER_PROB:
            pt = np.random.randint(1, CHROMOSOME_LENGTH)
            return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]
        return p1, p2


    # Турнирный отбор
    def tournament_selection(population, fitness):
        return [population[np.argmin([fitness[i] for i in np.random.choice(len(population), TOURNAMENT_SIZE)])] for _ in
                range(len(population))]


    # Основной алгоритм
    def genetic_algorithm():
        population = initialize_population()
        best_fitness_history = []
        best_x, best_fitness = None, float('inf')
        no_improve = 0

        for gen in range(MAX_GENERATIONS):
            decoded = [binary_decode(chrom) for chrom in population]
            fitness = [target_function(x) for x in decoded]
            current_best = min(fitness)

            if current_best < best_fitness:
                best_fitness = current_best
                best_x = decoded[np.argmin(fitness)]
                no_improve = 0
            else:
                no_improve += 1

            best_fitness_history.append(best_fitness)
            print(f"Поколение {gen + 1}: x = {best_x:.5f}, f(x) = {best_fitness:.5f}")

            if no_improve >= EARLY_STOP_GEN:
                print(f"Ранняя остановка: нет улучшений {EARLY_STOP_GEN} поколений")
                break

            selected = tournament_selection(population, fitness)
            new_population = []
            for i in range(0, len(selected) - 1, 2):
                child1, child2 = crossover(selected[i], selected[i + 1])
                new_population.extend([mutate(child1), mutate(child2)])
            population = new_population[:POPULATION_SIZE]

        # Визуализация
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(best_fitness_history, 'r-', label='Лучшая приспособленность')
        plt.xlabel('Поколение')
        plt.ylabel('f(x)')
        plt.legend()

        plt.subplot(1, 2, 2)
        x_vals = np.linspace(SEARCH_MIN, SEARCH_MAX, 1000)
        plt.plot(x_vals, target_function(x_vals), 'b-', label='Целевая функция')
        plt.scatter(best_x, best_fitness, c='red', label=f'Найденный минимум: x={best_x:.3f}')
        plt.legend()
        plt.grid()
        plt.show()

        return best_x, best_fitness


    best_x, best_fitness = genetic_algorithm()
    print(f"Результат: x = {best_x:.5f}, f(x) = {best_fitness:.5f}")

elif choice == 2:
    # =============================================
    # ЗАДАЧА 2: Генерация бинарных последовательностей (вариант 15)
    # =============================================
    # Параметры алгоритма (вариант 15)
    N = 38  # Длина последовательности
    P = 70  # Размер популяции
    PSL = 4  # Макс. уровень боковых лепестков
    K = 3  # Количество искомых последовательностей
    Pk = 0.85  # Вероятность скрещивания
    Pm = 0.15  # Вероятность мутации
    SELECTION_METHOD = "рулетка"  # Способ селекции


    # Автокорреляционная функция (АКФ)
    def autocorrelation(sequence):
        n = len(sequence)
        acf = []
        for shift in range(n):
            sum_val = 0
            for i in range(n - shift):
                sum_val += sequence[i] * sequence[i + shift]
            acf.append(sum_val)
        return acf


    # Фитнес-функция (минимизация PSL)
    def fitness(sequence):
        acf = autocorrelation(sequence)
        main_lobe = acf[0]
        side_lobes = [abs(x) for x in acf[1:]]  # Боковые лепестки (без главного)
        max_side_lobe = max(side_lobes) if side_lobes else 0
        return max_side_lobe  # Чем меньше, тем лучше


    # Генерация случайной последовательности
    def generate_sequence():
        return [np.random.choice([-1, 1]) for _ in range(N)]


    # Инициализация популяции
    def initialize_population():
        return [generate_sequence() for _ in range(P)]


    # Мутация (инверсия бита)
    def mutate(sequence):
        return [x * -1 if np.random.random() < Pm else x for x in sequence]


    # Скрещивание (одноточечное)
    def crossover(parent1, parent2):
        if np.random.random() < Pk:
            point = np.random.randint(1, N)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            return child1, child2
        return parent1, parent2


    # Селекция (рулетка)
    def roulette_selection(population, fitness_values):
        inverse_fitness = [1 / (f + 1e-6) for f in fitness_values]  # Чтобы минимизировать PSL
        total = sum(inverse_fitness)
        probs = [f / total for f in inverse_fitness]
        return population[np.random.choice(len(population), p=probs)]


    # Основной алгоритм
    def genetic_algorithm():
        population = initialize_population()
        best_sequences = []
        best_fitness = float('inf')

        for gen in range(100):  # Макс. число поколений
            # Оценка приспособленности
            fitness_values = [fitness(seq) for seq in population]

            # Проверка на соответствие PSL
            for i, f in enumerate(fitness_values):
                if f <= PSL and population[i] not in best_sequences:
                    best_sequences.append(population[i])
                    print(f"Найдена последовательность с PSL={f} в поколении {gen + 1}")
                    if len(best_sequences) >= K:
                        return best_sequences

            # Селекция
            if SELECTION_METHOD == "рулетка":
                selected = [roulette_selection(population, fitness_values) for _ in range(P)]
            else:
                raise ValueError("Неизвестный метод селекции")

            # Скрещивание и мутация
            new_population = []
            for i in range(0, P - 1, 2):
                child1, child2 = crossover(selected[i], selected[i + 1])
                new_population.extend([mutate(child1), mutate(child2)])
            population = new_population[:P]

        return best_sequences[:K]  # Возвращаем только K лучших


    # Запуск алгоритма
    print(f"\nПоиск {K} бинарных последовательностей длины {N} с PSL ≤ {PSL}...")
    best_sequences = genetic_algorithm()

    # Вывод результатов
    print("\nНайденные последовательности:")
    for i, seq in enumerate(best_sequences):
        acf = autocorrelation(seq)
        print(f"\nПоследовательность {i + 1}:")
        print("Код:", ''.join(['1' if x == 1 else '0' for x in seq]))
        print("PSL:", max([abs(x) for x in acf[1:]]))
        print("АКФ:", acf)

else:
    print("Ошибка: выбран неверный номер задачи.")