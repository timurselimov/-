import random
import numpy as np
from tabulate import tabulate
import pandas as pd

# --------------------------
# Параметры вариантов (пример первых двух, остальные заполняете по таблицам)
# --------------------------
variants_main = {
    1: {"function": lambda x: 2 * x - 3.3 * np.cos(9.3 * x) - 2.3 * np.sin(1.7 * x),
        "extremum": "min", "coding": "binary", "Pc": 0.9, "Pm": 0.2, "selection": "best_with_best",
        "mating": "tournament"},

    2: {"function": lambda x: np.exp(-0.5 * x ** 2) * np.sign(np.cos(9.3 * x - 1)),
        "extremum": "min", "coding": "binary", "Pc": 0.9, "Pm": 0.2, "selection": "best_with_best",
        "mating": "tournament"},

    3: {"function": lambda x: -0.6 * x + 5.3 * np.abs(np.cos(6.1 * x)) * np.cos(3.6 * x),
        "extremum": "min", "coding": "gray", "Pc": 0.9, "Pm": 0.1, "selection": "random", "mating": "roulette"},

    4: {"function": lambda x: -1.3 * np.sin(1.6 * x ** 2 - 0.3) * np.exp(-0.3 * x + 0.5),
        "extremum": "max", "coding": "binary", "Pc": 0.8, "Pm": 0.2, "selection": "random", "mating": "tournament"},

    5: {"function": lambda x: 0.9 * np.abs(np.sin(3.7 * x)) * np.cos(6.1 * x),
        "extremum": "max", "coding": "gray", "Pc": 0.85, "Pm": 0.15, "selection": "best_with_best",
        "mating": "roulette"},

    6: {"function": lambda x: 0.8 * x + 1.4 * np.cos(1.8 * x ** 2) * np.exp(0.4 * x),
        "extremum": "min", "coding": "binary", "Pc": 0.9, "Pm": 0.15, "selection": "random", "mating": "tournament"},

    7: {"function": lambda x: -np.sin(0.9 * x - 1) - np.sin(1.8 * x - 1) * np.cos(7.8 * x),
        "extremum": "max", "coding": "binary", "Pc": 0.85, "Pm": 0.1, "selection": "best_with_best",
        "mating": "roulette"},

    8: {"function": lambda x: 0.8 * x + 1.1 * x * np.sin(9.3 * x) - 0.7 * np.cos(0.8 * x),
        "extremum": "min", "coding": "binary", "Pc": 0.8, "Pm": 0.15, "selection": "best_with_best",
        "mating": "tournament"},

    9: {"function": lambda x: 0.2 * x - 1.1 * np.exp(-0.4 * x ** 2) * np.sign(np.cos(9.5 * x + 1.5)),
        "extremum": "min", "coding": "gray", "Pc": 0.9, "Pm": 0.1, "selection": "random", "mating": "roulette"},

    10: {"function": lambda x: 0.3 * x + x * np.cos(7.3 * x) - 0.7 * np.sin(1.3 * x),
         "extremum": "max", "coding": "binary", "Pc": 0.85, "Pm": 0.15, "selection": "best_with_best",
         "mating": "tournament"},

    11: {"function": lambda x: 1.5 * x + 3.5 * np.cos(2.1 * x ** 2 + 3) - 0.5 * (x ** 2),
         "extremum": "max", "coding": "gray", "Pc": 0.8, "Pm": 0.2, "selection": "random", "mating": "roulette"},

    12: {"function": lambda x: 0.1 * x - 1.7 * np.abs(np.sin(5.8 * x)) * np.cos(3.2 * x),
         "extremum": "min", "coding": "binary", "Pc": 0.8, "Pm": 0.15, "selection": "random", "mating": "tournament"},

    13: {"function": lambda x: 5.1 * np.abs(np.cos(15 * x)) * np.cos(33 * x),
         "extremum": "max", "coding": "binary", "Pc": 0.85, "Pm": 0.15, "selection": "random", "mating": "roulette"},

    14: {"function": lambda x: 3.5 * x + np.sin(47 * x ** 2 + 2) - 6 * (x ** 2),
         "extremum": "max", "coding": "gray", "Pc": 0.8, "Pm": 0.1, "selection": "best_with_best",
         "mating": "tournament"},

    15: {"function": lambda x: 9 * x - 9.7 * np.abs(np.sin(37 * x)) * np.cos(27 * x),
         "extremum": "min", "coding": "binary", "Pc": 0.9, "Pm": 0.2, "selection": "random", "mating": "roulette"},

    16: {"function": lambda x: 3.1 * x - 3 * np.sin(31.4 * x ** 2 + 7) - 6 * (x ** 2),
         "extremum": "max", "coding": "gray", "Pc": 0.85, "Pm": 0.2, "selection": "random", "mating": "tournament"},

    17: {"function": lambda x: 6.9 * x + 7.5 * np.abs(np.cos(39 * x)) * np.cos(31 * x),
         "extremum": "max", "coding": "binary", "Pc": 0.8, "Pm": 0.1, "selection": "random", "mating": "roulette"},

    18: {"function": lambda x: 0.7 * x + 1.4 * x * np.sin(43 * x) - 2.3 * np.cos(5.4 * x),
         "extremum": "max", "coding": "gray", "Pc": 0.9, "Pm": 0.15, "selection": "best_with_best",
         "mating": "tournament"},

    19: {"function": lambda x: -2.3 * np.sin(36 * x ** 2) * np.exp(1.3 * x),
         "extremum": "min", "coding": "binary", "Pc": 0.8, "Pm": 0.1, "selection": "random", "mating": "roulette"},

    20: {"function": lambda x: 3.5 * x - 4.1 * x * np.cos(39 * x) + 2.8 * np.sin(4.6 * x),
         "extremum": "max", "coding": "gray", "Pc": 0.85, "Pm": 0.15, "selection": "random", "mating": "tournament"},

    21: {"function": lambda x: -0.6 * np.sin(4.3 * x + 2) - 3.9 * np.cos(5.3 * x - 4) * np.sin(38 * x),
         "extremum": "max", "coding": "binary", "Pc": 0.8, "Pm": 0.1, "selection": "best_with_best",
         "mating": "roulette"}
}

variants_extra = {
    1: {"N": 25, "P": 50, "PSL": 2, "K": 4, "Pc": 0.85, "Pm": 0.15, "selection": "roulette", "mating": "random"},
    2: {"N": 27, "P": 50, "PSL": 2, "K": 4, "Pc": 0.90, "Pm": 0.15, "selection": "tournament", "mating": "random"},
    3: {"N": 29, "P": 50, "PSL": 2, "K": 4, "Pc": 0.80, "Pm": 0.10, "selection": "roulette", "mating": "best_with_best"},
    4: {"N": 31, "P": 70, "PSL": 2, "K": 4, "Pc": 0.85, "Pm": 0.15, "selection": "tournament", "mating": "best_with_best"},
    5: {"N": 33, "P": 70, "PSL": 3, "K": 3, "Pc": 0.90, "Pm": 0.20, "selection": "roulette", "mating": "random"},
    6: {"N": 35, "P": 70, "PSL": 3, "K": 3, "Pc": 0.80, "Pm": 0.20, "selection": "roulette", "mating": "best_with_best"},
    7: {"N": 37, "P": 70, "PSL": 4, "K": 4, "Pc": 0.90, "Pm": 0.10, "selection": "tournament", "mating": "random"},
    8: {"N": 39, "P": 70, "PSL": 4, "K": 3, "Pc": 0.85, "Pm": 0.15, "selection": "roulette", "mating": "random"},
    9: {"N": 26, "P": 50, "PSL": 2, "K": 4, "Pc": 0.80, "Pm": 0.15, "selection": "tournament", "mating": "random"},
    10: {"N": 28, "P": 50, "PSL": 2, "K": 4, "Pc": 0.85, "Pm": 0.15, "selection": "roulette", "mating": "random"},
    11: {"N": 30, "P": 50, "PSL": 2, "K": 4, "Pc": 0.90, "Pm": 0.10, "selection": "tournament", "mating": "best_with_best"},
    12: {"N": 32, "P": 70, "PSL": 3, "K": 3, "Pc": 0.80, "Pm": 0.20, "selection": "roulette", "mating": "random"},
    13: {"N": 34, "P": 70, "PSL": 3, "K": 3, "Pc": 0.80, "Pm": 0.10, "selection": "roulette", "mating": "best_with_best"},
    14: {"N": 36, "P": 70, "PSL": 4, "K": 3, "Pc": 0.90, "Pm": 0.15, "selection": "tournament", "mating": "random"},
    15: {"N": 38, "P": 70, "PSL": 4, "K": 3, "Pc": 0.85, "Pm": 0.15, "selection": "roulette", "mating": "random"},
    16: {"N": 29, "P": 60, "PSL": 2, "K": 4, "Pc": 0.90, "Pm": 0.20, "selection": "tournament", "mating": "random"},
    17: {"N": 31, "P": 70, "PSL": 2, "K": 4, "Pc": 0.80, "Pm": 0.10, "selection": "roulette", "mating": "random"},
    18: {"N": 33, "P": 70, "PSL": 3, "K": 3, "Pc": 0.80, "Pm": 0.15, "selection": "tournament", "mating": "best_with_best"},
    19: {"N": 35, "P": 70, "PSL": 3, "K": 3, "Pc": 0.90, "Pm": 0.15, "selection": "tournament", "mating": "random"},
    20: {"N": 37, "P": 70, "PSL": 4, "K": 3, "Pc": 0.80, "Pm": 0.20, "selection": "roulette", "mating": "best_with_best"},
    21: {"N": 39, "P": 70, "PSL": 4, "K": 3, "Pc": 0.90, "Pm": 0.20, "selection": "tournament", "mating": "best_with_best"}
}

# --------------------------
# Выбор задачи и вариантов с защитой ввода
# --------------------------
def choose_task():
    while True:
        print("\nВыберите задачу:")
        print("1 — Поиск экстремума целевой функции")
        print("2 — Поиск кодовой последовательности по АКФ")
        choice = input("Ваш выбор (1/2): ").strip()
        if choice in ("1", "2"):
            return choice
        print("❌ Некорректно, введите 1 или 2.")

def choose_variant_main():
    while True:
        print("\nВарианты основной задачи:")
        print(tabulate([[k] for k in variants_main.keys()],
                       headers=["Номер варианта"], tablefmt="fancy_grid"))
        choice = input("Введите номер варианта (1–21): ").strip()
        try:
            v = int(choice)
            if v in variants_main:
                return v
        except ValueError:
            pass
        print("❌ Некорректный номер, попробуйте снова.")

def choose_variant_extra():
    while True:
        print("\nВарианты дополнительной задачи:")
        print(tabulate([[k] for k in variants_extra.keys()],
                       headers=["Номер варианта"], tablefmt="fancy_grid"))
        choice = input("Введите номер варианта (1–21): ").strip()
        try:
            v = int(choice)
            if v in variants_extra:
                return v
        except ValueError:
            pass
        print("❌ Некорректный номер, попробуйте снова.")

# --------------------------
# Генетический алгоритм для экстремума функции
# --------------------------
class GeneticAlgorithm:
    def __init__(self, func, extremum, coding, Pc, Pm, population_size=50, chromosome_length=20, search_range=(-5,5)):
        self.func = func
        self.extremum = extremum
        self.coding = coding
        self.Pc = Pc
        self.Pm = Pm
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.search_range = search_range
        self.population = self.initialize_population()
        self.history = []

    def initialize_population(self):
        return [''.join(random.choice('01') for _ in range(self.chromosome_length))
                for _ in range(self.population_size)]

    def decode(self, chrom):
        val = int(chrom, 2)
        lo, hi = self.search_range
        return lo + (hi - lo) * val / (2**self.chromosome_length - 1)

    def fitness(self, chrom):
        x = self.decode(chrom)
        y = self.func(x)
        return -y if self.extremum == "min" else y

    def selection(self):
        fit = np.array([self.fitness(c) for c in self.population])
        min_fit = fit.min()
        if min_fit < 0:
            fit = fit - min_fit
        total = fit.sum()
        if total == 0:
            probs = np.full_like(fit, 1/len(fit), dtype=float)
        else:
            probs = fit / total
        return np.random.choice(self.population, p=probs), np.random.choice(self.population, p=probs)

    def crossover(self, p1, p2):
        if random.random() < self.Pc:
            pt = random.randint(1, self.chromosome_length-1)
            return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]
        return p1, p2

    def mutation(self, chrom):
        return ''.join(
            ('1' if b=='0' else '0') if random.random() < self.Pm else b
            for b in chrom
        )

    def run(self, generations=100):
        for _ in range(generations):
            self.history.append(min(self.fitness(c) for c in self.population))
            new_pop = []
            while len(new_pop) < self.population_size:
                a, b = self.selection()
                c1, c2 = self.crossover(a, b)
                new_pop += [self.mutation(c1), self.mutation(c2)]
            self.population = new_pop[:self.population_size]
        best = min(self.population, key=self.fitness)
        return self.decode(best)

# --------------------------
# Генетический алгоритм для кодовых последовательностей
# --------------------------
class GeneticCodeSequence:
    def __init__(self, N, P, PSL, K, Pc, Pm):
        self.N, self.P, self.PSL, self.K, self.Pc, self.Pm = N, P, PSL, K, Pc, Pm
        self.population = self.initialize_population()
        self.solutions = []

    def initialize_population(self):
        return [''.join(random.choice('01') for _ in range(self.N)) for _ in range(self.P)]

    def seq_to_arr(self, seq):
        return np.array([1 if bit=='1' else -1 for bit in seq])

    def autocorr(self, arr):
        res = np.correlate(arr, arr, mode='full')
        mid = len(res)//2
        sid = np.concatenate((res[:mid], res[mid+1:])) / self.N
        return sid

    def fitness(self, seq):
        return -np.max(np.abs(self.autocorr(self.seq_to_arr(seq))))

    def selection(self):
        fits = np.array([self.fitness(s) for s in self.population])
        min_f = fits.min()
        if min_f < 0:
            fits = fits - min_f
        s = fits.sum()
        probs = np.full_like(fits, 1/len(fits), dtype=float) if s == 0 else fits/s
        return np.random.choice(self.population, p=probs), np.random.choice(self.population, p=probs)

    def crossover(self, p1, p2):
        if random.random() < self.Pc:
            pt = random.randint(1, self.N-1)
            return p1[:pt]+p2[pt:], p2[:pt]+p1[pt:]
        return p1, p2

    def mutation(self, seq):
        return ''.join(('1' if b=='0' else '0') if random.random()<self.Pm else b for b in seq)

    def run(self, generations=200):
        for _ in range(generations):
            new_pop = []
            while len(new_pop) < self.P:
                a, b = self.selection()
                c1, c2 = self.crossover(a, b)
                new_pop += [self.mutation(c1), self.mutation(c2)]
            self.population = new_pop[:self.P]
            for seq in self.population:
                if -self.fitness(seq) <= self.PSL and seq not in self.solutions:
                    self.solutions.append(seq)
                    if len(self.solutions) >= self.K:
                        return self.solutions
        return self.solutions

# --------------------------
# Вспомогательные выводы и сохранения
# --------------------------
def save_extremum(x, y):
    with open("extremum_result.txt","w") as f:
        f.write(f"x = {x:.5f}\ny = {y:.5f}\n")
    print("✅ Сохранено в extremum_result.txt")

def save_codes_txt(codes):
    with open("code_sequences.txt","w") as f:
        for i, s in enumerate(codes,1):
            f.write(f"{i}: {s}\n")
    print("✅ Сохранено в code_sequences.txt")

def save_codes_xlsx(codes):
    try:
        pd.DataFrame({'№':range(1,len(codes)+1),'seq':codes}) \
          .to_excel("code_sequences.xlsx", index=False)
        print("✅ Сохранено в code_sequences.xlsx")
    except ModuleNotFoundError:
        print("⚠️ Модуль openpyxl не установлен. Установите его: pip install openpyxl")



# --------------------------
# Главный блок
# --------------------------
if __name__ == "__main__":
    task = choose_task()

    if task == "1":
        v = choose_variant_main()
        vm = variants_main[v]
        ga = GeneticAlgorithm(vm["function"], vm["extremum"], vm["coding"],
                              vm["Pc"], vm["Pm"])
        best_x = ga.run(100)
        best_y = vm["function"](best_x)
        print(f"\n🔍 Найден экстремум: x = {best_x:.5f}, y = {best_y:.5f}\n")
        save_extremum(best_x, best_y)


    else:
        v = choose_variant_extra()
        ve = variants_extra[v]
        cs = GeneticCodeSequence(ve["N"], ve["P"], ve["PSL"],
                                 ve["K"], ve["Pc"], ve["Pm"])
        sols = cs.run(500)
        if sols:
            print("\n💡 Найденные коды:")
            print(tabulate([[i+1,s] for i,s in enumerate(sols)],
                           headers=["№","Sequence"], tablefmt="grid"))
            save_codes_txt(sols)
            save_codes_xlsx(sols)
        else:
            print("❌ Не удалось найти нужные последовательности.")
