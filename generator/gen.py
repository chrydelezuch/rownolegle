import random

# Pobranie N z terminala
N = int(input("Podaj liczbę N: "))

# Nazwa pliku wyjściowego
filename = "dane.txt"

with open(filename, "w") as file:
    # Pierwsza linia: N
    file.write(f"{N}\n")

    # Kolejne N linii, każda z N losowymi liczbami float
    for _ in range(N):
        numbers = [f"{random.uniform(0, 100):.3f}" for _ in range(N)]
        file.write(" ".join(numbers) + "\n")

print(f"Dane zostały zapisane do pliku '{filename}'.")
