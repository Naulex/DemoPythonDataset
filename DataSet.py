import pandas as pd
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import seaborn as sns

print("==============================================================")
print("Анализ Нью-Йоркского датасета.")
print("==============================================================")

try:
    print("Чтение таблицы...")
    df = pd.read_excel('DataSetNewYork.xlsx')
    print("...успешно.")
except IOError as e:
    print("...ошибка!")
    print("Файл \'DataSetNewYorkNaumov.xlsx\' не обнаружен. Нажмите любую кнопку для выхода.")
    input()
    sys.exit()

def seabornGraph():
    types = df.dropna(subset=['ENERGY STAR Score'])
    types = types['Primary Property Type - Self Selected'].value_counts()
    types = list(types[types.values > 100].index)
    for b_type in types:
        subset = df[df['Primary Property Type - Self Selected'] == b_type]
        sns.kdeplot(subset['ENERGY STAR Score'].dropna(),
                    label=b_type, shade=False, alpha=0.8)
    plt.xlabel('Energy Star score', size=14)
    plt.ylabel('Density', size=14)
    plt.title('Primary Property Type - Self Selected/Energy Star score', size=14)
    plt.show()

def modelTrain():
    print("==============================================================")
    print("Тренировка модели на: KNeighborsRegressor")
    X = df
    y = df['ENERGY STAR Score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    reg = KNeighborsRegressor(n_neighbors=3)
    reg.fit(X_train, y_train)
    print("Правильность на обучающем наборе: {:.2f}".format(reg.score(X_train, y_train)))
    print("Правильность на контрольном наборе: {:.2f}".format(reg.score(X_test, y_test)))

    x = int(input("Введите количество данных для прогноза - "))
    for i in range(x):
        a = random.randint(1, 5)
        b = random.randint(1, 22)
        c = random.randint(5, 100)
        d = random.randint(1, 100)
        e = random.uniform(10, 700)
        f = random.uniform(10, 700)
        g = random.uniform(2, 320)
        h = random.uniform(2, 320)
        i = random.uniform(0, 72)
        j = random.uniform(0, 3)
        k = random.uniform(10, 6500)
        l = random.uniform(0, 3400)

        print("=====")
        X_new = np.array([a, b, c, d, e, f, g, h, i, j, k, l]).reshape(1, -1)
        print(X_new)
        prediction = reg.predict(X_new)
        print("Прогноз: {}".format(prediction))
        print("=====")

def arithmetic_mean(A, B, C):
    sum = 0
    meanIcount = 0
    for i in range(df[A].size):
        if df[A].values[i] == C:
            meanIcount = meanIcount + 1
            print(meanIcount)
            sum = sum + df[B].values[i]
            print(sum)
            print('==================================')
    try:
        mean = sum / meanIcount
        return mean
    except:
        return 1

def distributionCreator():
    print('Выполняется расчёт, ждите...')
    distribution = []
    for i in range(df['ENERGY STAR Score'].size):
        distribution.append(df['ENERGY STAR Score'].values[i])
    plt.hist(distribution, bins=20, color='green')
    plt.title('Распределение по параметру Energy Star score', fontsize=14)
    plt.xlabel('Energy Star score', fontsize=14)
    plt.ylabel('Количество', fontsize=14)
    plt.grid(True)
    print('График построен! Закройте окно графика для продолжения.')
    plt.show()

def energoEfficiency():
    print('Выполняется расчёт, ждите...')
    average = []
    district = [1, 2, 3, 4, 5]
    for i in range(5):
        average.append(arithmetic_mean('Borough', 'ENERGY STAR Score', i+1))
    plt.bar(district, average, color='blue')
    plt.title('Средняя энергоэффективность по районам', fontsize=14)
    plt.xlabel('Районы', fontsize=14)
    plt.ylabel('Энергоэффективность', fontsize=14)
    plt.grid(True)
    print('График построен! Закройте окно графика для продолжения.')
    plt.show()

def yearEfficiency():
    print('Выполняется расчёт, ждите...')
    plt.plot(df['Total GHG Emissions (Metric Tons CO2e)'], df['ENERGY STAR Score'], 's', color='blue')
    plt.title('Total GHG Emissions (Metric Tons CO2e) / Энергоэффективность', fontsize=14)
    plt.xlabel('Total GHG Emissions', fontsize=14)
    plt.ylabel('ENERGY STAR Score', fontsize=14)
    plt.grid(True)
    print('График построен! Закройте окно графика для продолжения.')
    plt.show()

def EUIEfficiency():
    print('Выполняется расчёт, ждите...')
    plt.plot(df['Source EUI (kBtu/ft²)'], df['ENERGY STAR Score'], 's')
    plt.title('Source EUI (kBtu/ft²) / Энергоэффективность', fontsize=14)
    plt.xlabel('Source EUI (kBtu/ft²)', fontsize=14)
    plt.ylabel('ENERGY STAR Score', fontsize=14)
    plt.grid(True)
    print('График построен! Закройте окно графика для продолжения.')
    plt.show()

def weatherEUIEfficiency():
    print('Выполняется расчёт, ждите...')
    plt.plot(df['Weather Normalized Source EUI (kBtu/ft²)'], df['ENERGY STAR Score'], 's', color='green')
    plt.title('Weather Normalized Source EUI (kBtu/ft²) / Энергоэффективность', fontsize=14)
    plt.xlabel('Weather Normalized Source EUI (kBtu/ft²)', fontsize=14)
    plt.ylabel('ENERGY STAR Score', fontsize=14)
    plt.grid(True)
    print('График построен! Закройте окно графика для продолжения.')
    plt.show()

def siteEUIEfficiency():
    print('Выполняется расчёт, ждите...')
    plt.plot(df['Site EUI (kBtu/ft²)'], df['ENERGY STAR Score'], 's', color='blue')
    plt.title('Site EUI (kBtu/ft²) / Энергоэффективность', fontsize=14)
    plt.xlabel('Site EUI (kBtu/ft²)', fontsize=14)
    plt.ylabel('ENERGY STAR Score', fontsize=14)
    plt.grid(True)
    print('График построен! Закройте окно графика для продолжения.')
    plt.show()

def steamUseEfficiency():
    print('Выполняется расчёт, ждите...')
    plt.plot(df['Direct GHG Emissions (Metric Tons CO2e)'], df['ENERGY STAR Score'], 's')
    plt.title('Direct GHG Emissions (Metric Tons CO2e) / Энергоэффективность', fontsize=14)
    plt.xlabel('Direct GHG Emissions (Metric Tons CO2e)', fontsize=14)
    plt.ylabel('ENERGY STAR Score', fontsize=14)
    plt.grid(True)
    print('График построен! Закройте окно графика для продолжения.')
    plt.show()

def addiction():
    print('Выполняется расчёт, ждите...')
    plt.bar(df['Primary Property Type - Self Selected'], df['ENERGY STAR Score'])
    plt.title('ENERGY STAR Score / Primary Property Type - Self Selected', fontsize=14)
    plt.xlabel('Primary Property Type - Self Selected', fontsize=14)
    plt.ylabel('ENERGY STAR Score', fontsize=14)
    plt.grid(True)
    print('График построен! Закройте окно графика для продолжения.')
    plt.show()

def showMenu():
    print("==============================================================")
    print("Возможности:")
    print("1 - построить график: Энергоэффективность по районам.")
    print("2 - построить график: Total GHG Emissions (Metric Tons CO2e) / Энергоэффективность.")
    print("3 - построить график: Source EUI (kBtu/ft²) / Энергоэффективность.")
    print("4 - построить график: Weather Normalized Source EUI (kBtu/ft²) / Энергоэффективность.")
    print("5 - построить график: Site EUI (kBtu/ft²) / Энергоэффективность.")
    print("6 - построить график: Direct GHG Emissions (Metric Tons CO2e) / Энергоэффективность.")
    print("7 - построить график: распределение по параметру Energy Star score (AD).")
    print("8 - построить график: зависимость параметров Energy Star от типа здания Primary Property Type - Self Selected (Q).")
    print("9 - провести тренировку модели и вывести результаты.")
    print("10 - построить график: Primary Property Type - Self Selected/Energy Star score.")
    print("0 - выход.")
    print("Выберите действие: ")
    x = int(input())
    if x == 1:
        energoEfficiency()
        showMenu()
    elif x == 2:
        yearEfficiency()
        showMenu()
    elif x == 3:
        EUIEfficiency()
        showMenu()
    elif x == 4:
        weatherEUIEfficiency()
        showMenu()
    elif x == 5:
        siteEUIEfficiency()
        showMenu()
    elif x == 6:
        steamUseEfficiency()
        showMenu()
    elif x == 7:
        distributionCreator()
        showMenu()
    elif x == 8:
        addiction()
        showMenu()
    elif x == 9:
        modelTrain()
        showMenu()
    elif x == 10:
        seabornGraph()
        showMenu()
    else:
        sys.exit()

showMenu()

