import numpy as np
import random
import cv2
import math
import os
from tqdm import tqdm
from icecream import ic
from emb import Image1, Embeding, Extract
from attack import Attacks
import shutil


# Function


class abc:



    @staticmethod
    def target_function():
        return


    # Function: Initialize Variables

    @staticmethod
    def initial_sources(food_sources=3, min_values=[-5, -5],
                        max_values=[5, 5], target_function=target_function):
        sources = np.zeros((food_sources, len(min_values) + 1))
        for i in tqdm(range(0, food_sources), desc="Initial source", ascii=True, colour="yellow"):
            for j in range(0, len(min_values)):
                sources[i, j] = random.uniform(min_values[j], max_values[j])
            sources[i, -1] = target_function(sources[i, 0:sources.shape[1] - 1])
        return sources


    # Function: Fitness Value

    @staticmethod
    def fitness_calc(function_value):
        if (function_value >= 0):
            fitness_value = 1.0 / (1.0 + function_value)
        else:
            fitness_value = 1.0 + abs(function_value)
        return fitness_value

    @staticmethod
    def get_alpha_withABC(path_im, path_wm, size_im, size_wm):
        alphas = np.arange(0, 40, 1)
        res = []
        for alpha in alphas:
            im = Image1(path_im, size_im, size_im)
            wm = Image1(path_wm, size_wm, size_wm)
            prec = np.round(np.random.uniform(-1.5, 1.5), 2)
            metric_values = [Embeding.apply(im, wm, alpha + prec)]
            Attacks.execute(cv2.imread(im.emb_file, cv2.IMREAD_UNCHANGED))
            folder_path = './attack-result'
            file_list = os.listdir(folder_path)
            for file_name in file_list:
                metric_values.append(
                    Extract.apply("attack-result/" + file_name, im.size_x, im.size_y, file_name.split('.')[-1]))
            Embeding.clear_dir()
            shutil.rmtree('attack-result/')
            Fv = Image1.Fitness_calculate(metric_values)
            res.append([Fv, alpha + prec])
        a_opt = sorted(res)[0][1]
        return a_opt
    # Function: Fitness

    @staticmethod
    def fitness_function(searching_in_sources):
        fitness = np.zeros((searching_in_sources.shape[0], 2))
        for i in range(0, fitness.shape[0]):
            # fitness[i,0] = 1/(1+ searching_in_sources[i,-1] + abs(searching_in_sources[:,-1].min()))
            fitness[i, 0] = fitness_calc(searching_in_sources[i, -1])
        fit_sum = fitness[:, 0].sum()
        fitness[0, 1] = fitness[0, 0]
        for i in range(1, fitness.shape[0]):
            fitness[i, 1] = (fitness[i, 0] + fitness[i - 1, 1])
        for i in range(0, fitness.shape[0]):
            fitness[i, 1] = fitness[i, 1] / fit_sum
        return fitness


    # Function: Selection
    @staticmethod
    def roulette_wheel(fitness):
        ix = 0
        random = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
        for i in range(0, fitness.shape[0]):
            if (random <= fitness[i, 1]):
                ix = i
                break
        return ix


    # Function: Employed Bee

    @staticmethod
    def employed_bee(sources, min_values=[-5, -5], max_values=[5, 5], target_function=target_function):
        searching_in_sources = np.copy(sources)
        new_solution = np.zeros((1, len(min_values)))
        trial = np.zeros((sources.shape[0], 1))
        for i in (range(0, searching_in_sources.shape[0])):
            phi = random.uniform(-1, 1)
            j = np.random.randint(len(min_values), size=1)[0]
            k = np.random.randint(searching_in_sources.shape[0], size=1)[0]
            while i == k:
                k = np.random.randint(searching_in_sources.shape[0], size=1)[0]
            xij = searching_in_sources[i, j]
            xkj = searching_in_sources[k, j]
            vij = xij + phi * (xij - xkj)
            for variable in range(0, len(min_values)):
                new_solution[0, variable] = searching_in_sources[i, variable]
            new_solution[0, j] = np.clip(vij, min_values[j], max_values[j])
            new_function_value = target_function(
                new_solution[0, 0:new_solution.shape[1]])
            if (fitness_calc(new_function_value) > fitness_calc(searching_in_sources[i, -1])):
                searching_in_sources[i, j] = new_solution[0, j]
                searching_in_sources[i, -1] = new_function_value
            else:
                trial[i, 0] = trial[i, 0] + 1
            for variable in range(0, len(min_values)):
                new_solution[0, variable] = 0.0
        return searching_in_sources, trial


    # Function: Oulooker

    @staticmethod
    def outlooker_bee(searching_in_sources, fitness, trial, min_values=[-5, -5], max_values=[5, 5],
                      target_function=target_function):
        improving_sources = np.copy(searching_in_sources)
        new_solution = np.zeros((1, len(min_values)))
        trial_update = np.copy(trial)
        for repeat in range(0, improving_sources.shape[0]):
            i = roulette_wheel(fitness)
            phi = random.uniform(-1, 1)
            j = np.random.randint(len(min_values), size=1)[0]
            k = np.random.randint(improving_sources.shape[0], size=1)[0]
            while i == k:
                k = np.random.randint(improving_sources.shape[0], size=1)[0]
            xij = improving_sources[i, j]
            xkj = improving_sources[k, j]
            vij = xij + phi * (xij - xkj)
            for variable in range(0, len(min_values)):
                new_solution[0, variable] = improving_sources[i, variable]
            new_solution[0, j] = np.clip(vij, min_values[j], max_values[j])
            new_function_value = target_function(
                new_solution[0, 0:new_solution.shape[1]])
            if (fitness_calc(new_function_value) > fitness_calc(improving_sources[i, -1])):
                improving_sources[i, j] = new_solution[0, j]
                improving_sources[i, -1] = new_function_value
                trial_update[i, 0] = 0
            else:
                trial_update[i, 0] = trial_update[i, 0] + 1
            for variable in range(0, len(min_values)):
                new_solution[0, variable] = 0.0
        return improving_sources, trial_update


    # Function: Scouter

    @staticmethod
    def scouter_bee(improving_sources, trial_update, limit=3, target_function=target_function):
        for i in range(0, improving_sources.shape[0]):
            if (trial_update[i, 0] > limit):
                for j in range(0, improving_sources.shape[1] - 1):
                    improving_sources[i, j] = np.random.normal(0, 1, 1)[0]
                function_value = target_function(
                    improving_sources[i, 0:improving_sources.shape[1] - 1])
                improving_sources[i, -1] = function_value
        return improving_sources


    # ABC Function
    @staticmethod
    def artificial_bee_colony_optimization(food_sources=3, iterations=100, min_values=[-5, -5], max_values=[5, 5],
                                           employed_bees=3, outlookers_bees=3, limit=3, target_function=target_function):
        count = 0
        best_value = float("inf")
        sources = initial_sources(food_sources=food_sources, min_values=min_values,
                                  max_values=max_values, target_function=target_function)
        fitness = fitness_function(sources)
        while (count <= iterations):
            if (count > 0):
                ic.configureOutput(prefix="Bee >>")
                ic("Iteration =", count)
                ic("f(x) =", best_value)
            e_bee = employed_bee(sources, min_values=min_values,
                                 max_values=max_values, target_function=target_function)
            for i in range(0, employed_bees - 1):
                e_bee = employed_bee(e_bee[0], min_values=min_values,
                                     max_values=max_values, target_function=target_function)
            fitness = fitness_function(e_bee[0])
            o_bee = outlooker_bee(e_bee[0], fitness, e_bee[1], min_values=min_values,
                                  max_values=max_values, target_function=target_function)
            for i in range(0, outlookers_bees - 1):
                o_bee = outlooker_bee(o_bee[0], fitness, o_bee[1], min_values=min_values,
                                      max_values=max_values, target_function=target_function)
            value = np.copy(o_bee[0][o_bee[0][:, -1].argsort()][0, :])
            if (best_value > value[-1]):
                best_solution = np.copy(value)
                best_value = np.copy(value[-1])
            sources = scouter_bee(
                o_bee[0], o_bee[1], limit=limit, target_function=target_function)
            fitness = fitness_function(sources)
            count = count + 1
        best_solution = np.round(np.array([best_solution[0], best_solution[2]]), 5)
        output = f'args: {best_solution[:-1]}, func_value: {best_solution[-1]}'
        ic(output)
        return best_solution

    @staticmethod
    def corr(alpha):
        if alpha >= 15:
            return np.round(np.random.uniform(10, 14), 2)
        return alpha
    ######################## Использование ####################################

    # (Six Hump Camel Back). Функция ->  f(x1, x2) = -1.0316; x1 = 0.0898, x2 = -0.7126 or x1 = -0.0898, x2 = 0.7126

    # def six_hump_camel_back(variables_values=[0, 0]):
    #     func_value = 4*variables_values[0]**2 - 2.1*variables_values[0]**4 + (
    #         1/3)*variables_values[0]**6 + variables_values[0]*variables_values[1] - 4*variables_values[1]**2 + 4*variables_values[1]**4
    #     return func_value


    # abc = artificial_bee_colony_optimization(food_sources=20, iterations=10, min_values=[-5, -5], max_values=[
    #                                          5, 5], employed_bees=20, outlookers_bees=20, limit=40, target_function=six_hump_camel_back)

    # (Rosenbrocks Valley). Функция ->  f(x) = 0; xi = 1
    @staticmethod
    def rosenbrocks_valley(variables_values=[0]):
        func_value = 0
        last_x = variables_values[0]
        for i in range(1, len(variables_values)):
            func_value = func_value + \
                         -2 * last_x + math.exp(last_x) + 1
        return func_value


#abc = ABC.artificial_bee_colony_optimization(food_sources=350, iterations=10, min_values=[-5, -5], max_values=[
#                                             5, 5], employed_bees=40, outlookers_bees=40, limit=80, target_function=rosenbrocks_valley)

