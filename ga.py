import numpy as np
import matplotlib.pyplot as plt


# 二进制转十进制
def binary2decimal(binary):
    rows, cols = binary.shape
    decimal = np.sum(binary * np.power(2, np.arange(cols)[::-1]), axis=1)
    return decimal * 10 / 1023


# 计算适应度函数
def cal_objvalue(pop):
    x = binary2decimal(pop)
    objvalue = 10 * np.sin(5 * x) + 7 * np.abs(x - 5) + 10
    return objvalue


# 初始化种群
def initpop(popsize, chromlength):
    return np.round(np.random.rand(popsize, chromlength))


# 选择
def selection(pop, fitvalue):
    totalfit = sum(fitvalue)
    p_fitvalue = fitvalue / totalfit
    p_fitvalue = np.cumsum(p_fitvalue)
    popsize = len(pop)
    index = np.searchsorted(p_fitvalue, np.random.rand(popsize))
    newpop = pop[index]
    return newpop


# 交叉
def crossover(pop, pc):
    popsize, chromlength = pop.shape
    newpop = np.ones_like(pop)
    for i in range(0, popsize, 2):
        if np.random.rand() < pc:
            cpoint = np.random.randint(1, chromlength)
            newpop[i, :cpoint], newpop[i + 1, :cpoint] = (
                pop[i, :cpoint],
                pop[i + 1, :cpoint],
            )
            newpop[i, cpoint:], newpop[i + 1, cpoint:] = (
                pop[i + 1, cpoint:],
                pop[i, cpoint:],
            )
        else:
            newpop[i, :], newpop[i + 1, :] = pop[i, :], pop[i + 1, :]
    return newpop


# 变异
def mutation(pop, pm):
    popsize, chromlength = pop.shape
    newpop = np.copy(pop)
    for i in range(popsize):
        if np.random.rand() < pm:
            mpoint = np.random.randint(chromlength)
            newpop[i, mpoint] = 1 - newpop[i, mpoint]
    return newpop


# 寻找最优解
def best(pop, fitvalue):
    bestindex = np.argmax(fitvalue)
    bestindividual = pop[bestindex, :]
    bestfit = fitvalue[bestindex]
    return bestindividual, bestfit


# 主函数
def main():
    # 参数设置
    popsize = 1000
    chromlength = 10
    pc = 0.6
    pm = 0.01
    generations = 500

    # 初始化种群
    pop = initpop(popsize, chromlength)
    best_solutions = []
    for i in range(generations):
        objvalue = cal_objvalue(pop)
        fitvalue = objvalue
        newpop = selection(pop, fitvalue)
        newpop = crossover(newpop, pc)
        newpop = mutation(newpop, pm)
        pop = newpop
        bestindividual, bestfit = best(pop, fitvalue)
        best_solutions.append((bestindividual, bestfit))

    # 绘制所有最优解和突出显示最终最优解
    plt.figure()
    x = np.linspace(0, 10, 100)
    y = 10 * np.sin(5 * x) + 7 * np.abs(x - 5) + 10
    plt.plot(x, y, "b-")
    for solution in best_solutions[:-1]:
        x2 = binary2decimal(np.array([solution[0]]))
        y2 = cal_objvalue(np.array([solution[0]]))
        plt.plot(x2, y2, "g*")
    # 最终最优解
    final_best = best_solutions[-1]
    x_final = binary2decimal(np.array([final_best[0]]))
    y_final = cal_objvalue(np.array([final_best[0]]))
    plt.plot(x_final, y_final, "r*", markersize=10)
    plt.title("Genetic Algorithm Optimization")
    plt.show()
    plt.savefig("ga.jpg")

    print(f"The best X is --->> {binary2decimal(np.array([final_best[0]]))[0]:.2f}")
    print(f"The best Y is --->> {final_best[1]:.2f}")


if __name__ == "__main__":
    main()
