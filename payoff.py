import matplotlib.pyplot as plt
import numpy as np


def pay_off(K):

    Long_call = []
    Short_call = []

    Long_put = []
    Short_put = []

    for St in range(0,100):

        Long_call_p = max(St-K,0)
        Long_call.append(Long_call_p)

        Short_call_p = min(K - St, 0)
        Short_call.append(Short_call_p)

        Long_put_p = max(K - St, 0)
        Long_put.append(Long_put_p)

        Short_put_p = min(St-K,0)
        Short_put.append(Short_put_p)


    St = np.arange(0,100)
    plt.plot(St, Long_call, label = 'long call')
    plt.plot(St, Short_call, label = 'short call')
    plt.plot(K,0, 'o', color = 'red')
    plt.xlabel('St')
    plt.ylabel('Profit')
    plt.xlim(0,100)
    plt.ylim(-50,50)
    plt.text(49, -5,'K')
    plt.legend()
    plt.savefig("Call", dpi=300)
    plt.show()

    plt.plot(St, Long_put, label = 'long put')
    plt.plot(St, Short_put, label = 'short put')
    plt.plot(K,0, 'o', color = 'red')
    plt.xlabel('St')
    plt.ylabel('Profit')
    plt.xlim(0,100)
    plt.ylim(-50,50)
    plt.text(49, -5,'K')
    plt.legend()
    plt.savefig("Put", dpi=300)
    plt.show()

    return


K = 50

pay_off(K)