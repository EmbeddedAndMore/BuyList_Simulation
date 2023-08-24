import multiprocessing
import pandas as pd

class Foo:
    def __init__(self):
        self.foo = None

    def proc_func(self, arg):
        l, ns, val = arg
        l.append(val)
        ns.df = pd.DataFrame(data={'col1': [10, 20], 'col2': [3, 4]})

def simulation(arg):
    foo = Foo()
    foo.proc_func(arg)




if __name__ == "__main__":
    # with multiprocessing.Manager()as manager:
    #     d = {'col1': [1, 2], 'col2': [3, 4]}
    #     df = pd.DataFrame(data=d)
    #     shared_list = manager.list()
    #     namespace = manager.Namespace()
    #     namespace.df = df

    #     with multiprocessing.Pool(5) as pool:
    #         pool.map(simulation, [(shared_list,namespace, 1), (shared_list,namespace,2), (shared_list,namespace,3)])


    #     print(namespace.df)
    d = {'col1': [1, 2], 'col2': [3, 4]}
    df = pd.DataFrame(data=d)
    res = df.loc[:,"col1"].sub([1,2])
    df["col1"] = res
    print(df)