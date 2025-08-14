import itertools
import time
from collections import defaultdict
import pandas as pd
from joblib import Parallel, delayed

# Carica le transazioni
def load_transactions(path):
    trans_df = pd.read_csv(path, header=None)
    trans_df.drop(trans_df.columns[0], axis=1, inplace=True)
    transactions = []
    for transaction in trans_df.values.tolist():
        items = [str(item) for item in transaction
                 if pd.notna(item) and str(item).lower() != 'nan' and 'item' not in str(item).lower()]
        if items:
            transactions.append(frozenset(items))
    return transactions

def load_transactions_from_long(path):
    df = pd.read_csv(path)
    # Raggruppa per transaction id e crea un frozenset per ciascun gruppo
    transactions = [
        frozenset(items)
        for items in df.groupby("tid")["item"].apply(list)
    ]
    return transactions


# Conta il supporto di un singolo candidato
def support_single(candidate, transactions):
    return candidate, sum(1 for transaction in transactions if candidate.issubset(transaction))

# Conta supporto in parallelo con joblib
def count_support_joblib(candidates, transactions, n_jobs):
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(support_single)(candidate, transactions) for candidate in candidates
    )
    return dict(results)

# Filtra itemset frequenti
def filter_frequent(support_count, minsup, n_transactions):
    return {
        itemset: count / n_transactions
        for itemset, count in support_count.items()
        if (count / n_transactions) >= minsup
    }

# Apriori con joblib
def apriori_joblib(transactions, minsup, n_jobs=4):
    n_transactions = len(transactions)
    items = sorted({item for transaction in transactions for item in transaction})
    L = []

    candidates = [frozenset([item]) for item in items]
    k = 1
    while candidates:
        support_count = count_support_joblib(candidates, transactions, n_jobs)
        Lk = filter_frequent(support_count, minsup, n_transactions)
        if not Lk:
            break
        L.append(Lk)
        prev_frequent = list(Lk.keys())
        candidates = list(set([i.union(j) for i in prev_frequent for j in prev_frequent if len(i.union(j)) == k + 1]))
        k += 1
    return L

if __name__ == '__main__':
    # transactions = load_transactions('groceries - groceries.csv')
    transactions = load_transactions_from_long('retail_long.csv')  # Per testare con il dataset Kosarak
    minsup = 0.02
    start_time = time.time()
    frequent_itemsets = apriori_joblib(transactions, minsup, n_jobs=8)
    end_time = time.time()
    print(f"Tempo di esecuzione: {end_time - start_time:.2f} secondi")
    for i, level in enumerate(frequent_itemsets):
        print(f"Livello {i+1} - {len(level)} itemset trovati")
