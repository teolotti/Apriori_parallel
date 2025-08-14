import multiprocessing
import time
from collections import defaultdict
import pandas as pd

_transactions = None

def init_worker(transactions):
    global _transactions
    _transactions = transactions

# Carica le transazioni da file
def load_transactions(path):
    trans_df = pd.read_csv(path, header=None)
    trans_df.drop(trans_df.columns[0], axis=1, inplace=True)
    transactions = []
    for transaction in trans_df.values.tolist():
        items = [str(item) for item in transaction if
                 pd.notna(item) and str(item).lower() != 'nan' and 'item' not in str(item).lower()]
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


# Funzione di supporto per il processo: calcola il supporto per una parte dei candidati
def support_worker(candidates_chunk):
    support = defaultdict(int)
    for transaction in _transactions:
        for candidate in candidates_chunk:
            if candidate.issubset(transaction):
                support[candidate] += 1
    return support

# Conta supporto in parallelo
def count_support_parallel(candidates, transactions, n_processes):
    chunk_size = len(candidates) // n_processes + 1
    chunks = [candidates[i:i + chunk_size] for i in range(0, len(candidates), chunk_size)]
    with multiprocessing.Pool(processes=n_processes, initializer=init_worker, initargs=(transactions,)) as pool:
        results = pool.map(support_worker, chunks)
    # Merge dei dizionari
    merged = defaultdict(int)
    for partial in results:
        for itemset, count in partial.items():
            merged[itemset] += count
    return merged

# Filtra itemset frequenti
def filter_frequent(support_count, minsup, n_transactions):
    return {
        itemset: count / n_transactions
        for itemset, count in support_count.items()
        if (count / n_transactions) >= minsup
    }

def apriori_parallel(transactions, minsup, n_processes=4):
    n_transactions = len(transactions)
    items = sorted({item for transaction in transactions for item in transaction})
    L = []

    candidates = [frozenset([item]) for item in items]
    k = 1
    while candidates:
        support_count = count_support_parallel(candidates, transactions, n_processes)
        Lk = filter_frequent(support_count, minsup, n_transactions)
        if not Lk:
            break
        L.append(Lk)
        prev_frequent = list(Lk.keys())
        candidates = [i.union(j) for i in prev_frequent for j in prev_frequent if len(i.union(j)) == k + 1]
        candidates = list(set(candidates))
        k += 1
    return L

if __name__ == '__main__':
    # transactions = load_transactions('groceries - groceries.csv')
    transactions = load_transactions_from_long('retail_long.csv')  # Per testare con il dataset Kosarak
    minsup = 0.02
    start_time = time.time()
    frequent_itemsets = apriori_parallel(transactions, minsup, n_processes=8)
    end_time = time.time()
    print(f"Tempo di esecuzione: {end_time - start_time:.2f} secondi")
    for i, level in enumerate(frequent_itemsets):
        print(f"Livello {i+1} - {len(level)} itemset trovati")