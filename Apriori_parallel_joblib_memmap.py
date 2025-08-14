import itertools
import time
from collections import defaultdict
import pandas as pd
from joblib import Parallel, delayed, dump, load
import math
import os

def load_transactions(path):
    """Carica dataset CSV in lista di frozenset"""
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

def save_transactions_memmap(transactions, filename="transactions.pkl"):
    """Salva le transazioni in formato joblib memmap"""
    dump(transactions, filename)
    return filename


def load_transactions_memmap(filename="transactions.pkl"):
    """Carica transazioni con memory mapping"""
    return load(filename, mmap_mode='r')


def support_worker_chunk(candidates_chunk, transactions):
    """Calcola il supporto per un chunk di candidati"""
    support = defaultdict(int)
    for transaction in transactions:
        for candidate in candidates_chunk:
            if candidate.issubset(transaction):
                support[candidate] += 1
    return support


def count_support_joblib_chunks(candidates, transactions, n_jobs, chunk_size):
    """Conta il supporto usando joblib su chunk di candidati"""
    chunks = [candidates[i:i + chunk_size] for i in range(0, len(candidates), chunk_size)]
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(support_worker_chunk)(chunk, transactions) for chunk in chunks
    )
    merged = defaultdict(int)
    for partial in results:
        for itemset, count in partial.items():
            merged[itemset] += count
    return merged


def filter_frequent(support_count, minsup, n_transactions):
    return {
        itemset: count / n_transactions
        for itemset, count in support_count.items()
        if (count / n_transactions) >= minsup
    }


def apriori_joblib_memmap(transactions, minsup, n_jobs=4, chunk_size=1000):
    n_transactions = len(transactions)
    items = sorted({item for transaction in transactions for item in transaction})
    L = []

    candidates = [frozenset([item]) for item in items]
    k = 1
    while candidates:
        support_count = count_support_joblib_chunks(candidates, transactions, n_jobs, chunk_size)
        Lk = filter_frequent(support_count, minsup, n_transactions)
        if not Lk:
            break
        L.append(Lk)
        prev_frequent = list(Lk.keys())
        candidates = list(set([i.union(j) for i in prev_frequent for j in prev_frequent if len(i.union(j)) == k + 1]))
        k += 1
    return L


if __name__ == "__main__":
    dataset_csv = "retail_long.csv"
    memmap_file = "transactions.pkl"

    # Se non esiste, crea memmap
    if not os.path.exists(memmap_file):
        print("📥 Caricamento CSV e salvataggio in memmap...")
        trans_list = load_transactions_from_long(dataset_csv)
        save_transactions_memmap(trans_list, memmap_file)

    # Carica da memmap
    transactions_memmap = load_transactions_memmap(memmap_file)

    minsup = 0.02
    start_time = time.time()
    frequent_itemsets = apriori_joblib_memmap(transactions_memmap, minsup, n_jobs=8, chunk_size=2000)
    end_time = time.time()

    print(f"⏱ Tempo di esecuzione: {end_time - start_time:.2f} secondi")
    for i, level in enumerate(frequent_itemsets):
        print(f"Livello {i+1} - {len(level)} itemset trovati")
