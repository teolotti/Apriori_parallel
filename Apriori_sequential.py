import itertools
import time
from collections import defaultdict
from turtledemo.penrose import start
import pandas as pd


# Carica le transazioni da file (una transazione per riga, item separati da spazio)
def load_transactions(path):
    trans_df = pd.read_csv(path, header=None)
    trans_df.drop(trans_df.columns[0], axis=1, inplace=True)
    transactions = []
    for transaction in trans_df.values.tolist():
        items = [str(item) for item in transaction if pd.notna(item) and str(item).lower() != 'nan' and 'item' not in str(item).lower()]
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

# Conta il supporto degli itemset nel dataset
def count_support(candidates, transactions):
    support = defaultdict(int)
    for transaction in transactions:
        for candidate in candidates:
            if candidate.issubset(transaction):
                support[candidate] += 1
    return support

# Filtra itemset con supporto >= minsup
def filter_frequent(support_count, minsup, n_transactions):
    return {
        itemset: count / n_transactions
        for itemset, count in support_count.items()
        if (count / n_transactions) >= minsup
    }

def apriori(transactions, minsup):
    n_transactions = len(transactions)
    items = sorted({item for transaction in transactions for item in transaction})
    L = []

    # Itemset di lunghezza 1
    candidates = [frozenset([item]) for item in items]
    k = 1
    while candidates:
        support_count = count_support(candidates, transactions)
        Lk = filter_frequent(support_count, minsup, n_transactions)
        if not Lk:
            break
        L.append(Lk)
        # Genera nuovi candidati di lunghezza k+1
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
    frequent_itemsets = apriori(transactions, minsup)
    end_time = time.time()
    print(f"Tempo di esecuzione: {end_time - start_time:.2f} secondi")
    for i, level in enumerate(frequent_itemsets):
        print(f"Livello {i+1} - {len(level)} itemset trovati")